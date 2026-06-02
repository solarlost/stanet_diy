import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers

# ==============================================================================
# EF-IB Net: Enhanced Fusion with Information Bottleneck & Cross-Modal Alignment
# ==============================================================================

def pearson_r(eeg, fnirs):
    mx = tf.math.reduce_mean(eeg, axis=1, keepdims=True)
    my = tf.math.reduce_mean(fnirs, axis=1, keepdims=True)
    xm, ym = eeg-mx, fnirs-my
    r_num = tf.math.reduce_mean(tf.multiply(xm,ym), axis=1)        
    r_den = tf.math.reduce_std(xm, axis=1) * tf.math.reduce_std(ym, axis=1) + 1e-6
    plcc = r_num / r_den
    plcc = tf.math.abs(plcc)
    plcc_meanbatch = tf.math.reduce_mean(plcc)
    return plcc_meanbatch

class pos_embedding(layers.Layer):
    def __init__(self, max_seq_len=256):
        super(pos_embedding, self).__init__()
        self.max_seq_len = max_seq_len

    def build(self, input_shape):
        channel_dim = input_shape[-1]
        self.channel_dim = int(channel_dim)
        self.pos_embedding = self.add_weight(
            name='pos_embedding',
            shape=(1, self.max_seq_len, self.channel_dim),
            initializer=tf.keras.initializers.HeUniform(),
            trainable=True
        )
        
    def call(self, inputs):
        seq_len = tf.shape(inputs)[1]
        pos_slice = self.pos_embedding[:, :seq_len, :]
        if inputs.shape.rank is not None:
            inputs.set_shape((None, None, self.channel_dim))
        return inputs + pos_slice

class e_f_attention(keras.layers.Layer):
    def __init__(self, emb_size, d_model, heads, drop, max_seq_len=256):
        super(e_f_attention, self).__init__()
        self.q_flat = layers.Flatten()
        self.q_proj = layers.Dense(emb_size)
        self.fusion_proj = layers.Dense(emb_size)
        self.k_proj = layers.Dense(emb_size)
        self.pos = pos_embedding(max_seq_len=max_seq_len)
        self.dot_product_attention = layers.MultiHeadAttention(num_heads=heads, key_dim=d_model, dropout=drop)
        self.ef_plcc_tracker = keras.metrics.Mean(name="ef_plcc")

    def call(self, inputs):
        eeg, fnirs = inputs
        q_eeg = self.q_flat(eeg)
        fusion_output = self.fusion_proj(q_eeg)
        q_eeg = self.q_proj(q_eeg)
        q_eeg = tf.expand_dims(q_eeg, axis=1) 
        
        fnirs_shape = tf.shape(fnirs)
        batch_size = fnirs_shape[0]
        channels = fnirs_shape[4]
        k_fnirs = tf.reshape(fnirs, [batch_size, -1, channels])
        
        k_fnirs = self.pos(k_fnirs)
        k_fnirs = self.k_proj(k_fnirs) 
        
        fnirs_weighted, attention_weights = self.dot_product_attention(q_eeg, k_fnirs, return_attention_scores=True) 
        attention_weights = tf.math.reduce_mean(attention_weights, axis=(1, 2))
        
        q_eeg = tf.math.reduce_mean(q_eeg, axis=1)
        fnirs_weighted = tf.math.reduce_mean(fnirs_weighted, axis=1)

        ef_loss = pearson_r(q_eeg, fnirs_weighted)
        self.add_loss(1-ef_loss)
        self.ef_plcc_tracker.update_state(ef_loss)
       
        return fusion_output, fnirs_weighted, attention_weights

class gap(keras.layers.Layer):
    def __init__(self):
        super(gap, self).__init__()
    def call(self, inputs):
        return tf.reduce_mean(inputs, axis=-2, keepdims=True)

class fga(keras.layers.Layer):
    def __init__(self, tem_kernel_size, fga_loss_name):
        super(fga, self).__init__()
        self.channel_pooling = layers.Conv3D(filters=1, kernel_size=(3, 3, tem_kernel_size), strides=(1, 1, 1), padding='same')
        self.tap_fnirs = gap()
        self.residual_para = self.add_weight(name='residual_para', initializer="zeros", trainable=True)
        self.add_eeg = layers.Add()
        self.add = layers.Add()
        self.eeg_flatten = layers.Flatten()
        self.fnirs_flatten = layers.Flatten()
        self.fga_loss_tracker = keras.metrics.Mean(name=fga_loss_name)

    def call(self, inputs):
        eeg_fusion, eeg, fnirs = inputs
        fnirs_attention = self.channel_pooling(fnirs) 
        fnirs_attention_map = self.tap_fnirs(fnirs_attention) 
        fnirs_attention_map_for_plcc = fnirs_attention_map
        fnirs_attention_map = tf.math.reduce_mean(fnirs_attention_map, axis=1, keepdims=True) 
        fnirs_attention_map_norm = keras.activations.sigmoid(fnirs_attention_map) 
        fnirs_attention_map_norm_for_plcc = keras.activations.sigmoid(fnirs_attention_map_for_plcc)

        eeg_fusion_guided = tf.math.multiply(eeg_fusion, fnirs_attention_map_norm) 
        residual_para_norm = keras.activations.sigmoid(self.residual_para)
        eeg_add = self.add_eeg([residual_para_norm*eeg, (1-residual_para_norm)*eeg_fusion]) 
        fga_feature = self.add([eeg_fusion_guided, eeg_add]) 

        eeg_plcc = tf.math.reduce_mean(eeg, axis=(-1, -2)) 
        eeg_plcc = self.eeg_flatten(eeg_plcc) 
        fnirs_attention_map_norm_plcc = tf.math.reduce_mean(fnirs_attention_map_norm_for_plcc, axis=(-1, -2))
        fnirs_attention_map_norm_plcc = self.fnirs_flatten(fnirs_attention_map_norm_plcc) 

        fga_loss = pearson_r(eeg_plcc, fnirs_attention_map_norm_plcc)
        self.add_loss(1-fga_loss)
        self.fga_loss_tracker.update_state(fga_loss)
        return fga_feature

class conv_block(keras.layers.Layer):
    def __init__(self, eeg_filter, eeg_size, eeg_stride,
                fnirs_filter, fnirs_size, fnirs_stride,
                eegfusion_filter, eegfusion_size, eegfusion_stride, 
                tem_kernel_size, fga_loss_name, padding):
        super(conv_block, self).__init__()
        self.eeg_conv = layers.Conv3D(filters=eeg_filter, kernel_size=eeg_size, strides=eeg_stride, padding=padding)
        self.eeg_act = layers.Activation('elu')
        self.eeg_bn = layers.BatchNormalization()

        self.fnirs_conv = layers.Conv3D(filters=fnirs_filter, kernel_size=fnirs_size, strides=fnirs_stride, padding=padding)
        self.fnirs_act = layers.Activation('elu')
        self.fnirs_bn = layers.BatchNormalization()

        self.eegfusion_conv = layers.Conv3D(filters=eegfusion_filter, kernel_size=eegfusion_size, strides=eegfusion_stride, padding=padding)
        self.eegfusion_act = layers.Activation('elu')
        self.eegfusion_bn = layers.BatchNormalization()

        self.fga = fga(tem_kernel_size, fga_loss_name)

    def call(self, inputs):
        eegfusion, eeg, fnirs = inputs
        eeg_feature = self.eeg_conv(eeg) 
        eeg_feature = self.eeg_bn(eeg_feature) 
        eeg_feature = self.eeg_act(eeg_feature) 

        fnirs_shape_tensor = tf.shape(fnirs)
        batch_size = fnirs_shape_tensor[0]
        fnirs_static_shape = fnirs.shape

        if fnirs_static_shape.ndims == 6:
            static_dims = fnirs_static_shape.as_list()
            _, dim1, dim2, dim3, dim4, dim5 = static_dims
            if None not in [dim1, dim2, dim3, dim4, dim5]:
                new_channels = dim1 * dim5
                reshape_tail = tf.constant([dim2, dim3, dim4, new_channels], dtype=tf.int32)
                new_shape = tf.concat([[batch_size], reshape_tail], axis=0)
                fnirs_reshaped = tf.reshape(fnirs, new_shape)
                fnirs_reshaped.set_shape((None, dim2, dim3, dim4, new_channels))
            else:
                dim1 = fnirs_shape_tensor[1]
                dim2 = fnirs_shape_tensor[2]
                dim3 = fnirs_shape_tensor[3]
                dim4 = fnirs_shape_tensor[4]
                dim5 = fnirs_shape_tensor[5]
                new_channels = dim1 * dim5
                new_shape = tf.stack([batch_size, dim2, dim3, dim4, new_channels])
                fnirs_reshaped = tf.reshape(fnirs, new_shape)
        else:
            fnirs_reshaped = fnirs
        
        fnirs_feature = self.fnirs_conv(fnirs_reshaped) 
        fnirs_feature = self.fnirs_bn(fnirs_feature) 
        fnirs_feature = self.fnirs_act(fnirs_feature) 

        eegfusion_feature = self.eegfusion_conv(eegfusion) 
        eegfusion_feature = self.eegfusion_bn(eegfusion_feature) 
        eegfusion_feature = self.eegfusion_act(eegfusion_feature) 

        eegfusion_fga = self.fga((eegfusion_feature, eeg_feature, fnirs_feature)) 
        return eegfusion_fga, eeg_feature, fnirs_feature

class reduce_sum_layer(keras.layers.Layer):
    def __init__(self, axis, keepaxis, name=None, *args, **kwargs):
        super(reduce_sum_layer, self).__init__(name=name)
        self.axis = axis
        self.keepaxis = keepaxis
    def call(self, inputs):
        return tf.math.reduce_sum(inputs, axis=self.axis, keepdims=self.keepaxis)

class expand_dims_layer(keras.layers.Layer):
    def __init__(self, axis, *args, **kwargs):
        super(expand_dims_layer, self).__init__()
        self.axis = axis
    def call(self, inputs):
        return tf.expand_dims(inputs, axis=self.axis)

class InformationBottleneck(layers.Layer):
    def __init__(self, latent_dim, beta=1e-3, name=None):
        super(InformationBottleneck, self).__init__(name=name)
        self.latent_dim = latent_dim
        self.beta = beta
        self.dense_mu = layers.Dense(latent_dim, name=f"{self.name}_mu")
        self.dense_logvar = layers.Dense(latent_dim, name=f"{self.name}_logvar")
        metric_name = f"{self.name}_kl" if self.name else "ib_kl"
        self.kl_tracker = keras.metrics.Mean(name=metric_name)

    def call(self, inputs, training=None):
        from keras.mixed_precision import global_policy
        dtype = global_policy().compute_dtype 

        mu = self.dense_mu(inputs)
        logvar = self.dense_logvar(inputs)
        logvar = tf.clip_by_value(logvar, -10.0, 10.0)

        mu = tf.cast(mu, dtype)
        logvar = tf.cast(logvar, dtype)

        if training or training is None:
            std = tf.exp(0.5 * logvar)
            eps = tf.random.normal(tf.shape(mu), dtype=dtype)
            z = mu + std * eps
        else:
            z = mu

        kl_div = -0.5 * tf.reduce_mean(1 + logvar - tf.square(mu) - tf.exp(logvar))
        self.add_loss(self.beta * kl_div)
        self.kl_tracker.update_state(kl_div)
        return z

class AlignmentLoss(layers.Layer):
    """
    跨模态对齐损失层
    计算两个潜在表示之间的距离，并作为辅助损失添加到模型中。
    """
    def __init__(self, weight=0.1, name="alignment_loss"):
        super(AlignmentLoss, self).__init__(name=name)
        self.weight = weight
        self.loss_tracker = keras.metrics.Mean(name="align_loss")

    def call(self, inputs):
        z1, z2 = inputs
        # 使用 MSE (均方误差) 来衡量两个模态潜在向量的一致性
        # 也可以尝试 Cosine Similarity
        loss = tf.reduce_mean(tf.square(z1 - z2))
        self.add_loss(self.weight * loss)
        self.loss_tracker.update_state(loss)
        return loss

def efib_net(latent_dim=128, beta=1e-3, align_weight=0.1):
    """
    Enhanced Fusion IB Net (EF-IB Net)
    包含:
    1. Information Bottleneck (IB) 用于去噪
    2. Alignment Loss 用于跨模态特征对齐
    """
    eeg_input = keras.Input(shape=(16, 16, 600, 1), name="eeg_input")
    fnirs_input = keras.Input(shape=(11, 16, 16, 30, 2), name="fnirs_input")

    # --- Encoder Blocks ---
    eegfusion1, eeg1, fnirs1 = conv_block(eeg_filter=16, eeg_size=(2, 2, 13), eeg_stride=(2, 2, 6),
                                          fnirs_filter=16, fnirs_size=(2, 2, 5), fnirs_stride=(2, 2, 2),
                                          eegfusion_filter=16, eegfusion_size=(2, 2, 13), eegfusion_stride=(2, 2, 6),
                                          tem_kernel_size=5, fga_loss_name='fgsa1_plcc', padding='same')((eeg_input, eeg_input, fnirs_input))
    eegfusion1 = layers.Dropout(0.5)(eegfusion1)
    eeg1 = layers.Dropout(0.5)(eeg1)
    fnirs1 = layers.Dropout(0.5)(fnirs1)

    eegfusion2, eeg2, fnirs2 = conv_block(eeg_filter=32, eeg_size=(2, 2, 5), eeg_stride=(2, 2, 2),
                                          fnirs_filter=32, fnirs_size=(2, 2, 3), fnirs_stride=(2, 2, 2),
                                          eegfusion_filter=32, eegfusion_size=(2, 2, 5), eegfusion_stride=(2, 2, 2),
                                          tem_kernel_size=3, fga_loss_name='fgsa2_plcc', padding='same')((eegfusion1, eeg1, fnirs1))
    eegfusion2 = gap()(eegfusion2)
    eeg2 = gap()(eeg2)
    fnirs2 = gap()(fnirs2)

    eegfusion2 = layers.Dropout(0.5)(eegfusion2)
    eeg2 = layers.Dropout(0.5)(eeg2)
    fnirs2 = layers.Dropout(0.5)(fnirs2)

    # --- Attention Fusion ---
    eegfusion_feature, fnirs_feature, _ = e_f_attention(emb_size=256, d_model=256, heads=10, drop=0.5)((eegfusion2, fnirs2))
    eegfusion_feature = layers.Activation('elu')(eegfusion_feature)
    fnirs_feature = layers.Activation('elu')(fnirs_feature)

    eegfusion_feature = layers.Dense(256, activation='elu')(eegfusion_feature)
    fnirs_feature = layers.Dense(256, activation='elu')(fnirs_feature)

    eeg_feature = layers.Flatten()(eeg2)
    eeg_feature = layers.Dense(256, activation='elu')(eeg_feature)

    # --- Information Bottleneck (IB) Layers ---
    # 这里的 latent_dim 决定了公共空间的维度
    eegfusion_latent = InformationBottleneck(latent_dim=latent_dim, beta=beta, name='eegfusion_ib')(eegfusion_feature)
    fnirs_latent = InformationBottleneck(latent_dim=latent_dim, beta=beta, name='fnirs_ib')(fnirs_feature)
    eeg_latent = InformationBottleneck(latent_dim=latent_dim, beta=beta, name='eeg_ib')(eeg_feature)

    # --- Cross-Modal Alignment (NEW!) ---
    # 强制 EEG 和 fNIRS 的潜在表示在空间中对齐
    # 这有助于模型学习到两个模态共有的、与任务相关的本质特征
    AlignmentLoss(weight=align_weight, name="eeg_fnirs_align")([eeg_latent, fnirs_latent])

    # --- Classification Heads ---
    eegfusion_pred = layers.Dense(2)(eegfusion_latent)
    fnirs_pred = layers.Dense(2)(fnirs_latent)
    eeg_pred = layers.Dense(2)(eeg_latent)

    eeg_pred = layers.Activation('softmax', name='eeg_output')(eeg_pred)
    eegfusion_pred = layers.Activation('softmax')(eegfusion_pred)
    fnirs_pred = layers.Activation('softmax')(fnirs_pred)

    eegfusion_pred = expand_dims_layer(axis=1)(eegfusion_pred)
    fnirs_pred = expand_dims_layer(axis=1)(fnirs_pred)

    the_pred = layers.Concatenate(axis=1)([eegfusion_pred, fnirs_pred])

    fnirs_p_weight = layers.Dense(1)(fnirs_latent)
    eegfusion_p_weight = layers.Dense(1)(eegfusion_latent)

    p_weight = layers.Concatenate()([eegfusion_p_weight, fnirs_p_weight])
    p_weight = layers.Activation('softmax')(p_weight)
    p_weight = expand_dims_layer(axis=-1)(p_weight)

    the_pred = layers.Multiply()([the_pred, p_weight])
    the_pred = reduce_sum_layer(axis=1, keepaxis=False, name='class_output')(the_pred)

    model = keras.Model(inputs=[eeg_input, fnirs_input], outputs=[the_pred, eeg_pred], name="efib_net")

    return model