import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers

# ==============================================================================
# MS-SE-STA-Net: Multi-Scale Convolution with SE Blocks (No IB)
# ==============================================================================

class SEBlock(layers.Layer):
    def __init__(self, channels, reduction=16, name="se_block"):
        super(SEBlock, self).__init__(name=name)
        self.channels = channels
        self.avg_pool = layers.GlobalAveragePooling3D()
        self.fc1 = layers.Dense(channels // reduction, activation='relu', kernel_initializer='he_normal')
        self.fc2 = layers.Dense(channels, activation='sigmoid', kernel_initializer='he_normal')
        self.reshape = layers.Reshape((1, 1, 1, channels))

    def call(self, inputs):
        x = self.avg_pool(inputs)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.reshape(x)
        return inputs * x

class MultiScaleConv3D(layers.Layer):
    def __init__(self, filters, spatial_size, temporal_strides, padding='same', name="ms_conv"):
        super(MultiScaleConv3D, self).__init__(name=name)
        self.conv1 = layers.Conv3D(filters=filters // 3, kernel_size=(spatial_size[0], spatial_size[1], 5), 
                                   strides=temporal_strides, padding=padding)
        self.conv2 = layers.Conv3D(filters=filters // 3, kernel_size=(spatial_size[0], spatial_size[1], 13), 
                                   strides=temporal_strides, padding=padding)
        self.conv3 = layers.Conv3D(filters=filters - 2*(filters//3), kernel_size=(spatial_size[0], spatial_size[1], 25), 
                                   strides=temporal_strides, padding=padding)
        self.bn = layers.BatchNormalization()
        self.act = layers.Activation('elu')
        self.se = SEBlock(channels=filters)

    def call(self, inputs):
        x1 = self.conv1(inputs)
        x2 = self.conv2(inputs)
        x3 = self.conv3(inputs)
        x = layers.Concatenate(axis=-1)([x1, x2, x3])
        x = self.bn(x)
        x = self.act(x)
        x = self.se(x)
        return x

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

    def call(self, inputs):
        eeg, fnirs = inputs
        q_eeg = self.q_flat(eeg)
        fusion_output = self.fusion_proj(q_eeg)
        q_eeg = self.q_proj(q_eeg)
        q_eeg = tf.expand_dims(q_eeg, axis=1) 
        
        fnirs_shape = tf.shape(fnirs)
        batch_size = fnirs_shape[0]
        channels = fnirs_shape[-1]
        k_fnirs = tf.reshape(fnirs, [batch_size, -1, channels])
        
        k_fnirs = self.pos(k_fnirs)
        k_fnirs = self.k_proj(k_fnirs) 
        
        fnirs_weighted, _ = self.dot_product_attention(q_eeg, k_fnirs, return_attention_scores=True) 
        
        q_eeg = tf.math.reduce_mean(q_eeg, axis=1)
        fnirs_weighted = tf.math.reduce_mean(fnirs_weighted, axis=1)
       
        return fusion_output, fnirs_weighted

class gap(keras.layers.Layer):
    def __init__(self):
        super(gap, self).__init__()
    def call(self, inputs):
        return tf.reduce_mean(inputs, axis=-2, keepdims=True)

class conv_block_ms_se(keras.layers.Layer):
    def __init__(self, eeg_filter, eeg_size, eeg_stride,
                fnirs_filter, fnirs_size, fnirs_stride,
                eegfusion_filter, eegfusion_size, eegfusion_stride, 
                padding):
        super(conv_block_ms_se, self).__init__()
        self.eeg_ms_conv = MultiScaleConv3D(filters=eeg_filter, spatial_size=(eeg_size[0], eeg_size[1]), 
                                            temporal_strides=eeg_stride, padding=padding, name="eeg_ms")
        self.fnirs_ms_conv = MultiScaleConv3D(filters=fnirs_filter, spatial_size=(fnirs_size[0], fnirs_size[1]), 
                                              temporal_strides=fnirs_stride, padding=padding, name="fnirs_ms")
        self.eegfusion_ms_conv = MultiScaleConv3D(filters=eegfusion_filter, spatial_size=(eegfusion_size[0], eegfusion_size[1]), 
                                                  temporal_strides=eegfusion_stride, padding=padding, name="fusion_ms")

    def call(self, inputs):
        eegfusion, eeg, fnirs = inputs
        eeg_feature = self.eeg_ms_conv(eeg)
        fnirs_shape_tensor = tf.shape(fnirs)
        if fnirs.shape.ndims == 6:
            fnirs_reshaped = tf.reshape(fnirs, [fnirs_shape_tensor[0], fnirs_shape_tensor[2], fnirs_shape_tensor[3], fnirs_shape_tensor[4], fnirs_shape_tensor[1] * fnirs_shape_tensor[5]])
        else:
            fnirs_reshaped = fnirs
        fnirs_feature = self.fnirs_ms_conv(fnirs_reshaped)
        eegfusion_feature = self.eegfusion_ms_conv(eegfusion)
        return eegfusion_feature, eeg_feature, fnirs_feature

class reduce_sum_layer(keras.layers.Layer):
    def __init__(self, axis, keepaxis, **kwargs):
        super(reduce_sum_layer, self).__init__(**kwargs)
        self.axis = axis
        self.keepaxis = keepaxis
    def call(self, inputs):
        return tf.math.reduce_sum(inputs, axis=self.axis, keepdims=self.keepaxis)

class expand_dims_layer(keras.layers.Layer):
    def __init__(self, axis, *args, **kwargs):
        super(expand_dims_layer, self).__init__(**kwargs)
        self.axis = axis
    def call(self, inputs):
        return tf.expand_dims(inputs, axis=self.axis)

def sta_net_ms_se_only():
    eeg_input = keras.Input(shape=(16, 16, 600, 1), name="eeg_input")
    fnirs_input = keras.Input(shape=(11, 16, 16, 30, 2), name="fnirs_input")

    eegfusion1, eeg1, fnirs1 = conv_block_ms_se(eeg_filter=16, eeg_size=(2, 2, 13), eeg_stride=(2, 2, 6),
                                          fnirs_filter=16, fnirs_size=(2, 2, 5), fnirs_stride=(2, 2, 2),
                                          eegfusion_filter=16, eegfusion_size=(2, 2, 13), eegfusion_stride=(2, 2, 6),
                                          padding='same')((eeg_input, eeg_input, fnirs_input))
    eegfusion1, eeg1, fnirs1 = layers.Dropout(0.5)(eegfusion1), layers.Dropout(0.5)(eeg1), layers.Dropout(0.5)(fnirs1)                            

    eegfusion2, eeg2, fnirs2 = conv_block_ms_se(eeg_filter=32, eeg_size=(2, 2, 5), eeg_stride=(2, 2, 2),
                                          fnirs_filter=32, fnirs_size=(2, 2, 3), fnirs_stride=(2, 2, 2),
                                          eegfusion_filter=32, eegfusion_size=(2, 2, 5), eegfusion_stride=(2, 2, 2),
                                          padding='same')((eegfusion1, eeg1, fnirs1)) 
    eegfusion2, eeg2, fnirs2 = gap()(eegfusion2), gap()(eeg2), gap()(fnirs2)
    eegfusion2, eeg2, fnirs2 = layers.Dropout(0.5)(eegfusion2), layers.Dropout(0.5)(eeg2), layers.Dropout(0.5)(fnirs2)

    eegfusion_feature, fnirs_feature = e_f_attention(emb_size=256, d_model=256, heads=10, drop=0.5)((eegfusion2, fnirs2))
    
    # 关键修改: 给特征层命名
    eegfusion_feature = layers.Activation('elu', name="fusion_feature_output")(eegfusion_feature)
    fnirs_feature = layers.Activation('elu', name="fnirs_feature_output")(fnirs_feature)

    eegfusion_feature_pweight = layers.Dense(256, activation='elu')(eegfusion_feature)
    fnirs_feature_pweight = layers.Dense(256, activation='elu')(fnirs_feature)

    eeg_feature = layers.Flatten()(eeg2)
    eeg_feature = layers.Dense(256, activation='elu', name="eeg_feature_output")(eeg_feature)
    
    eegfusion_pred = layers.Dense(2)(eegfusion_feature_pweight)
    fnirs_pred = layers.Dense(2)(fnirs_feature_pweight)
    eeg_pred = layers.Dense(2)(eeg_feature)

    eeg_pred = layers.Activation('softmax', name='eeg_output')(eeg_pred) 
    eegfusion_pred = layers.Activation('softmax')(eegfusion_pred) 
    fnirs_pred = layers.Activation('softmax')(fnirs_pred) 
    
    eegfusion_pred = expand_dims_layer(axis=1)(eegfusion_pred) 
    fnirs_pred = expand_dims_layer(axis=1)(fnirs_pred) 

    the_pred = layers.Concatenate(axis=1)([eegfusion_pred, fnirs_pred]) 
    
    fnirs_p_weight = layers.Dense(1)(fnirs_feature_pweight) 
    eegfusion_p_weight = layers.Dense(1)(eegfusion_feature_pweight) 
    
    p_weight = layers.Concatenate()([eegfusion_p_weight, fnirs_p_weight]) 
    p_weight = layers.Activation('softmax')(p_weight) 
    p_weight = expand_dims_layer(axis=-1)(p_weight) 

    the_pred = layers.Multiply()([the_pred, p_weight]) 
    the_pred = reduce_sum_layer(axis=1, keepaxis=False, name='class_output')(the_pred) 
    
    model = keras.Model(inputs=[eeg_input, fnirs_input], outputs=[the_pred, eeg_pred], name="sta_net_ms_se_only")

    return model