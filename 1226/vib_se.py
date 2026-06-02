import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers

# ==============================================================================
# VIB-SE-Net: Variational Information Bottleneck with SE Blocks
# ==============================================================================

# --- Base Layers (Copied from previous versions) ---

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

class conv_block_se(keras.layers.Layer):
    def __init__(self, eeg_filter, eeg_size, eeg_stride,
                fnirs_filter, fnirs_size, fnirs_stride,
                eegfusion_filter, eegfusion_size, eegfusion_stride, 
                padding):
        super(conv_block_se, self).__init__()
        self.eeg_conv = layers.Conv3D(filters=eeg_filter, kernel_size=eeg_size, strides=eeg_stride, padding=padding)
        self.eeg_bn = layers.BatchNormalization()
        self.eeg_act = layers.Activation('elu')
        self.eeg_se = SEBlock(channels=eeg_filter, name="eeg_se")

        self.fnirs_conv = layers.Conv3D(filters=fnirs_filter, kernel_size=fnirs_size, strides=fnirs_stride, padding=padding)
        self.fnirs_bn = layers.BatchNormalization()
        self.fnirs_act = layers.Activation('elu')
        self.fnirs_se = SEBlock(channels=fnirs_filter, name="fnirs_se")

        self.eegfusion_conv = layers.Conv3D(filters=eegfusion_filter, kernel_size=eegfusion_size, strides=eegfusion_stride, padding=padding)
        self.eegfusion_bn = layers.BatchNormalization()
        self.eegfusion_act = layers.Activation('elu')
        self.eegfusion_se = SEBlock(channels=eegfusion_filter, name="fusion_se")

    def call(self, inputs):
        eegfusion, eeg, fnirs = inputs
        eeg_feature = self.eeg_se(self.eeg_act(self.eeg_bn(self.eeg_conv(eeg))))
        
        fnirs_shape = tf.shape(fnirs)
        if fnirs.shape.ndims == 6:
            fnirs = tf.reshape(fnirs, [fnirs_shape[0], fnirs_shape[2], fnirs_shape[3], fnirs_shape[4], fnirs_shape[1] * fnirs_shape[5]])

        fnirs_feature = self.fnirs_se(self.fnirs_act(self.fnirs_bn(self.fnirs_conv(fnirs))))
        eegfusion_feature = self.eegfusion_se(self.eegfusion_act(self.eegfusion_bn(self.eegfusion_conv(eegfusion))))
        
        # Simplified: Removed FGA for clarity and to isolate VIB effect
        return eegfusion_feature, eeg_feature, fnirs_feature

class gap(keras.layers.Layer):
    def __init__(self): super(gap, self).__init__()
    def call(self, inputs): return tf.reduce_mean(inputs, axis=-2, keepdims=True)

class expand_dims_layer(keras.layers.Layer):
    def __init__(self, axis, **kwargs): super(expand_dims_layer, self).__init__(**kwargs); self.axis = axis
    def call(self, inputs): return tf.expand_dims(inputs, axis=self.axis)

class reduce_sum_layer(keras.layers.Layer):
    def __init__(self, axis, keepaxis, **kwargs): super(reduce_sum_layer, self).__init__(**kwargs); self.axis, self.keepaxis = axis, keepaxis
    def call(self, inputs): return tf.math.reduce_sum(inputs, axis=self.axis, keepdims=self.keepaxis)


# --- Core Innovation: Variational Information Bottleneck ---

class VariationalInformationBottleneck(layers.Layer):
    """
    VIB Layer: Combines IB's compression with a VAE's reconstruction.
    Forces the latent space `z` to be both informative (for reconstruction)
    and minimal (for classification).
    """
    def __init__(self, input_dim, latent_dim, beta=1e-4, gamma=1.0, name=None):
        super(VariationalInformationBottleneck, self).__init__(name=name)
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.beta = beta   # Weight for KL divergence (IB Loss)
        self.gamma = gamma # Weight for Reconstruction Loss

        # Encoder
        self.encoder_mu = layers.Dense(latent_dim, name=f"{self.name}_mu")
        self.encoder_logvar = layers.Dense(latent_dim, name=f"{self.name}_logvar")

        # Decoder
        self.decoder = layers.Dense(input_dim, name=f"{self.name}_decoder")

        # Metrics
        self.kl_tracker = keras.metrics.Mean(name=f"{self.name}_kl")
        self.recon_tracker = keras.metrics.Mean(name=f"{self.name}_recon")

    def call(self, inputs, training=None):
        # 1. Encode
        mu = self.encoder_mu(inputs)
        logvar = self.encoder_logvar(inputs)
        logvar = tf.clip_by_value(logvar, -10.0, 10.0)

        # 2. Sample latent vector `z`
        if training or training is None:
            std = tf.exp(0.5 * logvar)
            eps = tf.random.normal(tf.shape(mu), dtype=inputs.dtype)
            z = mu + std * eps
        else:
            z = mu

        # 3. Calculate and add KL Loss (IB Loss)
        kl_div = -0.5 * tf.reduce_mean(1 + logvar - tf.square(mu) - tf.exp(logvar))
        self.add_loss(self.beta * kl_div)
        self.kl_tracker.update_state(kl_div)

        # 4. Decode and add Reconstruction Loss
        reconstructed_inputs = self.decoder(z)
        recon_loss = tf.reduce_mean(tf.square(inputs - reconstructed_inputs))
        self.add_loss(self.gamma * recon_loss)
        self.recon_tracker.update_state(recon_loss)

        # 5. Return the compressed latent vector for classification
        return z

# --- Main Model: VIB-SE-Net ---

def vib_se_net(latent_dim=128, beta=1e-4, gamma=1.0):
    eeg_input = keras.Input(shape=(16, 16, 600, 1), name="eeg_input")
    fnirs_input = keras.Input(shape=(11, 16, 16, 30, 2), name="fnirs_input")

    # 1. SE-Encoder Blocks
    eegfusion1, eeg1, fnirs1 = conv_block_se(eeg_filter=16, eeg_size=(2, 2, 13), eeg_stride=(2, 2, 6),
                                          fnirs_filter=16, fnirs_size=(2, 2, 5), fnirs_stride=(2, 2, 2),
                                          eegfusion_filter=16, eegfusion_size=(2, 2, 13), eegfusion_stride=(2, 2, 6),
                                          padding='same')((eeg_input, eeg_input, fnirs_input))
    eegfusion1, eeg1, fnirs1 = layers.Dropout(0.5)(eegfusion1), layers.Dropout(0.5)(eeg1), layers.Dropout(0.5)(fnirs1)

    eegfusion2, eeg2, fnirs2 = conv_block_se(eeg_filter=32, eeg_size=(2, 2, 5), eeg_stride=(2, 2, 2),
                                          fnirs_filter=32, fnirs_size=(2, 2, 3), fnirs_stride=(2, 2, 2),
                                          eegfusion_filter=32, eegfusion_size=(2, 2, 5), eegfusion_stride=(2, 2, 2),
                                          padding='same')((eegfusion1, eeg1, fnirs1))
    eegfusion2, eeg2, fnirs2 = gap()(eegfusion2), gap()(eeg2), gap()(fnirs2)
    eegfusion2, eeg2, fnirs2 = layers.Dropout(0.5)(eegfusion2), layers.Dropout(0.5)(eeg2), layers.Dropout(0.5)(fnirs2)

    # Flatten features
    eegfusion_feature = layers.Flatten()(eegfusion2)
    fnirs_feature = layers.Flatten()(fnirs2)
    eeg_feature = layers.Flatten()(eeg2)

    # Project to a common dimension before VIB
    feature_dim = 256
    eegfusion_feature = layers.Dense(feature_dim, activation='elu')(eegfusion_feature)
    fnirs_feature = layers.Dense(feature_dim, activation='elu')(fnirs_feature)
    eeg_feature = layers.Dense(feature_dim, activation='elu')(eeg_feature)

    # 2. VIB Layers for Disentanglement and Compression
    eegfusion_latent = VariationalInformationBottleneck(input_dim=feature_dim, latent_dim=latent_dim, beta=beta, gamma=gamma, name='vib_fusion')(eegfusion_feature)
    fnirs_latent = VariationalInformationBottleneck(input_dim=feature_dim, latent_dim=latent_dim, beta=beta, gamma=gamma, name='vib_fnirs')(fnirs_feature)
    eeg_latent = VariationalInformationBottleneck(input_dim=feature_dim, latent_dim=latent_dim, beta=beta, gamma=gamma, name='vib_eeg')(eeg_feature)

    # 3. Classification Heads (using the compressed latent vectors)
    eegfusion_pred = layers.Dense(2, activation='softmax')(layers.Dense(2)(eegfusion_latent))
    fnirs_pred = layers.Dense(2, activation='softmax')(layers.Dense(2)(fnirs_latent))
    eeg_pred = layers.Dense(2, activation='softmax', name='eeg_output')(layers.Dense(2)(eeg_latent))
    
    eegfusion_pred = expand_dims_layer(axis=1)(eegfusion_pred)
    fnirs_pred = expand_dims_layer(axis=1)(fnirs_pred)
    the_pred = layers.Concatenate(axis=1)([eegfusion_pred, fnirs_pred])
    
    # Weighting based on latent features
    fnirs_p_weight = layers.Dense(1)(fnirs_latent)
    eegfusion_p_weight = layers.Dense(1)(eegfusion_latent)
    p_weight = layers.Concatenate()([eegfusion_p_weight, fnirs_p_weight])
    p_weight = layers.Activation('softmax')(p_weight)
    p_weight = expand_dims_layer(axis=-1)(p_weight)
    
    the_pred = layers.Multiply()([the_pred, p_weight])
    the_pred = reduce_sum_layer(axis=1, keepaxis=False, name='class_output')(the_pred)
    
    model = keras.Model(inputs=[eeg_input, fnirs_input], outputs=[the_pred, eeg_pred], name="vib_se_net")
    return model