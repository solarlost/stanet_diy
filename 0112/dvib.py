import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers

# ==============================================================================
# D-Net: Disentangled Network (No VIB)
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
    def __init__(self, filters, spatial_size, temporal_strides, padding='same', name=None):
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

class OrthogonalityLoss(layers.Layer):
    def __init__(self, weight=0.1, name="ortho_loss"):
        super(OrthogonalityLoss, self).__init__(name=name)
        self.weight = weight
        self.loss_tracker = keras.metrics.Mean(name=name)

    def call(self, inputs):
        z1, z2 = inputs
        dim = tf.cast(tf.shape(z1)[-1], dtype=z1.dtype)
        dot_product = tf.reduce_sum(z1 * z2, axis=1)
        loss = tf.reduce_mean(tf.square(dot_product)) / dim
        self.add_loss(self.weight * loss)
        self.loss_tracker.update_state(loss)
        return loss

class SimilarityLoss(layers.Layer):
    def __init__(self, weight=0.1, name="sim_loss"):
        super(SimilarityLoss, self).__init__(name=name)
        self.weight = weight
        self.loss_tracker = keras.metrics.Mean(name=name)

    def call(self, inputs):
        z1, z2 = inputs
        loss = tf.reduce_mean(tf.square(z1 - z2))
        self.add_loss(self.weight * loss)
        self.loss_tracker.update_state(loss)
        return loss

class SharedEncoder(layers.Layer):
    def __init__(self, hidden_dim=256, name="shared_enc"):
        super(SharedEncoder, self).__init__(name=name)
        self.dense1 = layers.Dense(hidden_dim, activation='elu')
        self.bn = layers.BatchNormalization()
        self.dense2 = layers.Dense(hidden_dim, activation='elu')
        
    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.bn(x)
        x = self.dense2(x)
        return x

def dvib_net(latent_dim=64, ortho_weight=0.1, sim_weight=1.0, **kwargs): # beta is not used
    """
    D-Net Architecture (No VIB)
    """
    eeg_input = keras.Input(shape=(16, 16, 600, 1), name="eeg_input")
    fnirs_input = keras.Input(shape=(11, 16, 16, 30, 2), name="fnirs_input")

    # --- 1. Feature Extraction (MS-SE Backbone) ---
    eeg_feat = MultiScaleConv3D(16, (2, 2), (2, 2, 6), padding='same', name="eeg_ms1")(eeg_input)
    eeg_feat = layers.Dropout(0.5)(eeg_feat)
    eeg_feat = MultiScaleConv3D(32, (2, 2), (2, 2, 2), padding='same', name="eeg_ms2")(eeg_feat)
    eeg_feat = layers.GlobalAveragePooling3D()(eeg_feat)
    eeg_feat = layers.Dense(256, activation='elu')(eeg_feat) 

    fnirs_reshaped = tf.reshape(fnirs_input, [-1, 11, 16, 16, 60])
    fnirs_feat = MultiScaleConv3D(16, (2, 2), (2, 2, 6), padding='same', name="fnirs_ms1")(fnirs_reshaped)
    fnirs_feat = layers.Dropout(0.5)(fnirs_feat)
    fnirs_feat = MultiScaleConv3D(32, (2, 2), (2, 2, 2), padding='same', name="fnirs_ms2")(fnirs_feat)
    fnirs_feat = layers.GlobalAveragePooling3D()(fnirs_feat)
    fnirs_feat = layers.Dense(256, activation='elu')(fnirs_feat) 

    # --- 2. Shared & Private Encoders (Simple Dense Layers) ---
    
    # A. Private Encoders
    z_eeg_spec = layers.Dense(latent_dim, name="eeg_spec")(eeg_feat)
    z_fnirs_spec = layers.Dense(latent_dim, name="fnirs_spec")(fnirs_feat)

    # B. Shared Encoder
    shared_encoder = SharedEncoder(hidden_dim=256, name="shared_encoder")
    common_feat_eeg = shared_encoder(eeg_feat)
    common_feat_fnirs = shared_encoder(fnirs_feat)
    
    z_common_eeg = layers.Dense(latent_dim, name="common_eeg")(common_feat_eeg)
    z_common_fnirs = layers.Dense(latent_dim, name="common_fnirs")(common_feat_fnirs)

    # --- 3. Constraints ---
    SimilarityLoss(weight=sim_weight, name="sim_loss")([z_common_eeg, z_common_fnirs])
    OrthogonalityLoss(weight=ortho_weight, name="ortho_eeg")([z_common_eeg, z_eeg_spec])
    OrthogonalityLoss(weight=ortho_weight, name="ortho_fnirs")([z_common_fnirs, z_fnirs_spec])

    # --- 4. Classification ---
    z_final = layers.Concatenate()([z_eeg_spec, z_fnirs_spec, z_common_eeg, z_common_fnirs])
    
    x = layers.Dense(128, activation='elu')(z_final)
    x = layers.Dropout(0.5)(x)
    class_output = layers.Dense(2, activation='softmax', name="class_output")(x)
    
    eeg_output = layers.Dense(2, activation='softmax', name="eeg_output")(z_eeg_spec)

    model = keras.Model(inputs=[eeg_input, fnirs_input], outputs=[class_output, eeg_output], name="d_net")
    return model