import numpy as np

import tensorflow as tf

from tensorflow.keras.layers import Input, Conv2D, Flatten, Dense, Conv2DTranspose, Lambda, Reshape, Layer
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K


INPUT_DIM = (10,10,1)

CONV_FILTERS = [32,64,64, 64]
CONV_KERNEL_SIZES = [2,2,2,2]
CONV_STRIDES = [1,2,2,2]
CONV_ACTIVATIONS = ['relu','relu','relu','relu']

DENSE_SIZE = 64

# CONV_T_FILTERS = [64,64,32,3]
CONV_T_FILTERS = [2,1]
CONV_T_KERNEL_SIZES = [3,3,3,3]
CONV_T_STRIDES = [1,2,2,2]
# CONV_T_ACTIVATIONS = ['relu','relu','relu','sigmoid']
CONV_T_ACTIVATIONS = ['relu','sigmoid']

Z_DIM = 16

BATCH_SIZE = 50
LEARNING_RATE = 0.0001
KL_TOLERANCE = 0.5




class Sampling(Layer):
    def call(self, inputs):
        mu, log_var = inputs
        epsilon = K.random_normal(shape=K.shape(mu), mean=0., stddev=1.)
        return mu + K.exp(log_var / 2) * epsilon


class VAEModel(Model):
    def __init__(self, encoder, decoder, r_loss_factor, **kwargs):
        super(VAEModel, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.r_loss_factor = r_loss_factor

    def train_step(self, data):
        if isinstance(data, tuple):
            data = data[0]
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(
                tf.square(data - reconstruction), axis = [1,2,3]
            )
            reconstruction_loss *= self.r_loss_factor
            kl_loss = 1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
            kl_loss = tf.reduce_sum(kl_loss, axis = 1)
            kl_loss *= -0.5
            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        return {
            "loss": total_loss,
            "reconstruction_loss": reconstruction_loss,
            "kl_loss": kl_loss,
        }
    
    def call(self,inputs):
        # latent = self.encoder(inputs)
        z_mean, z_log_var, latent = self.encoder(inputs)
        return self.decoder(latent)
    
    
class VAE():
    def __init__(self):
        self.models = self._build()
        self.full_model = self.models[0]
        self.encoder = self.models[1]
        self.decoder = self.models[2]

        self.input_dim = INPUT_DIM
        self.z_dim = Z_DIM
        self.learning_rate = LEARNING_RATE
        self.kl_tolerance = KL_TOLERANCE

    def _build(self):
        INPUT_SHAPE = (10, 10, 1)
        
        LATENT_DIM = 8

       
        input_layer = Input(shape=INPUT_SHAPE, name='observation_input')
        flattened_input = Flatten()(input_layer)
        encoder_output = Dense(LATENT_DIM, activation='relu', name='encoder_output')(flattened_input)

 
        decoder_input = Input(shape=(LATENT_DIM,), name='decoder_input')
        decoder_output = Dense(np.prod(INPUT_SHAPE), activation='sigmoid', name='decoder_output')(decoder_input)
        output_layer = Reshape(INPUT_SHAPE)(decoder_output)

        encoder = Model(inputs=input_layer, outputs=encoder_output, name='encoder')
        decoder = Model(inputs=decoder_input, outputs=output_layer, name='decoder')

        autoencoder = Model(inputs=input_layer, outputs=decoder(encoder(input_layer)), name='autoencoder')

        autoencoder.compile(optimizer='adam', loss='mse')  

        # input_img = Input(shape=INPUT_DIM, name='observation_input')
        # vae_z_in = Flatten()(input_img)
        # encoded = Dense(Z_DIM, activation='relu')(vae_z_in)
        
        # encoded_input = Input(shape=(Z_DIM,))
        # decoded = Dense(100, activation='sigmoid')(encoded_input)
        # output_layer = Reshape((10,10,1), name='unflatten')(decoded)
        
        
        
        
        
        
        # encoder = Model(inputs=input_img, outputs=encoded)
        # decoder = Model(inputs=encoded_input, outputs=output_layer)
        
        # autoencoder = Model(inputs=input_img, outputs=decoder(encoder(input_img)))
        # # autoencoder = Model(inputs=encoder, outputs=decoder)#batu
        # # autoencoder = Model(inputs=input_img, outputs=output_layer)#batu
        # # autoencoder = Model(inputs=input_img, outputs=decoder(encoded_input))#batu
        # autoencoder.compile(optimizer='adam', loss='mse')


        # autoencoder.summary()

        # opti = Adam(lr=LEARNING_RATE)
        # vae_full.compile(optimizer=opti)
        
        return (autoencoder,encoder,decoder)

    def set_weights(self, filepath):
        self.full_model.load_weights(filepath)
        
    def train(self, checkpoint_path,data):
        cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, save_weights_only=True,verbose=1)
        self.full_model.fit(data, data,
                shuffle=True,
                epochs=1,
                batch_size=BATCH_SIZE,
                callbacks=[cp_callback]
                )
        
    def save_weights(self, filepath):
        self.full_model.save_weights(filepath)
        
    