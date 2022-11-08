# import the necessary packages
import numpy as np
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Lambda
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import UpSampling2D
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dropout
from keras import backend as K

class CVAE:
    @staticmethod
    def build(width, height, depth, filters=(16, 32, 64, 128, 256), latent_dim = 2):
        # initialize the input shape to be "channels last" along with
        # the channels dimension itself
        inputShape = (height, width, depth)
        chanDim = -1

        # define the input to the encoder
        inputs = Input(shape = inputShape, name='encoder_input')
        cx = inputs

        # loop over the number of filters
        for f in filters:
            # apply a CONV => RELU => BN operation
            cx = Conv2D(f, (3, 3), strides = 2, padding="same")(cx)
            cx = LeakyReLU(alpha=0.2)(cx)
            cx = BatchNormalization(axis=chanDim)(cx)
            #x = MaxPooling2D(pool_size=(2,2))(x)

    # flatten the network and then construct our latent vector
        x = Flatten()(cx)
        x = Dense(50, activation='relu')(x)
        x = Dropout(0.1)(x)
        x = BatchNormalization()(x)
        mu = Dense(latent_dim, name='latent_mu')(x)
        sigma = Dense(latent_dim, name='latent_sigma')(x)

    # Get Conv2D shape for Conv2DTranspose operation in decoder
        volumeSize = K.int_shape(cx)

    # Define sampling with reparameterization trick
        def sample_z(args):
            mu, sigma = args
            batch = K.shape(mu)[0]
            dim   = K.int_shape(mu)[1]
            eps   = K.random_normal(shape=(batch, dim))
            return mu + K.exp(sigma / 2) * eps

    # Use reparameterization trick to ensure correct gradient
        z = Lambda(sample_z, output_shape=(latent_dim, ), name='z')([mu, sigma])

    # build the encoder model
        encoder = Model(inputs, [mu, sigma, z], name='encoder')

    # Decoder model ==> It will accept the output of the encoder as its inputs
        latentInputs = Input(shape=(latent_dim,), name='decoder_input')
        x = BatchNormalization()(latentInputs)
        x = Dense(np.prod(volumeSize[1:]))(x)
        cx = Reshape((volumeSize[1], volumeSize[2], volumeSize[3]))(x)

        # loop over our number of filters again, but this time in reverse order
        for f in filters[::-1]: # filters[::-1] = [256,128,64,32,16]
            # apply a CONV_TRANSPOSE => RELU => BN operation
            cx = Conv2DTranspose(f, (3, 3), strides = 2, padding="same")(cx)
            cx = LeakyReLU(alpha=0.2)(cx)
            cx = BatchNormalization(axis=chanDim)(cx)
            #x = UpSampling2D(size=(2,2))(x)

        # apply a single CONV_TRANSPOSE layer used to recover the original depth of the image
        cx = Conv2DTranspose(depth, (3, 3), padding="same")(cx)
        outputs = Activation("sigmoid")(cx)

        # build the decoder model
        decoder = Model(latentInputs, outputs, name="decoder")

        # our variational autoencoder is the encoder + decoder
        vae = Model(inputs, decoder(encoder(inputs)), name="vae")

        # return a 3-tuple of the encoder, decoder, and autoencoder
        return (encoder, decoder, vae, mu, sigma)