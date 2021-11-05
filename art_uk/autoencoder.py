import tensorflow as tf 
from tensorflow import keras 

# tf.executing_eagerly()

LATENT_ENCODING_SIZE = 128 

class ResidualBlock(keras.models.Model):
    def __init__(self, input_depth, window):
        super(ResidualBlock, self).__init__()

        self.layer = keras.Sequential([
            keras.layers.Conv2D(
                filters = input_depth, kernel_size = window, padding = 'same'
            ),
            keras.layers.BatchNormalization(),
            keras.layers.ReLU(),
            keras.layers.Conv2D(
                filters = input_depth, kernel_size = window, padding = 'same'
            ),
            keras.layers.BatchNormalization()
        ])

    def call(self, x):
        # x = keras.layers.ReLU(tf.add(self.layer(x), x))
        relu_layer = keras.layers.ReLU()
        x = relu_layer(tf.add(self.layer(x), x))
        return x 

class EncoderBlock(keras.models.Model):
    def __init__(self, output_depth, input_depth, window, bottleneck = False):
        super(EncoderBlock, self).__init__()

        # self.residual_one = ResidualBlock(input_depth = input_depth, window = 3)
        # self.residual_two = ResidualBlock(input_depth = input_depth, window = 3)

        self.layer = keras.Sequential([
            keras.layers.Conv2D(
                filters = output_depth, kernel_size = window,
                padding = 'valid' if bottleneck else 'same',
                strides = 1 if bottleneck else 2
            ), # Conv(output_depth, (window_height, window_width))
            keras.layers.BatchNormalization(),
            keras.layers.ReLU()
        ])
    
    def call(self, x):
        # x = self.residual_one(x)
        # x = self.residual_two(x)
        x = self.layer(x)
        return x 

class DecoderBlock(keras.models.Model):
    def __init__(self, output_depth, input_depth, window, bottleneck = False):
        super(DecoderBlock, self).__init__()

        self.layer = keras.Sequential([
            keras.layers.Conv2DTranspose(
                filters = output_depth, kernel_size = window,
                padding = 'valid' if bottleneck else 'same',
                strides = 1 if bottleneck else 2 
            ),
            keras.layers.BatchNormalization(),
            keras.layers.ReLU()
        ])

        # self.residual_one = ResidualBlock(input_depth = input_depth, window = 3)
        # self.residual_two = ResidualBlock(input_depth = input_depth, window = 3)

    def call(self, x):
        x = self.layer(x)
        # x = self.residual_one(x)
        # x = self.residual_two(x)
        return x

class Autoencoder(keras.models.Model):
    def __init__(self):
        super(Autoencoder, self).__init__()

        self.encoder = keras.Sequential([
            keras.layers.Input((256, 256, 3)),
            keras.layers.Conv2D(filters = 256, kernel_size = 3, padding = 'same'),
            keras.layers.BatchNormalization(),
            keras.layers.ReLU(),
            EncoderBlock(output_depth = 64, input_depth = 256, window = 3),
            EncoderBlock(output_depth = 32, input_depth = 64, window = 3),
            EncoderBlock(output_depth = 16, input_depth = 32, window = 3),
            EncoderBlock(output_depth = 8, input_depth = 16, window = 3),
            EncoderBlock(output_depth = 4, input_depth = 8, window = 3),
            EncoderBlock(output_depth = LATENT_ENCODING_SIZE, input_depth = 4, window = 3, bottleneck = True),
            keras.layers.Flatten(),
            keras.layers.Dense(LATENT_ENCODING_SIZE),
            keras.layers.BatchNormalization(),
            keras.layers.ReLU(),
            keras.layers.Dense(LATENT_ENCODING_SIZE)
        ], name = 'encoder')

        self.decoder = keras.Sequential([
            keras.layers.Input((LATENT_ENCODING_SIZE)),
            keras.layers.Dense(LATENT_ENCODING_SIZE),
            keras.layers.BatchNormalization(),
            keras.layers.ReLU(),
            keras.layers.Reshape((LATENT_ENCODING_SIZE, 1, 1)),
            DecoderBlock(output_depth = 4, input_depth = LATENT_ENCODING_SIZE, window = 3, bottleneck = True),
            DecoderBlock(output_depth = 8, input_depth = 4, window = 3),
            DecoderBlock(output_depth = 16, input_depth = 8, window = 3),
            DecoderBlock(output_depth = 32, input_depth = 16, window = 3),
            DecoderBlock(output_depth = 64, input_depth = 32, window = 3),
            DecoderBlock(output_depth = 256, input_depth = 64, window = 3),
            keras.layers.Conv2D(filters = 256, kernel_size = 3, padding = 'same')
        ], name = 'decoder')

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

