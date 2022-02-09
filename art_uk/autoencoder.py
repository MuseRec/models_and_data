import tensorflow as tf 
import numpy as np 
from tensorflow import keras 
from tensorflow.keras import layers

# tf.executing_eagerly()

tf.random.set_seed(42)
np.random.seed(42)

class Autoencoder:
    def __init__(self):
        super().__init__()

       
        self.encoder = keras.models.Sequential([
            layers.Conv2D(8, (3, 3), padding = 'same', activation = 'relu', input_shape = (32, 32, 3)),
            layers.MaxPooling2D(pool_size = (2, 2), padding = 'same'),
            layers.Conv2D(16, (3, 3), padding = 'same', activation = 'relu'),
            layers.MaxPooling2D(pool_size = (2, 2), padding = 'same'),
            layers.Conv2D(32, (3, 3), padding = 'same', activation = 'relu'),
            layers.MaxPooling2D(pool_size = (2, 2), padding = 'same'),
        ])
        self.decoder = keras.models.Sequential([
            layers.Conv2D(32, (3, 3), padding = 'same', activation = 'relu'),
            layers.UpSampling2D((2, 2)),
            layers.Conv2D(16, (3, 3), padding = 'same', activation = 'relu'),
            layers.UpSampling2D((2, 2)),
            layers.Conv2D(8, (3, 3), padding = 'same', activation = 'relu'),
            layers.UpSampling2D((2, 2)),
            layers.Conv2D(3, (3, 3), padding = 'same', activation = 'sigmoid')
        ])
        self.autoencoder = keras.models.Sequential([self.encoder, self.decoder])
