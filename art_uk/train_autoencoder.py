import tensorflow as tf 
from tensorflow import keras 
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.4
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

from autoencoder import Autoencoder

BATCH_SIZE = 8

image_generator = keras.preprocessing.image.ImageDataGenerator(
    rescale = 1./255
)

train_generator = image_generator.flow_from_directory(
    'data/images/train', target_size = (256, 256), batch_size = BATCH_SIZE,
    class_mode = 'input'
)

validation_generator = image_generator.flow_from_directory(
    'data/images/validation', target_size = (256, 256), batch_size = BATCH_SIZE,
    class_mode = 'input'
)

loss = keras.losses.MeanSquaredError()
optimizer = keras.optimizers.Adam()

ae = Autoencoder()
ae.compile(optimizer = optimizer, loss = loss, run_eagerly = True)

# print(ae.summary())

history = ae.fit(
    train_generator,
    epochs = 1,
    validation_data = validation_generator,
)