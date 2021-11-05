import tensorflow as tf 
from tensorflow import keras

img_gen = keras.preprocessing.image.ImageDataGenerator(
    rescale = 1./255
)
train_generator = img_gen.flow_from_directory(
    'data/images/train', target_size = (256, 256), batch_size = 8, class_mode = 'input'
)
validation_generator = img_gen.flow_from_directory(
    'data/images/validation', target_size = (256, 256), batch_size = 8, class_mode = 'input'
)