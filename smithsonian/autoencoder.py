# # from tensorflow.keras.preprocessing.image import ImageDataGenerator

# # train_datagen = ImageDataGenerator(rescale = 1./255)
# # train_batches = train_datagen.flow_from_directory(
# #     'images/train', target_size = (256, 256), shuffle = True, class_mode = 'input',
# #     batch_size = 256
# # )

# print(tf.__version__)

# image_dataset = tf.keras.preprocessing.image_dataset_from_directory(
#     'images/train/', image_size = (256, 256), batch_size = 64, seed = 42,
#     label_mode = None
# )
# https://github.com/keras-team/keras/issues/3923

import pickle
import numpy as np

np.random.seed(42)

import tensorflow as tf 
from tensorflow.keras import layers 
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import TensorBoard

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.4
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

BATCH_SIZE = 8
EPOCHS = 1

class ConvAutoencoder(Model):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()
        self.encoder = keras.Sequential([
            # layers.Conv2D(128, (3, 3), padding = 'same', activation = 'relu', input_shape = (256, 256, 3)),
            # layers.MaxPooling2D((2, 2), padding = 'same'),
            # layers.Conv2D(64, (3, 3), padding = 'same', activation = 'relu', input_shape = (256, 256, 3)),
            # layers.MaxPooling2D((2, 2), padding = 'same'),
            # layers.Conv2D(32, (3, 3), padding = 'same', activation = 'relu', input_shape = (256, 256, 3)),
            # layers.MaxPooling2D((2, 2), padding = 'same'),
            layers.Conv2D(16, (3, 3), padding = 'same', activation = 'relu', input_shape = (256, 256, 3)),
            layers.MaxPooling2D((2, 2), padding = 'same'),
            layers.Conv2D(8, (3, 3), padding = 'same', activation = 'relu'),
            layers.MaxPooling2D((2, 2), padding = 'same'),
            layers.Conv2D(8, (3, 3), padding = 'same', activation = 'relu'),
            layers.MaxPooling2D((2, 2), padding = 'same')
        ])
        self.decoder = keras.Sequential([
            layers.Conv2D(8, (3, 3), activation='relu', padding='same'),
            layers.UpSampling2D((2, 2)),
            layers.Conv2D(8, (3, 3), activation='relu', padding='same'),
            layers.UpSampling2D((2, 2)),
            layers.Conv2D(16, (3, 3), activation='relu', padding='same'),
            layers.UpSampling2D((2, 2)),
            # layers.Conv2D(32, (3, 3), activation='relu', padding = 'same'),
            # layers.UpSampling2D((2, 2)),
            # layers.Conv2D(64, (3, 3), activation = 'relu', padding = 'same'),
            # layers.UpSampling2D((2, 2)),
            # layers.Conv2D(128, (3, 3), activation = 'relu', padding = 'same'),
            # layers.UpSampling2D((2, 2)),
            layers.Conv2D(3, (3, 3), activation = 'sigmoid', padding = 'same')
        ])
    
    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded



# ------
# DATA
# ------
img_gen = keras.preprocessing.image.ImageDataGenerator(rescale = 1./255, validation_split = 0.05)
generator = img_gen.flow_from_directory(
    'images/train', target_size = (256, 256), batch_size = BATCH_SIZE, class_mode = 'input', shuffle = False, subset='validation')

# save the order of the data - to link back to their encoded vectors
data_order = generator.filenames
pickle.dump(data_order, open('data_order.pickle', 'wb'))

print(data_order[0:10])
print(f"Number of samples: {generator.samples}")

# -------
# MODEL
# -------
autoencoder = ConvAutoencoder()
autoencoder.compile(optimizer = 'adam', loss = 'binary_crossentropy')


history = autoencoder.fit(
    generator, epochs = EPOCHS, 
    steps_per_epoch = generator.samples // BATCH_SIZE,
    shuffle = False,
    # validation_data = validation_generator,
    # validation_steps = validation_generator.samples // batch_size,
    callbacks = [TensorBoard(log_dir = '\\tmp\\autoencoder')]
)

# save the history
# pickle.dump(history.history, open('training_history.pickle', 'wb'))

# save the model
# autoencoder.save('20_epoch_model_ae', save_format = 'tf')

# get the compressed vectors for the images
encoded_images = autoencoder.encoder.predict(generator)

print(f"Encoded image shapes: {encoded_images.shape}")
n_samples, nx, ny, nz = encoded_images.shape

# encoded_images = encoded_images.reshape((n_samples, nx * ny))
encoded_images = np.reshape(encoded_images, (len(encoded_images), nx * ny * nz))

print(f"Encoded image shapes: {encoded_images.shape}")

encoded_images = np.array_split(encoded_images, 40)

print(f"Encoded image shapes: {encoded_images[0].shape}")

# exit()

# import matplotlib.pyplot as plt 
# decoded_imgs = autoencoder.predict(generator)

# print(decoded_imgs.shape)
# print(decoded_imgs[0].shape)

# n = 2
# plt.figure(figsize = (20, 4))
# for i in range(1, n + 1):
#     ax = plt.subplot(2, n, i + n)
#     plt.imshow(decoded_imgs[i])
#     print(data_order[i])
# plt.show()

position = 0
for i, arr in enumerate(encoded_images, start = 1):
    img_data = {}
    for img in arr: 
        file_name = data_order[position].split('\\')[-1].split('.jpg')[0]
        img_data[file_name] = img
        position += 1

    pickle.dump(
        img_data, 
        open('encoded_imgs/encoded_images_' + str(i) + '.pickle', 'wb')
    )
# save the encoded images
# pickle.dump(encoded_images, open('encoded_images.pickle', 'wb'))
