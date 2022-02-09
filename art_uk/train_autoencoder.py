import tensorflow as tf 
import numpy as np 
from tensorflow import keras 
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

tf.random.set_seed(42)
np.random.seed(42)

config = ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.4
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

from autoencoder import Autoencoder
from tensorflow.keras.callbacks import Callback
from datetime import datetime as dt 
from pathlib import Path 
import json, os  
import pickle as p 

class LoggerCallback(Callback):
    def __init__(self, checkpoint_path):
        super().__init__()
        self.checkpoint_path = checkpoint_path

    def on_epoch_end(self, epoch, logs = None):
        run_info = {
            'date': dt.now().strftime('%Y-%m-%d %H:%M:%S'),
            'directory': str(Path('.').resolve()),
            'loss': logs['loss'],
            'rounded_accuracy': logs['rounded_accuracy'],
            'val_loss': logs['val_loss'],
            'val_rounded_accuracy': logs['val_rounded_accuracy'],
            'epoch': epoch 
        }

        save_location = Path(self.checkpoint_path + '/training_log.json')
        if save_location.is_file():
            stored = json.load(open(save_location, 'r'))
            stored.append(run_info)
            json.dump(stored, open(save_location, 'w'))
        else:
            json.dump([run_info], open(save_location, 'w'))

def rounded_accuracy(y_true, y_pred):
    return keras.metrics.binary_accuracy(tf.round(y_true), tf.round(y_pred))


BATCH_SIZE = 32
EPOCHS = 150

# ----- IMAGE GENERATORS -----
image_generator = keras.preprocessing.image.ImageDataGenerator(
    rescale = 1./255, data_format = 'channels_last'
)

train_generator = image_generator.flow_from_directory(
    'data/images/images/train', target_size = (32, 32), batch_size = BATCH_SIZE,
    class_mode = 'input'
)

validation_generator = image_generator.flow_from_directory(
    'data/images/images/validation', target_size = (32, 32), batch_size = BATCH_SIZE,
    class_mode = 'input'
)
# ---------

# save the order of the data in the generators so we can link them to the encoded vectors
data_order_train = train_generator.filenames
data_order_val = validation_generator.filenames

p.dump(data_order_train, open('data/autoencoder_output/data_order_train.pickle', 'wb'))
p.dump(data_order_val, open('data/autoencoder_output/data_order_validation.pickle', 'wb'))

print(f"Number of training samples: {train_generator.samples}; validation samples: {validation_generator.samples}")

# defining the model
loss = keras.losses.BinaryCrossentropy()
optimizer = keras.optimizers.Adam()

ae_obj = Autoencoder()
ae_obj.autoencoder.compile(optimizer = optimizer, loss = loss, metrics = [rounded_accuracy])
ae = ae_obj.autoencoder

# callbacks
early_stopping = keras.callbacks.EarlyStopping(monitor = 'val_loss', patience = 3)
logger = LoggerCallback('data/autoencoder_output')

# fit the model
history = ae.fit(
    train_generator,
    epochs = EPOCHS,
    validation_data = validation_generator,
    callbacks = [early_stopping, logger]
)

# encode the images
encoded_train = ae_obj.encoder.predict(train_generator)
encoded_val = ae_obj.encoder.predict(validation_generator)

# reshape the encoded image vector so it's flat
n_samples, nx, ny, nz = encoded_train.shape
encoded_train = np.reshape(encoded_train, (n_samples, nx * ny * nz))

# do the same for the validation set
n_samples, nx, ny, nz = encoded_val.shape
encoded_val = np.reshape(encoded_val, (n_samples, nx * ny * nz))

print(f"Train encoded shape: {encoded_train.shape}")
print(f"Val encoded shape: {encoded_val.shape}")

# bring together the two encoded sets
encoded_imgs_train = {
    f_name.split(os.sep)[-1].split('.jpg')[0]: encoded_train[idx]
    for idx, f_name in enumerate(data_order_train)
}

encoded_imgs_val = {
    f_name.split(os.sep)[-1].split('.jpg')[0]: encoded_val[idx]
    for idx, f_name in enumerate(data_order_val)
}

encoded_imgs = {**encoded_imgs_train, **encoded_imgs_val}

p.dump(
    encoded_imgs,
    open('data/images/encoded_imgs.pickle', 'wb')
)

# import matplotlib.pyplot as plt 

# data_list = []
# batch_index = 0
# while batch_index <= train_generator.batch_index:
#     data = train_generator.next()
#     data_list.append(data[0])
#     batch_index = batch_index + 1
# data_list[0].shape 

# predicted = ae.predict(data_list[0])
# plt.imshow(data_list[0][0])
# plt.show()
# plt.imshow(predicted[0])
# plt.show()