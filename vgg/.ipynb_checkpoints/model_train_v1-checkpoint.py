import numpy as np
import tensorflow as tf
from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split
from model import Model

'''
Data Description
----------------
Num Tuples: 3023
Image Size: 62 x 47 x 3
Unique Labels: 62
print(dataset.images.shape)
print(len(set(dataset.target_names)))
'''

epochs = 10000


data_shape = [62, 47, 3]
batch_size = int(3023 * 0.6 * 0.1)


model = Model(sess=tf.Session(),
              data_shape=data_shape,
              batch_size=batch_size,
              epochs=epochs,
              learning_rate,
              conv_parameters,
              max_pool_parameters,
              dropout_parameters,
              use_batch_norm,
              use_dropout,
              tensorboard_directory)

dataset = fetch_lfw_people(data_home=None,
                           # resize=0.6,
                           color=True,
                           download_if_missing=True,
                           min_faces_per_person=20)

images = dataset.images
labels = dataset.target

X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.30)
model.input_data(data=X_train,
                 labels=y_train,
                 val_data=X_test,
                 val_labels=y_test)
