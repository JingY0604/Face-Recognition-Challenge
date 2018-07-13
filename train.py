#!/usr/bin/env python3

import numpy as np
import tensorflow as tf
from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split

import random

from fire import Fire

from utils.tensorboard import Tensorboard
from model import Model
from utils.augment import Augment


'''
Data Description
----------------
Num Tuples: 3023
Image Size: (3023, 125, 94, 3)
Unique Labels: 62
print(dataset.images.shape)
print(len(set(dataset.target_names)))
'''
augments = ['brightness', 'hue', 'contrast', 'flip_h']


def train(restore=True):
    tensorboard_directory = r'./tmp/tensorboard/015'
    tensorboard_paths = [r'C:\Users\parth\Documents\GitHub\Facial-Recognition\tmp\tensorboard\005',
                         r'C:\Users\parth\Documents\GitHub\Facial-Recognition\tmp\tensorboard\007',
                         r'C:\Users\parth\Documents\GitHub\Facial-Recognition\tmp\tensorboard\008',
                         r'C:\Users\parth\Documents\GitHub\Facial-Recognition\tmp\tensorboard\009',
                         r'C:\Users\parth\Documents\GitHub\Facial-Recognition\tmp\tensorboard\010',
                         r'C:\Users\parth\Documents\GitHub\Facial-Recognition\tmp\tensorboard\011',
                         r'C:\Users\parth\Documents\GitHub\Facial-Recognition\tmp\tensorboard\012',
                         r'C:\Users\parth\Documents\GitHub\Facial-Recognition\tmp\tensorboard\013',
                         r'C:\Users\parth\Documents\GitHub\Facial-Recognition\tmp\tensorboard\014',
                         r'C:\Users\parth\Documents\GitHub\Facial-Recognition\tmp\tensorboard\015']
    tensorboard_names = ['mine-64x3-64x2', 'mine-64x4-150x1', 'mine-64x3-543-62x1', 'mine-64x3-543-62x2-dr20',
                         'mine-64x3-543-62x2', 'full_image-64x3-543-62x2', 'full_image-64x3-345-62x2', 'full_image-64x3-357-62x2', 'aug_full_image-64x3-357-62x2', 'cropped']
    Tensorboard.make(paths=tensorboard_paths,
                     names=tensorboard_names,
                     host='127.0.0.1',
                     _print=True)

    epochs = 50000
    use_batch_norm = True
    use_dropout = True

    learning_rate = 0.001

    # Conv2d inputs
    #     filters : Integer, dimensionality of the output space (ie. the number of filters in the convolution)
    #     kernel_size : An integer or tuple/list of 2 integers, specifying the height and width of the 2D convolution window
    #                   Can be a single integer to specify the same value for all spatial dimensions
    #     strides : An integer or tuple/list of 2 integers, specifying the strides of the convolution along the height and width
    #               Can be a single integer to specify the same value for all spatial dimensions
    conv2d_specifications = [[{'filters': 64, 'kernel_size': [3, 3], 'strides': (1, 1)}],
                             [{'filters': 64, 'kernel_size': [5, 5], 'strides': (1, 1)}],
                             [{'filters': 64, 'kernel_size': [7, 7], 'strides': (1, 1)}]]

    # Max Pool inputs
    #     pool_size : An integer or tuple/list of 2 integers: (pool_height, pool_width) specifying the size of the pooling window
    #                 Can be a single integer to specify the same value for all spatial dimensions
    #     strides : n integer or tuple/list of 2 integers, specifying the strides of the pooling operation
    #               Can be a single integer to specify the same value for all spatial dimensions
    max_pool_specifications = [[{'use': True, 'pool_size': [3, 3], 'strides': [1, 1]}],
                               [{'use': True, 'pool_size': [3, 3], 'strides': [1, 1]}],
                               [{'use': True, 'pool_size': [3, 3], 'strides': [1, 1]}]]

    # Dropout inputs
    #     use : to use dropout in this layer
    #     rate : dropout rate
    dropout_parameters = [{'use': True, 'rate': 0.5},
                          {'use': True, 'rate': 0.5}]

    # Fully Connected Layers unit size
    num_dense = 2
    fc_parameters = [{'units': 62}, {'units': 62}]

    # Data
    #------------------------------------------------------------------------------

    dataset = fetch_lfw_people(data_home=None,
                               resize=1.0,
                               color=True,
                               download_if_missing=True,
                               min_faces_per_person=20)

    images = dataset.images
    labels = dataset.target

    data_shape = [63, 63, 3]
    aug = Augment()
    model = Model(sess=tf.Session(),
                  data_shape=data_shape,
                  num_classes=len(set(labels)),
                  num_dense=num_dense,
                  learning_rate=learning_rate,
                  use_batch_norm=use_dropout,
                  use_dropout=use_dropout,
                  conv_parameters=conv2d_specifications,
                  max_pool_parameters=max_pool_specifications,
                  dropout_parameters=dropout_parameters,
                  fc_parameters=fc_parameters,
                  tensorboard_directory=tensorboard_directory)

    labels_encoded = np.zeros((len(labels), len(set(labels))))
    labels_encoded[np.arange(len(labels)), labels] = 1

    croped = []
    for _ in range(2):
        croped.append(aug.randomCropAll(images, 63, 63))

    # a1 = aug.augment(images=images,
    #                  operations=['flip_h', 'brightness', 'hue'],
    #                  width=63,
    #                  height=63)
    #
    # a2 = aug.augment(images=images,
    #                  operations=['flip_h', 'hue'],
    #                  width=63,
    #                  height=63)
    #
    # a3 = aug.augment(images=images,
    #                  operations=['flip_h', 'contrast', 'hue'],
    #                  width=63,
    #                  height=63)

    images_selected = np.concatenate((croped[0], croped[1]), axis=0)
    labels_selected = np.concatenate([labels_encoded, labels_encoded], axis=0)
    if type(images_selected).__module__ is not np.__name__:
        images_selected = np.array(images_selected)
    if type(labels_selected).__module__ is not np.__name__:
        labels_selected = np.array(labels_selected)

    print('> Data Shape: {}'.format(images_selected.shape))

    X_train, X_test, y_train, y_test = train_test_split(images_selected, labels_selected, test_size=0.30)
    X_train, X_val, y_train, y_val = train_test_split(images_selected, labels_selected, test_size=0.30)

    print('> Number of Classes: {}'.format(len(set(labels))))

    batch_size = 30

    model.train_data(data=X_train,
                     labels=y_train)
    model.val_data(data=X_val,
                   labels=y_val)
    model.test_data(data=X_test,
                    labels=y_test)

    model.train(batch_size=batch_size,
                batch_size_val=batch_size,
                epochs=epochs,
                is_restore=restore)

    return tensorboard_directory


if __name__ == '__main__':
    Fire(train)
