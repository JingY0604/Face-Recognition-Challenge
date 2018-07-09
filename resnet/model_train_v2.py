from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split
import tensorflow as tf
import tensorflow.contrib.slim.nets as nets
import numpy as np

slim = tf.contrib.slim
resnet = nets.resnet_v2


train_log_dir = './tmp/tensorboard/first_training'

if not tf.gfile.Exists(train_log_dir):
    tf.gfile.MakeDirs(train_log_dir)

dataset = fetch_lfw_people(data_home=None,
                           # resize=0.6,
                           color=True,
                           download_if_missing=True,
                           min_faces_per_person=20)

images = dataset.images
labels = dataset.target

labels_encoded = np.zeros((len(labels), len(set(labels))))
labels_encoded[np.arange(len(labels)), labels] = 1

X_train, X_test, y_train, y_test = train_test_split(images, labels_encoded, test_size=0.30)
X_train, X_val, y_train, y_val = train_test_split(images, labels_encoded, test_size=0.30)


with tf.Graph().as_default():
    # Set up the data loading:
    images, labels = X_train, y_train

    # Define the model:
    predictions = resnet.resnet_v2_50(inputs=images,
                                      is_training=True)

    # Specify the loss function:
    slim.losses.softmax_cross_entropy(predictions, labels)

    total_loss = slim.losses.get_total_loss()
    tf.summary.scalar('losses/total_loss', total_loss)

    # Specify the optimization scheme:
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=.001)

    # create_train_op that ensures that when we evaluate it to get the loss,
    # the update_ops are done and the gradient updates are computed.
    train_tensor = slim.learning.create_train_op(total_loss, optimizer)

    # Actually runs training.
    slim.learning.train(train_tensor, train_log_dir)
