from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim.nets import resnet_v2 as resnet
from tensorflow.python.saved_model import tag_constants
import numpy as np
import os

from utils.paths import Paths


class Model(object):

    def __init__(self, sess, data_shape, num_classes, num_dense, batch_size, batch_size_val, batch_size_test, epochs, learning_rate, use_batch_norm, use_dropout, dropout_parameters, fc_parameters, tensorboard_directory, val_epoch=10):
        self.sess = sess
        self.data_shape = data_shape
        self.num_classes = num_classes
        self.num_dense = num_dense
        self.batch_size = batch_size
        self.batch_size_val = batch_size_val
        self.batch_size_test = batch_size_test
        self.epochs = epochs
        self.val_epoch = val_epoch
        self.learning_rate = learning_rate
        self.use_batch_norm = use_batch_norm
        self.use_dropout = use_dropout
        self.dropout_parameters = dropout_parameters
        self.fc_parameters = fc_parameters
        self.tensorboard_directory = tensorboard_directory

        self.init_model()
        self.metrics()

    def train_data(self, data, labels):
        assert type(data).__module__ == np.__name__
        assert type(labels).__module__ == np.__name__

        self.train_data = data
        self.train_labels = labels

    def test_data(self, data, labels):
        assert type(data).__module__ == np.__name__
        assert type(labels).__module__ == np.__name__

        self.test_data = data
        self.test_labels = labels

    def val_data(self, data, labels):
        assert type(data).__module__ == np.__name__
        assert type(labels).__module__ == np.__name__

        self.val_data = data
        self.val_labels = labels

    def init_model(self):
        self.x = tf.placeholder(dtype=tf.float32,
                                shape=[None,
                                       self.data_shape[0],
                                       self.data_shape[1],
                                       self.data_shape[2]])
        self.y = tf.placeholder(dtype=tf.float32,
                                shape=[None, self.num_classes])

        self.is_training = tf.placeholder(dtype=tf.bool,
                                          shape=None)

        self.global_epoch = tf.get_variable('global_epoch', shape=[], dtype=tf.int32,
                                            initializer=tf.zeros_initializer, trainable=False)

        net = self.x

        print('> Input Tensor: {}'.format(str(list(net.get_shape())).rjust(10, ' ')))

        net, _ = resnet.resnet_v2_50(inputs=net,
                                     num_classes=self.num_classes,
                                     is_training=self.is_training,
                                     scope='resnet_v2_50')

        net = slim.flatten(inputs=net)

        for i, dropout_params, fc_params in zip(range(self.num_dense-1),
                                                self.dropout_parameters,
                                                self.fc_parameters):
            net = slim.fully_connected(inputs=net,
                                       num_outputs=fc_params['units'],
                                       activation_fn=tf.nn.relu,
                                       trainable=True,
                                       scope='fc_{}'.format(i+1))
            print('> Fully Connected {}: {}'.format(i+1,
                                                    str(list(net.get_shape())).rjust(10, ' ')))
            if self.use_dropout:
                net = slim.dropout(inputs=net,
                                   keep_prob=dropout_params['rate'],
                                   is_training=self.is_training,
                                   scope='dropout_{}'.format(i+1))
            # if self.use_batch_norm:
            net = slim.batch_norm(inputs=net,
                                  is_training=self.is_training)
            net = tf.nn.relu(net)

        net = slim.fully_connected(inputs=net,
                                   num_outputs=self.num_classes,
                                   activation_fn=None,
                                   scope='fc_{}'.format(self.num_dense))
        print('> Fully Connected {}: {}'.format(self.num_dense,
                                                str(list(net.get_shape())).rjust(10, ' ')))

        self.net = net

    def metrics(self):
        self.loss = tf.losses.softmax_cross_entropy(onehot_labels=self.y,
                                                    logits=self.net)

        self.loss_summary = tf.summary.scalar(name='Loss',
                                              tensor=self.loss)

        self.predicted = tf.argmax(input=self.net,
                                   axis=1,
                                   name='predicted')
        self.actual = tf.argmax(input=self.y,
                                axis=1,
                                name='actual')

        self.accuracy = tf.cast(tf.equal(self.predicted, self.actual),
                                dtype=tf.float32)
        self.accuracy = tf.reduce_mean(self.accuracy)

        # self.accuracy = tf.metrics.accuracy(predictions=self.predicted,
        #                                     labels=self.actual)
        self.accuracy_summary = tf.summary.scalar(name='Accuracy',
                                                  tensor=self.accuracy)

        self.merged_summaries = tf.summary.merge(inputs=[self.loss_summary, self.accuracy_summary])
        self.val_accuracy = tf.placeholder(dtype=tf.float32, shape=None)
        self.val_summary = tf.summary.scalar(name='Val Accuracy',
                                             tensor=self.val_accuracy)

    def train_init(self):

        model_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.optimizer = tf.train.AdamOptimizer().minimize(self.loss,
                                                               var_list=model_variables)
        self.sess.run(tf.variables_initializer(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)))

    def train(self, is_restore=True):
        last_epoch = 1

        tf.logging.set_verbosity(tf.logging.INFO)
        self.train_init()

        # Initalize Saver
        saver = tf.train.Saver()

        # If the tensorboard directory does not exist make it
        # Else if the user wishes to restore, restore the model
        if not Paths.exists(self.tensorboard_directory):
            Paths.make_dir(self.tensorboard_directory)
        elif is_restore:
            try:
                restore_path = tf.train.latest_checkpoint(checkpoint_dir=self.tensorboard_directory + '/model/')
                if not restore_path:
                    ValueError('Restore Path is not valid: {}'.format(repr(restore_path)))
                saver.restore(sess=self.sess,
                              save_path=restore_path)
                last_epoch = self.sess.run(self.global_epoch)
            except:
                IOError('Failed to restore from checkpoint')

        train_writer, val_writer = [tf.summary.FileWriter(os.path.join(
            self.tensorboard_directory, phase), self.sess.graph) for phase in ['train', 'val']]

        # self.sess.run(init_op)
        num_batches = int(len(self.train_labels) / self.batch_size)
        global_epoch = None
        train_writer.add_graph(self.sess.graph)
        val_writer.add_graph(self.sess.graph)

        print('--------------------------------------------------------')
        print('> Begin Training ...')
        print('--------------------------------------------------------')

        for epoch in range(last_epoch, self.epochs+1):

            # If global_epoch is not defined then, initalize global_epoch
            # global_epoch will not exist if training the model from scratch
            if global_epoch:
                global_epoch = self.sess.run(self.global_epoch) - 1
            else:
                global_epoch = self.sess.run(self.global_epoch)

            for step in range(num_batches):
                # step += 1
                batch_x, batch_y = next_batch(self.batch_size,
                                              self.train_data,
                                              self.train_labels)

                loss, summary, _, = self.sess.run([self.loss, self.merged_summaries, self.optimizer],
                                                  feed_dict={self.is_training: True,
                                                             self.x: batch_x,
                                                             self.y: batch_y})

            # Output Loss to Stdout, Summary to TensorBoard
            print("> Global Epoch: {} Epoch: {} Loss: {}".format(
                str(global_epoch).ljust(len(str(abs(self.epochs)))),
                str(epoch).ljust(len(str(abs(self.epochs)))),
                round(loss, 7)))
            train_writer.add_summary(summary, step)

            # Validation
            if epoch % self.val_epoch is 0:
                val = self.validation()
                val_writer.add_summary(val['summary'], epoch)
                print('> Validation: Epoch: {} Accuracy: {}'.format(epoch,
                                                                    round(val['accuracy'], 5)))
                print('--------------------------------------------------------')

                self.sess.run(self.global_epoch.assign(global_epoch + self.val_epoch + 1))

                save_path = saver.save(sess=self.sess,
                                       save_path=self.tensorboard_directory + '/model/model',
                                       global_step=epoch)

                print('> Model Saved at {0}'.format(save_path))
                print('--------------------------------------------------------')

    def validation(self):
        num_val_batches = int(len(self.val_labels) / self.batch_size_val)
        running_loss, running_accuracy = 0, 0
        for val_step in range(num_val_batches):
            val_x, val_y = next_batch(self.batch_size_val,
                                      self.val_data,
                                      self.val_labels)
            loss, accuracy = self.sess.run([self.loss, self.accuracy],
                                           feed_dict={self.is_training: False,
                                                      self.x: val_x,
                                                      self.y: val_y})

            running_loss += loss
            running_accuracy += accuracy

        accuracy = running_accuracy / num_val_batches

        val_summary = self.sess.run(self.val_summary,
                                    feed_dict={self.val_accuracy: accuracy})

        return {'accuracy': accuracy, 'summary': val_summary}

    def test(self):
        num_val_batches = int(len(self.test_labels) / self.batch_size_test)
        running_loss, running_accuracy = 0, 0
        for val_step in range(num_val_batches):
            val_x, val_y = next_batch(self.batch_size_val,
                                      self.val_data,
                                      self.val_labels)
            loss, accuracy = self.sess.run([self.loss, self.accuracy],
                                           feed_dict={self.is_training: False,
                                                      self.x: val_x,
                                                      self.y: val_y})

            running_loss += loss
            running_accuracy += accuracy

        accuracy = running_accuracy / num_val_batches

        val_summary = self.sess.run(self.val_summary,
                                    feed_dict={self.val_accuracy: accuracy})

        return {'accuracy': accuracy, 'summary': val_summary}


def next_batch(batch_size, data, labels):
    # Randomly select data for the next batch
    # Random sampling with replacement
    index = np.arange(0, len(labels))
    np.random.shuffle(index)

    index = index[:batch_size]
    data_shuffled = [data[i] for i in index]
    labels_shuffled = [labels[i] for i in index]

    return np.asarray(data_shuffled), np.asarray(labels_shuffled)
