from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim.nets import resnet_v2 as resnet
import numpy as np
import os


class Model(object):

    def __init__(self, sess, data_shape, num_classes, num_dense, batch_size, batch_size_val, epochs, learning_rate, use_batch_norm, use_dropout, tensorboard_directory):
        self.sess = sess
        self.data_shape = data_shape
        self.num_classes = num_classes
        self.num_dense = num_dense
        self.batch_size = batch_size
        self.batch_size_val = batch_size_val
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.use_batch_norm = use_batch_norm
        self.use_dropout = use_dropout
        self.tensorboard_directory = tensorboard_directory

        self.init_model()

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
        data_shape = self.data_shape
        data_shape.insert(0, None)

        self.x = tf.placeholder(dtype=tf.float32,
                                shape=[None,
                                       self.data_shape[0],
                                       self.data_shape[1],
                                       self.data_shape[2]])
        self.y = tf.placeholder(dtype=tf.float32,
                                shape=[None, self.num_classes])

        self.is_training = tf.placeholder(dtype=tf.bool,
                                          shape=None)

        net = self.x

        print('> Input Tensor: {}'.format(str(list(net.get_shape())).rjust(10, ' ')))

        net, predictions = resnet.resnet_v2_50(inputs=net,
                                               num_classes=self.num_classes,
                                               is_training=self.is_training,
                                               scope='resnet_v2_50')

        net = slim.flatten(inputs=net)

        for i in range(self.num_dense-1):
            net = slim.fully_connected(inputs=net,
                                       num_outputs=100,
                                       activation_fn=tf.nn.relu,
                                       trainable=True,
                                       scope='fc_{}'.format(i+1))
            print('> Fully Connected {}: {}'.format(i+1,
                                                    str(list(net.get_shape())).rjust(10, ' ')))
            if self.use_dropout:
                net = slim.dropout(inputs=net,
                                   keep_prob=0.5,
                                   is_training=self.is_training,
                                   scope=None)
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

        # Results
        #----------------------------------------------------------------------
        self.loss = slim.losses.softmax_cross_entropy(net, self.y)

        self.loss_summary = tf.summary.scalar(name='Loss',
                                              tensor=self.loss)

        self.predicted_indices = tf.argmax(input=net,
                                           axis=1)
        self.real_indices = tf.argmax(input=self.y,
                                      axis=1)

        self.accuracy = tf.cast(tf.equal(self.predicted_indices, self.real_indices),
                                dtype=tf.float32)
        self.accuracy = tf.reduce_mean(self.accuracy)
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

    def train(self, isRestore=True):

        tf.logging.set_verbosity(tf.logging.INFO)
        self.train_init()

        saver = tf.train.Saver()

        model_path = self.tensorboard_directory + '/model.ckpt'

        if isRestore:
            if os.path.isfile(model_path):
                saver = tf.train.import_meta_graph(model_path + '.meta')
                saver.restore(model_path)

        # TensorBoard & Saver Init
        if not os.path.exists(self.tensorboard_directory):
            os.makedirs(self.tensorboard_directory)
        train_writer, val_writer = [tf.summary.FileWriter(os.path.join(self.tensorboard_directory, phase),
                                                          self.sess.graph) for phase in ['train', 'val']]

        # self.sess.run(init_op)
        num_batches = int(len(self.labels) / self.batch_size)
        train_writer.add_graph(self.sess.graph)
        val_writer.add_graph(self.sess.graph)
        for epoch in range(1, self.epochs+1):
            for step in range(num_batches):
                # step += 1
                batch_x, batch_y = self.next_batch(self.batch_size, self.train_data, self.train_labels)

                # print('Batch x: {}'.format(str(list(batch_x.shape)).rjust(10, ' ')))
                # print('Batch y: {}'.format(str(list(batch_y.shape)).rjust(10, ' ')))

                loss, summary, _, = self.sess.run([self.loss, self.merged_summaries, self.optimizer],
                                                  feed_dict={self.is_training: True,
                                                             self.x: batch_x,
                                                             self.y: batch_y})
                if step+1 is num_batches:
                    # Output Loss to Terminal, Summary to TensorBoard
                    print("> Epoch: {} Loss: {}".format(epoch, round(loss, 5)))
                    train_writer.add_summary(summary, step)

            # Validation
            if epoch % 10 is 0:
                # Batch the validation
                num_val_batches = int(len(self.val_labels) / self.batch_size_val)
                running_loss, running_accuracy = 0, 0
                for val_step in range(num_val_batches):
                    val_x, val_y = self.next_batch(self.batch_size_val,
                                                   self.val_data,
                                                   self.val_labels)
                    loss, accuracy = self.sess.run([self.loss, self.accuracy],
                                                   feed_dict={self.is_training: False,
                                                              self.x: val_x,
                                                              self.y: val_y})

                    running_loss += loss
                    running_accuracy += accuracy

                accuracy = running_accuracy/num_val_batches

                val_summary = self.sess.run(self.val_summary,
                                            feed_dict={self.val_accuracy: accuracy})
                val_writer.add_summary(val_summary, epoch)
                print('> Validation: Epoch: {} Accuracy: {}'.format(
                    epoch, round(accuracy, 5)))
                print('--------------------------------------------------------')
                save_path = saver.save(self.sess, model_path, global_step=epoch)

            if epoch % 200 is 0:
                print('> Model Saved at {0}'.format(save_path))
                print('--------------------------------------------------------')
