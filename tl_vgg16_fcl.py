import numpy as np
import tensorflow as tf

class fcl:
    def __init__(self, f_maps, no_classes, train_indicator=False):
        self.f_maps = f_maps
        self.output_shape = no_classes 
        training = train_indicator
        self.parameters = []
        self.fc_layers(training=train_indicator)
    
    def fc_layers(self, training):
        # fc1
        with tf.name_scope('fc1') as scope:
            shape = int(self.f_maps.shape[1])
            self.fc1w = tf.Variable(tf.truncated_normal([shape, 4096],\
                                                         dtype=tf.float32,\
                                                         stddev=1e-1), name='weights', trainable=training)
            fc1b = tf.Variable(tf.constant(1.0, shape=[4096], dtype=tf.float32),
                                 trainable=training, name='biases')
            fc1l = tf.nn.bias_add(tf.matmul(self.f_maps, self.fc1w), fc1b)
            self.fc1 = tf.nn.relu(fc1l)
            self.parameters += [self.fc1w, fc1b]

        # # fc2
        # with tf.name_scope('fc2') as scope:
        #     fc2w = tf.Variable(tf.truncated_normal([4096, 4096],
        #                                                  dtype=tf.float32,
        #                                                  stddev=1e-1), name='weights')
        #     fc2b = tf.Variable(tf.constant(1.0, shape=[4096], dtype=tf.float32),
        #                          trainable=True, name='biases')
        #     fc2l = tf.nn.bias_add(tf.matmul(self.fc1, fc2w), fc2b)
        #     self.fc2 = tf.nn.relu(fc2l)
        #     self.parameters += [fc2w, fc2b]

        # fc3
        with tf.name_scope('fc3') as scope:
            self.fc3w = tf.Variable(tf.truncated_normal([4096, self.output_shape],\
                                                         dtype=tf.float32,\
                                                         stddev=1e-1), name='weights', trainable=training)
            fc3b = tf.Variable(tf.constant(1.0, shape=[self.output_shape], dtype=tf.float32),\
                                 trainable=training, name='biases')
            self.fc3l = tf.nn.bias_add(tf.matmul(self.fc1, self.fc3w), fc3b)
            self.parameters += [self.fc3w, fc3b]



        
        