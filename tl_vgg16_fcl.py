import numpy as np
import tensorflow as tf

class tl_layers:
    def __init__(self, f_maps, no_classes, keep_prob, train_indicator=False):
        self.f_maps = f_maps
        self.output_shape = no_classes 
        training = train_indicator
        self.parameters = []
        self.keep_prob = keep_prob
        self.fc_layers(training=train_indicator)
    
    def fc_layers(self, training):
        # fc1
        with tf.name_scope('fc1') as scope:
            shape = int(self.f_maps.shape[1])
            self.fc1w = tf.Variable(tf.truncated_normal([shape, 500],\
                                                         dtype=tf.float32,\
                                                         stddev=1e-1), name='weights', trainable=training)
            self.fc1b = tf.Variable(tf.constant(1.0, shape=[500], dtype=tf.float32),
                                 trainable=training, name='biases')
            fc1l = tf.nn.bias_add(tf.matmul(self.f_maps, self.fc1w), self.fc1b)
            self.fc1 = tf.nn.relu(fc1l)
            self.parameters += [self.fc1b, self.fc1w]

        # fc2
        with tf.name_scope('fc2') as scope:
            self.fc2w = tf.Variable(tf.truncated_normal([500, 500],
                                                         dtype=tf.float32,
                                                         stddev=1e-1), name='weights')
            self.fc2b = tf.Variable(tf.constant(1.0, shape=[500], dtype=tf.float32),
                                 trainable=True, name='biases')
            fc2l = tf.nn.bias_add(tf.matmul(self.fc1, self.fc2w), self.fc2b)
            self.fc2 = tf.nn.relu(fc2l)
            self.parameters += [self.fc2b, self.fc2w]

        # fc3
        with tf.name_scope('fc3') as scope:
            self.fc3w = tf.Variable(tf.truncated_normal([500, self.output_shape],\
                                                         dtype=tf.float32,\
                                                         stddev=1e-1), name='weights', trainable=training)
            self.fc3b = tf.Variable(tf.constant(1.0, shape=[self.output_shape], dtype=tf.float32),\
                                 trainable=training, name='biases')
            self.fc3l = tf.nn.bias_add(tf.matmul(self.fc2, self.fc3w), self.fc3b)
            self.parameters += [self.fc3b, self.fc3w]
            
        # Dropout Regularization
        self.fc3l_drop = tf.nn.dropout(self.fc3l, self.keep_prob)


        
        
