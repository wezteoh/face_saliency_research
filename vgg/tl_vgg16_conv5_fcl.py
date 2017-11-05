import numpy as np
import tensorflow as tf

class tl_layers:
    def __init__(self, f_maps, no_classes, start_weights=None, sess=None,train_indicator=False):
        self.f_maps = f_maps
        self.output_shape = no_classes 
        training = train_indicator
        self.parameters = []
        self.conv_layers(training=train_indicator)
        self.flat = self.flatten(self.pool5)
        self.fc_layers(training=train_indicator)    
        
    
    def conv_layers(self, training):
        # vgg16 conv5_1
        with tf.name_scope('conv5_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,\
                                                     stddev=1e-1), name='weights', trainable=training)
            conv = tf.nn.conv2d(self.f_maps, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),\
                                 trainable=training, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv5_1 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # vgg16 conv5_2
        with tf.name_scope('conv5_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,\
                                                     stddev=1e-1), name='weights', trainable=training)
            conv = tf.nn.conv2d(self.conv5_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),\
                                 trainable=training, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv5_2 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # vgg16 conv5_3
        with tf.name_scope('conv5_3') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,\
                                                     stddev=1e-1), name='weights', trainable=training)
            conv = tf.nn.conv2d(self.conv5_2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),\
                                 trainable=training, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv5_3 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # vgg16 pool5
        self.pool5 = tf.nn.max_pool(self.conv5_3,\
                               ksize=[1, 2, 2, 1],\
                               strides=[1, 2, 2, 1],\
                               padding='SAME',\
                               name='pool5')
    
    def load_weights(self, weight_file, sess):
        weights = np.load(weight_file)
        keys = sorted(weights.keys())
        for i in range(len(self.parameters)):
            print (i, keys[20+i], np.shape(weights[keys[20+i]]))
            sess.run(self.parameters[i].assign(weights[keys[20+i]]))
    
    def flatten(self, outputs):
        return tf.reshape(outputs, [-1, int(np.product(outputs.shape[1:]))])
    
    def fc_layers(self, training):
        # fc1
        with tf.name_scope('fc1') as scope:
            shape = int(self.flat.shape[1])
            self.fc1w = tf.Variable(tf.truncated_normal([shape, 4096],\
                                                         dtype=tf.float32,\
                                                         stddev=1e-1), name='weights', trainable=training)
            fc1b = tf.Variable(tf.constant(1.0, shape=[4096], dtype=tf.float32),
                                 trainable=training, name='biases')
            fc1l = tf.nn.bias_add(tf.matmul(self.flat, self.fc1w), fc1b)
            self.fc1 = tf.nn.relu(fc1l)

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



        
        
