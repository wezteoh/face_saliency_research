import numpy as np
import tensorflow as tf

class vgg16:
    def __init__(self, imgs, no_classes, train_indicator=True):
        self.imgs = imgs
        self.parameters = []
        self.parameters2 = []
        self.output_shape = no_classes
        training = train_indicator
        self.preprocess()
        self.convlayers(training=train_indicator) 
        self.fc_layers(training=train_indicator)  
        self.all_parameters = self.parameters + self.parameters2       
    
    def preprocess(self):
        # zero-mean input
        with tf.name_scope('recentering') as scope:
            mean = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32, shape=[1, 1, 1, 3], name='img_mean')
            self.recentered_imgs = self.imgs-mean

    def convlayers(self, training=False):
        # conv1_1
        with tf.name_scope('conv1_1') as scope:
            self.conv1_1_W = tf.Variable(tf.truncated_normal([3, 3, 3, 64], dtype=tf.float32, stddev=1e-1),\
                                name='weights', trainable=training)
            conv = tf.nn.conv2d(self.recentered_imgs, self.conv1_1_W, [1, 1, 1, 1], padding='SAME')
            self.conv1_1_b = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32),\
                                 trainable=training, name='biases')
            out = tf.nn.bias_add(conv, self.conv1_1_b)
            self.conv1_1 = tf.nn.relu(out, name=scope)
            self.parameters += [self.conv1_1_W, self.conv1_1_b]

        # conv1_2
        with tf.name_scope('conv1_2') as scope:
            self.conv1_2_W = tf.Variable(tf.truncated_normal([3, 3, 64, 64], dtype=tf.float32,\
                                                     stddev=1e-1), name='weights', trainable=training)
            conv = tf.nn.conv2d(self.conv1_1, self.conv1_2_W, [1, 1, 1, 1], padding='SAME')
            self.conv1_2_b = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32),\
                                 trainable=training, name='biases')
            out = tf.nn.bias_add(conv, self.conv1_2_b)
            self.conv1_2 = tf.nn.relu(out, name=scope)
            self.parameters += [self.conv1_2_W, self.conv1_2_b]

        # pool1
        self.pool1 = tf.nn.max_pool(self.conv1_2,\
                               ksize=[1, 2, 2, 1],\
                               strides=[1, 2, 2, 1],\
                               padding='SAME',\
                               name='pool1')

        # conv2_1
        with tf.name_scope('conv2_1') as scope:
            self.conv2_1_W = tf.Variable(tf.truncated_normal([3, 3, 64, 128], dtype=tf.float32,\
                                                     stddev=1e-1), name='weights', trainable=training)
            conv = tf.nn.conv2d(self.pool1, self.conv2_1_W, [1, 1, 1, 1], padding='SAME')
            self.conv2_1_b = tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32),\
                                 trainable=training, name='biases')
            out = tf.nn.bias_add(conv,  self.conv2_1_b)
            self.conv2_1 = tf.nn.relu(out, name=scope)
            self.parameters += [self.conv2_1_W,  self.conv2_1_b]

        # conv2_2
        with tf.name_scope('conv2_2') as scope:
            self.conv2_2_W = tf.Variable(tf.truncated_normal([3, 3, 128, 128], dtype=tf.float32,\
                                                     stddev=1e-1), name='weights', trainable=training)
            conv = tf.nn.conv2d(self.conv2_1, self.conv2_2_W, [1, 1, 1, 1], padding='SAME')
            self.conv2_2_b = tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32),\
                                 trainable=training, name='biases')
            out = tf.nn.bias_add(conv, self.conv2_2_b)
            self.conv2_2 = tf.nn.relu(out, name=scope)
            self.parameters += [self.conv2_2_W, self.conv2_2_b]

        # pool2
        self.pool2 = tf.nn.max_pool(self.conv2_2,\
                               ksize=[1, 2, 2, 1],\
                               strides=[1, 2, 2, 1],\
                               padding='SAME',\
                               name='pool2')

        # conv3_1
        with tf.name_scope('conv3_1') as scope:
            self.conv3_1_W = tf.Variable(tf.truncated_normal([3, 3, 128, 256], dtype=tf.float32,\
                                                     stddev=1e-1), name='weights', trainable=training)
            conv = tf.nn.conv2d(self.pool2, self.conv3_1_W, [1, 1, 1, 1], padding='SAME')
            self.conv3_1_b = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                                 trainable=training, name='biases')
            out = tf.nn.bias_add(conv, self.conv3_1_b)
            self.conv3_1 = tf.nn.relu(out, name=scope)
            self.parameters += [self.conv3_1_W, self.conv3_1_b]

        # conv3_2
        with tf.name_scope('conv3_2') as scope:
            self.conv3_2_W = tf.Variable(tf.truncated_normal([3, 3, 256, 256], dtype=tf.float32,\
                                                     stddev=1e-1), name='weights', trainable=training)
            conv = tf.nn.conv2d(self.conv3_1, self.conv3_2_W, [1, 1, 1, 1], padding='SAME')
            self.conv3_2_b = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),\
                                 trainable=training, name='biases')
            out = tf.nn.bias_add(conv, self.conv3_2_b)
            self.conv3_2 = tf.nn.relu(out, name=scope)
            self.parameters += [self.conv3_2_W, self.conv3_2_b]

        # conv3_3
        with tf.name_scope('conv3_3') as scope:
            self.conv3_3_W = tf.Variable(tf.truncated_normal([3, 3, 256, 256], dtype=tf.float32,\
                                                     stddev=1e-1), name='weights', trainable=training)
            conv = tf.nn.conv2d(self.conv3_2, self.conv3_3_W, [1, 1, 1, 1], padding='SAME')
            self.conv3_3_b = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),\
                                 trainable=training, name='biases')
            out = tf.nn.bias_add(conv, self.conv3_3_b)
            self.conv3_3 = tf.nn.relu(out, name=scope)
            self.parameters += [self.conv3_3_W, self.conv3_3_b]

        # pool3
        self.pool3 = tf.nn.max_pool(self.conv3_3,\
                               ksize=[1, 2, 2, 1],\
                               strides=[1, 2, 2, 1],\
                               padding='SAME',\
                               name='pool3')

        # conv4_1
        with tf.name_scope('conv4_1') as scope:
            self.conv4_1_W = tf.Variable(tf.truncated_normal([3, 3, 256, 512], dtype=tf.float32,\
                                                     stddev=1e-1), name='weights', trainable=training)
            conv = tf.nn.conv2d(self.pool3, self.conv4_1_W, [1, 1, 1, 1], padding='SAME')
            self.conv4_1_b = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),\
                                 trainable=training, name='biases')
            out = tf.nn.bias_add(conv, self.conv4_1_b)
            self.conv4_1 = tf.nn.relu(out, name=scope)
            self.parameters += [self.conv4_1_W, self.conv4_1_b]

        # conv4_2
        with tf.name_scope('conv4_2') as scope:
            self.conv4_2_W = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,\
                                                     stddev=1e-1), name='weights', trainable=training)
            conv = tf.nn.conv2d(self.conv4_1, self.conv4_2_W, [1, 1, 1, 1], padding='SAME')
            self.conv4_2_b = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),\
                                 trainable=training, name='biases')
            out = tf.nn.bias_add(conv, self.conv4_2_b)
            self.conv4_2 = tf.nn.relu(out, name=scope)
            self.parameters += [self.conv4_2_W, self.conv4_2_b]

        # conv4_3
        with tf.name_scope('conv4_3') as scope:
            self.conv4_3_W = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,\
                                                     stddev=1e-1), name='weights',trainable=training)
            conv = tf.nn.conv2d(self.conv4_2, self.conv4_3_W, [1, 1, 1, 1], padding='SAME')
            self.conv4_3_b = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),\
                                 trainable=training, name='biases')
            out = tf.nn.bias_add(conv, self.conv4_3_b)
            self.conv4_3 = tf.nn.relu(out, name=scope)
            self.parameters += [self.conv4_3_W, self.conv4_3_b]

        # pool4
        self.pool4 = tf.nn.max_pool(self.conv4_3,\
                               ksize=[1, 2, 2, 1],\
                               strides=[1, 2, 2, 1],\
                               padding='SAME',\
                               name='pool4')

        # conv5_1
        with tf.name_scope('conv5_1') as scope:
            self.conv5_1_W = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,\
                                                     stddev=1e-1), name='weights', trainable=training)
            conv = tf.nn.conv2d(self.pool4, self.conv5_1_W, [1, 1, 1, 1], padding='SAME')
            self.conv5_1_b = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),\
                                 trainable=training, name='biases')
            out = tf.nn.bias_add(conv, self.conv5_1_b)
            self.conv5_1 = tf.nn.relu(out, name=scope)
            self.parameters += [self.conv5_1_W, self.conv5_1_b]

        # conv5_2
        with tf.name_scope('conv5_2') as scope:
            self.conv5_2_W = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,\
                                                     stddev=1e-1), name='weights', trainable=training)
            conv = tf.nn.conv2d(self.conv5_1, self.conv5_2_W, [1, 1, 1, 1], padding='SAME')
            self.conv5_2_b = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),\
                                 trainable=training, name='biases')
            out = tf.nn.bias_add(conv, self.conv5_2_b)
            self.conv5_2 = tf.nn.relu(out, name=scope)
            self.parameters += [self.conv5_2_W, self.conv5_2_b]

        # conv5_3
        with tf.name_scope('conv5_3') as scope:
            self.conv5_3_W = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,\
                                                     stddev=1e-1), name='weights', trainable=training)
            conv = tf.nn.conv2d(self.conv5_2, self.conv5_3_W, [1, 1, 1, 1], padding='SAME')
            self.conv5_3_b = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),\
                                 trainable=training, name='biases')
            out = tf.nn.bias_add(conv, self.conv5_3_b)
            self.conv5_3 = tf.nn.relu(out, name=scope)
            self.parameters += [self.conv5_3_W, self.conv5_3_b]

        # pool5
        self.pool5 = tf.nn.max_pool(self.conv5_3,\
                               ksize=[1, 2, 2, 1],\
                               strides=[1, 2, 2, 1],\
                               padding='SAME',\
                               name='pool5')

    def fc_layers(self, training):
        # fc1
        with tf.name_scope('fc1') as scope:
            shape = int(np.prod(self.pool5.get_shape()[1:]))
            self.fc1w = tf.Variable(tf.truncated_normal([shape, 4096],\
                                                         dtype=tf.float32,\
                                                         stddev=1e-1), name='weights', trainable=training)
            self.fc1b = tf.Variable(tf.constant(1.0, shape=[4096], dtype=tf.float32),
                                 trainable=training, name='biases')
            pool5_flat = tf.reshape(self.pool5, [-1, shape])
            fc1l = tf.nn.bias_add(tf.matmul(pool5_flat, self.fc1w), self.fc1b)
            self.fc1 = tf.nn.relu(fc1l)
            self.parameters += [self.fc1b, self.fc1w]

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
            self.fc3b = tf.Variable(tf.constant(1.0, shape=[self.output_shape], dtype=tf.float32),\
                                 trainable=training, name='biases')
            self.fc3l = tf.nn.bias_add(tf.matmul(self.fc1, self.fc3w), self.fc3b)
            self.parameters += [self.fc3b, self.fc3w]


    def load_weights(self, weight_file1, weight_file2, sess):
        weights = dict(np.load(weight_file1))
        weights.update(np.load(weight_file2).item())
        keys = sorted(weights.keys())
        for i in range(len(self.parameters)):
            print (i, keys[i], np.shape(weights[keys[i]]))
            sess.run(self.parameters[i].assign(weights[keys[i]]))
            
        
