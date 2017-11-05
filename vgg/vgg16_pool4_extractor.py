import numpy as np
import tensorflow as tf

class vgg16_extractor:
    def __init__(self, imgs, weights=None, sess=None, train_indicator=False):
        self.imgs = imgs
        self.parameters = []
        self.preprocess()
        self.convlayers(training=train_indicator)          
        self.f_maps=self.pool4
        if weights is not None and sess is not None:
            self.load_weights(weights, sess)
        
    def preprocess(self):
        # zero-mean input
        with tf.name_scope('recentering') as scope:
            mean = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32, shape=[1, 1, 1, 3], name='img_mean')
            self.recentered_imgs = self.imgs-mean

    def convlayers(self, training=False):
        # conv1_1
        with tf.name_scope('conv1_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 3, 64], dtype=tf.float32, stddev=1e-1),\
                                name='weights', trainable=training)
            conv = tf.nn.conv2d(self.recentered_imgs, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32),\
                                 trainable=training, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv1_1 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv1_2
        with tf.name_scope('conv1_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 64, 64], dtype=tf.float32,\
                                                     stddev=1e-1), name='weights', trainable=training)
            conv = tf.nn.conv2d(self.conv1_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32),\
                                 trainable=training, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv1_2 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # pool1
        self.pool1 = tf.nn.max_pool(self.conv1_2,\
                               ksize=[1, 2, 2, 1],\
                               strides=[1, 2, 2, 1],\
                               padding='SAME',\
                               name='pool1')

        # conv2_1
        with tf.name_scope('conv2_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 64, 128], dtype=tf.float32,\
                                                     stddev=1e-1), name='weights', trainable=training)
            conv = tf.nn.conv2d(self.pool1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32),\
                                 trainable=training, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv2_1 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv2_2
        with tf.name_scope('conv2_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 128, 128], dtype=tf.float32,\
                                                     stddev=1e-1), name='weights', trainable=training)
            conv = tf.nn.conv2d(self.conv2_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32),\
                                 trainable=training, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv2_2 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # pool2
        self.pool2 = tf.nn.max_pool(self.conv2_2,\
                               ksize=[1, 2, 2, 1],\
                               strides=[1, 2, 2, 1],\
                               padding='SAME',\
                               name='pool2')

        # conv3_1
        with tf.name_scope('conv3_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 128, 256], dtype=tf.float32,\
                                                     stddev=1e-1), name='weights', trainable=training)
            conv = tf.nn.conv2d(self.pool2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                                 trainable=training, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv3_1 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv3_2
        with tf.name_scope('conv3_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 256], dtype=tf.float32,\
                                                     stddev=1e-1), name='weights', trainable=training)
            conv = tf.nn.conv2d(self.conv3_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),\
                                 trainable=training, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv3_2 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv3_3
        with tf.name_scope('conv3_3') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 256], dtype=tf.float32,\
                                                     stddev=1e-1), name='weights', trainable=training)
            conv = tf.nn.conv2d(self.conv3_2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),\
                                 trainable=training, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv3_3 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # pool3
        self.pool3 = tf.nn.max_pool(self.conv3_3,\
                               ksize=[1, 2, 2, 1],\
                               strides=[1, 2, 2, 1],\
                               padding='SAME',\
                               name='pool3')

        # conv4_1
        with tf.name_scope('conv4_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 512], dtype=tf.float32,\
                                                     stddev=1e-1), name='weights', trainable=training)
            conv = tf.nn.conv2d(self.pool3, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),\
                                 trainable=training, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv4_1 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv4_2
        with tf.name_scope('conv4_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,\
                                                     stddev=1e-1), name='weights', trainable=training)
            conv = tf.nn.conv2d(self.conv4_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),\
                                 trainable=training, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv4_2 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv4_3
        with tf.name_scope('conv4_3') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,\
                                                     stddev=1e-1), name='weights',trainable=training)
            conv = tf.nn.conv2d(self.conv4_2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),\
                                 trainable=training, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv4_3 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # pool4
        self.pool4 = tf.nn.max_pool(self.conv4_3,\
                               ksize=[1, 2, 2, 1],\
                               strides=[1, 2, 2, 1],\
                               padding='SAME',\
                               name='pool4')


    def load_weights(self, weight_file, sess):
        weights = np.load(weight_file)
        keys = sorted(weights.keys())
        for i in range(len(self.parameters)):
            print (i, keys[i], np.shape(weights[keys[i]]))
            sess.run(self.parameters[i].assign(weights[keys[i]]))


feature_maps = {}
            
if __name__ == '__main__':
    image_set = loadmat('data/image_set.mat')
    del image_set['__header__']
    del image_set['__version__']
    del image_set['__globals__']
    
    with tf.Session() as sess:
        
        inputs = tf.placeholder(tf.float32, shape = [None, 224, 224, 3])
        extractor = vgg16_extractor(inputs, 'data/vgg16_weights.npz', sess)
        f_maps= extractor.f_maps
    
        batch_size = 25    
        for artist in image_set.keys():
            feature_maps[artist] = np.zeros([0] + list(f_maps.shape)[1:]) 
            for i in range(int(image_set[artist].shape[0]/batch_size)):
                batch = image_set[artist][i*batch_size:(i+1)*batch_size]
                feature_maps[artist] = np.concatenate((feature_maps[artist], sess.run(f_maps, feed_dict={inputs: batch})),axis=0)
                print(str((i+1)*100/(image_set[artist].shape[0]/batch_size)) + '% completed for ' + str(artist)) 
                
        savemat('data/vgg_pool4_outputs_2', feature_maps)