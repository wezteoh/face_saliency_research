import numpy as np
import tensorflow as tf
import os

from scipy.io import savemat
from scipy.io import loadmat

from tl_vgg16_fcl import * 


class tl_face_classifier:
    def __init__(self, face_data, test_size):
        self.face_data = face_data
        self.labels={}
        for i, k in enumerate(self.face_data.keys()):
            self.labels[i] = k
        self.test_size = test_size
        self.no_classes = len(self.face_data)    
        
    def train_graph(self, rate, decay_lam=0):
        self.input_shape = list(self.face_data.values())[0][0].shape
        self.f_maps = tf.placeholder(tf.float32, shape = [None]+list(self.input_shape))
        self.labels_1hot = tf.placeholder(tf.float32, shape=[None, self.no_classes])
        self.train_layers = tl_layers(self.f_maps, self.no_classes, train_indicator=True)
        
        # cost function
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = self.labels_1hot, logits=self.train_layers.fc3l))
        decay_penalty = decay_lam*(tf.reduce_sum(tf.square(self.train_layers.fc1w))+tf.reduce_sum(tf.square(self.train_layers.fc3w))) \
                                #+ decay_lam*tf.reduce_sum(tf.square(W_fc2))
        self.cost = cross_entropy + decay_penalty
        self.train_step = tf.train.GradientDescentOptimizer(rate).minimize(self.cost)
        
    def predict_graph(self):
        correct_prediction = tf.equal(tf.argmax(self.train_layers.fc3l,1), tf.argmax(self.labels_1hot,1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def train(self, train_size, validation_size, batch_size, iter=300):        
        # minibatch labels
        csize_per_batch = batch_size//self.no_classes
        train_classes_1hot = np.zeros([0,self.no_classes])
        for i in range(self.no_classes):
            train_class_1hot = np.zeros([csize_per_batch, self.no_classes])
            train_class_1hot[:,i]=1 
            train_classes_1hot = np.vstack((train_classes_1hot, train_class_1hot))

        # full training set
        csize = train_size
        classes_1hot = np.zeros([0,self.no_classes])
        for i in range(self.no_classes):
            class_1hot = np.zeros([csize, self.no_classes])
            class_1hot[:,i]=1 
            classes_1hot = np.vstack((classes_1hot, class_1hot))

        train_set = np.zeros([0] + list(self.input_shape))
        for i in range(self.no_classes):
            artist_data = self.face_data[self.labels[i]]
            train_class_set = artist_data[:csize]
            train_set = np.concatenate((train_set, train_class_set), axis=0)

        # full validation set
        validation_classes_1hot = np.zeros([0,self.no_classes])
        for i in range(self.no_classes):
            validation_class_1hot = np.zeros([validation_size, self.no_classes])
            validation_class_1hot[:,i]=1 
            validation_classes_1hot = np.vstack((validation_classes_1hot, validation_class_1hot))

        validation_set = np.zeros([0] + list(self.input_shape))
        for i in range(self.no_classes):
            artist_data = feature_maps[self.labels[i]]
            validation_class_set = artist_data[train_size:train_size+validation_size]
            validation_set = np.concatenate((validation_set, validation_class_set), axis=0)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            #self.train_layers.load_weights('vgg16_weights.npz', sess)
    
            for j in range(iter):
        
                # minibatch training
                fullbatch = np.zeros([0] + list(self.input_shape))
                for i in range(self.no_classes):
                    artist_data = self.face_data[self.labels[i]]
                    class_batch = artist_data[np.random.choice(csize, csize_per_batch)]
                    fullbatch = np.concatenate((fullbatch, class_batch), axis=0)
            
                self.train_step.run(feed_dict={self.f_maps:fullbatch, self.labels_1hot:train_classes_1hot})
    
                weight_names = ['fc1b', 'fc1w', 'fc3b', 'fc3w']
                best_validation_cost = 1E10
                if (j+1)%10 == 0:
                    print('iteration'+str(j+1))
                    train_accuracy = sess.run(self.accuracy, feed_dict={self.f_maps:train_set, self.labels_1hot:classes_1hot})
                    validation_accuracy = sess.run(self.accuracy, feed_dict={self.f_maps:validation_set, self.labels_1hot:validation_classes_1hot})
                    validation_cost = sess.run(self.cost, feed_dict={self.f_maps:validation_set, self.labels_1hot:validation_classes_1hot})
                    print('training accuracy is {}'.format(train_accuracy))
                    print('validation accuracy is {}'.format(validation_accuracy))
                    print('validation cost is {}'.format(validation_cost))
                    if validation_cost < best_validation_cost:
                        best_validation_cost = validation_cost
                        best_weights = {}
                        for i in range(len(self.train_layers.parameters)):
                            best_weights[weight_names[i]] = sess.run(self.train_layers.parameters[i])
                            #savemat('fcl_weights', best_weights)
            np.save('fcl_weights.npy', best_weights)               

    def load_weights(self, weight_file, sess):
        weights = np.load(weight_file).item()
        keys = sorted(weights.keys())
        for i in range(len(self.train_layers.parameters)):
            print (i, keys[i], np.shape(weights[keys[i]]))
            sess.run(self.train_layers.parameters[i].assign(weights[keys[i]]))
                              
    def test(self):
        # full test set
        test_classes_1hot = np.zeros([0,self.no_classes])
        for i in range(self.no_classes):
            test_class_1hot = np.zeros([self.test_size, self.no_classes])
            test_class_1hot[:,i]=1 
            test_classes_1hot = np.vstack((test_classes_1hot, test_class_1hot))

        test_set = np.zeros([0] + list(self.input_shape))
        for i in range(self.no_classes):
            artist_data = self.face_data[self.labels[i]]
            test_class_set = artist_data[self.face_data[self.labels[i]].shape[0]-self.test_size:]
            test_set = np.concatenate((test_set, test_class_set), axis=0)

        with tf.Session() as sess:
            self.load_weights('fcl_weights.npy', sess)
            test_accuracy = sess.run(self.accuracy, feed_dict={self.f_maps:test_set, self.labels_1hot:test_classes_1hot})
            print('test accuracy is {}'.format(test_accuracy))


feature_maps = loadmat('vgg_pool5_outputs.mat')
del feature_maps['__header__']
del feature_maps['__version__']
del feature_maps['__globals__']

np.random.seed(42)
for artist in feature_maps:
    np.random.shuffle(feature_maps[artist])
classifier1 = tl_face_classifier(feature_maps, 15)
classifier1.train_graph(0.001)
classifier1.predict_graph()
classifier1.train(70, 15, 50, iter=300)

classifier1.test()





