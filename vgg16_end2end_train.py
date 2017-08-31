import numpy as np
import tensorflow as tf
import os

from scipy.io import savemat
from scipy.io import loadmat

from vgg16_faces import *

class vgg16_face_classifier:
    def __init__(self, face_data, test_size):
        self.face_data = face_data
        self.labels={}
        for i, k in enumerate(self.face_data.keys()):
            self.labels[i] = k
        self.test_size = test_size
        self.no_classes = len(self.face_data)   
        
    def train_graph(self, rate, decay_lam=0):
        self.input_shape = list(self.face_data.values())[0][0].shape
        self.images = tf.placeholder(tf.float32, shape = [None]+list(self.input_shape))
        self.labels_1hot = tf.placeholder(tf.float32, shape=[None, self.no_classes])
        self.train_layers = vgg16(self.images, self.no_classes, train_indicator=True)
        
        # cost function
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = self.labels_1hot, logits=self.train_layers.fc3l))
        decay_penalty = decay_lam*(tf.reduce_sum(tf.square(self.train_layers.fc1w))+tf.reduce_sum(tf.square(self.train_layers.fc3w))) \
                                #+ decay_lam*tf.reduce_sum(tf.square(W_fc2))
        self.cost = cross_entropy + decay_penalty
        self.train_step = tf.train.MomentumOptimizer(rate,1.0).minimize(self.cost)
        
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
            artist_data = images[self.labels[i]]
            validation_class_set = artist_data[train_size:train_size+validation_size]
            validation_set = np.concatenate((validation_set, validation_class_set), axis=0)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            self.train_layers.load_weights('vgg16_weights.npz', 'fcl_weights.npy', sess)
    
            for j in range(iter):
        
                # minibatch training
                fullbatch = np.zeros([0] + list(self.input_shape))
                for i in range(self.no_classes):
                    artist_data = self.face_data[self.labels[i]]
                    class_batch = artist_data[np.random.choice(csize, csize_per_batch)]
                    fullbatch = np.concatenate((fullbatch, class_batch), axis=0)
            
                self.train_step.run(feed_dict={self.images:fullbatch, self.labels_1hot:train_classes_1hot})
    
    
                weight_names = ['conv1_1_W', 'conv1_1_b', 'conv1_2_W', 'conv1_2_b', 'conv2_1_W', 'conv2_1_b', 'conv2_2_W', \
                                    'conv2_2_b', 'conv3_1_W', 'conv3_1_b', 'conv3_2_W', 'conv3_2_b', 'conv3_3_W', 'conv3_3_b', \
                                    'conv4_1_W', 'conv4_1_b', 'conv4_2_W', 'conv4_2_b', 'conv4_3_W', 'conv4_3_b', 'conv5_1_W', \
                                    'conv5_1_b', 'conv5_2_W', 'conv5_2_b', 'conv5_3_W', 'conv5_3b', 'fc1b', 'fc1w', 'fc3b', 'fc3w']
                best_validation_cost = 1E10
                best_validation_accuracy = 0
                if (j+1)%5 == 0:
                    print('iteration'+str(j+1))
                    train_accuracies = []
                    for i in range(train_set.shape[0]//100):
                        train_accuracies.append(sess.run(self.accuracy, feed_dict={self.images:train_set[100*i:100*(i+1)], self.labels_1hot:classes_1hot[100*i:100*(i+1)]}))
                    train_accuracy = sum(train_accuracies)/float(len(train_accuracies))
                    validation_accuracy = sess.run(self.accuracy, feed_dict={self.images:validation_set, self.labels_1hot:validation_classes_1hot})
                    validation_cost = sess.run(self.cost, feed_dict={self.images:validation_set, self.labels_1hot:validation_classes_1hot})
                    print('training accuracy is {}'.format(train_accuracy))
                    print('validation accuracy is {}'.format(validation_accuracy))
                    print('validation cost is {}'.format(validation_cost))
                    if validation_accuracy > best_validation_accuracy:
                        best_validation_accuracy = validation_accuracy
                        best_weights = {}
                        for i in range(len(self.train_layers.all_parameters)):
                            best_weights[weight_names[i]] = sess.run(self.train_layers.all_parameters[i])

                            #savemat('fcl_weights', best_weights)
            np.save('best_vgg16_weights.npy', best_weights) 
        
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
            self.load_weights('best_vgg16_weights.npy', sess)
            test_accuracy = sess.run(self.accuracy, feed_dict={self.images:test_set, self.labels_1hot:test_classes_1hot})
            print('test accuracy is {}'.format(test_accuracy))
        
images = loadmat('image_set.mat')
del images['__header__']
del images['__version__']
del images['__globals__']

np.random.seed(42)
for artist in images:
    np.random.shuffle(images[artist])
classifier1 = vgg16_face_classifier(images, 15)
classifier1.train_graph(1E-6)
classifier1.predict_graph()
classifier1.train(70, 15, 50, iter=50)

classifier1.test()
        
        
        
        
        
        
        
        
        
        