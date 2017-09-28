import numpy as np
import tensorflow as tf
import os

from scipy.io import savemat
from scipy.io import loadmat
from scipy.misc import imread
from scipy.misc import imsave

from vgg16_faces import *
import matplotlib.pyplot as plt
plt.switch_backend('agg')


class guided_backpropagator:
    def __init__(self, input_shape, no_classes):
        self.input_shape = input_shape
        self.no_classes = no_classes
    
    def classifier_graph(self):
        self.images = tf.placeholder(tf.float32, shape = [None]+self.input_shape)
        self.labels_1hot = tf.placeholder(tf.float32, shape=[None, self.no_classes])
        self.cnn_layers = vgg16(self.images, self.no_classes, train_indicator=False)
        t = 100
        self.probabilities = tf.nn.softmax(self.cnn_layers.fc3l/t)
        self.score = tf.tensordot(self.cnn_layers.fc3l, self.labels_1hot, axes=[[1],[1]])
        self.probability = tf.tensordot(self.probabilities, self.labels_1hot, axes=[[1],[1]])
        self.correct_prediction = tf.equal(tf.argmax(self.cnn_layers.fc3l,1), tf.argmax(self.labels_1hot,1))
        
    def load_weights(self, weight_file, sess):
        weights = np.load(weight_file).item()
        keys = sorted(weights.keys())
        for i in range(len(self.cnn_layers.parameters)):
            print (i, keys[i], np.shape(weights[keys[i]]))
            sess.run(self.cnn_layers.parameters[i].assign(weights[keys[i]]))
    
    def backprop_graph(self):
        grad_fc3l = tf.nn.relu(tf.gradients(self.probability,self.cnn_layers.fc3l)[0])
        grad_fc2 = tf.nn.relu(tf.gradients(self.cnn_layers.fc3l, self.cnn_layers.fc2, grad_ys=grad_fc3l)[0])
        grad_fc1 = tf.nn.relu(tf.gradients(self.cnn_layers.fc2, self.cnn_layers.fc1, grad_ys=grad_fc2)[0])
        grad_conv5_3 = tf.nn.relu(tf.gradients(self.cnn_layers.fc1, self.cnn_layers.conv5_3, grad_ys=grad_fc1)[0])
        grad_conv5_2 = tf.nn.relu(tf.gradients(self.cnn_layers.conv5_3, self.cnn_layers.conv5_2, grad_ys=grad_conv5_3)[0])
        grad_conv5_1 = tf.nn.relu(tf.gradients(self.cnn_layers.conv5_2, self.cnn_layers.conv5_1, grad_ys=grad_conv5_2)[0])
        grad_conv4_3 = tf.nn.relu(tf.gradients(self.cnn_layers.conv5_1, self.cnn_layers.conv4_3, grad_ys=grad_conv5_1)[0])
        grad_conv4_2 = tf.nn.relu(tf.gradients(self.cnn_layers.conv4_3, self.cnn_layers.conv4_2, grad_ys=grad_conv4_3)[0])
        grad_conv4_1 = tf.nn.relu(tf.gradients(self.cnn_layers.conv4_2, self.cnn_layers.conv4_1, grad_ys=grad_conv4_2)[0])
        grad_conv3_3 = tf.nn.relu(tf.gradients(self.cnn_layers.conv4_1, self.cnn_layers.conv3_3, grad_ys=grad_conv4_1)[0])
        grad_conv3_2 = tf.nn.relu(tf.gradients(self.cnn_layers.conv3_3, self.cnn_layers.conv3_2, grad_ys=grad_conv3_3)[0])
        grad_conv3_1 = tf.nn.relu(tf.gradients(self.cnn_layers.conv3_2, self.cnn_layers.conv3_1, grad_ys=grad_conv3_2)[0])
        grad_conv2_2 = tf.nn.relu(tf.gradients(self.cnn_layers.conv3_1, self.cnn_layers.conv2_2, grad_ys=grad_conv3_1)[0])
        grad_conv2_1 = tf.nn.relu(tf.gradients(self.cnn_layers.conv2_2, self.cnn_layers.conv2_1, grad_ys=grad_conv2_2)[0])
        grad_conv1_2 = tf.nn.relu(tf.gradients(self.cnn_layers.conv2_1, self.cnn_layers.conv1_2, grad_ys=grad_conv2_1)[0])
        grad_conv1_1 = tf.nn.relu(tf.gradients(self.cnn_layers.conv1_2, self.cnn_layers.conv1_1, grad_ys=grad_conv1_2)[0])
        self.grad_image = tf.nn.relu(tf.gradients(self.cnn_layers.conv1_1, self.images, grad_ys=grad_conv1_1)[0])

    
    def run_backprop(self, image, onehot, onehot_false, i, sess):
        saliency_map = sess.run(self.grad_image, feed_dict={self.images:image, self.labels_1hot:onehot})[0]
        saliency_map_false = sess.run(self.grad_image, feed_dict={self.images:image, self.labels_1hot:onehot_false})[0]
        saliency_map_scaled = saliency_map/np.max(saliency_map)
        saliency_map_false_scaled = saliency_map_false/np.max(saliency_map_false)
        combined_saliency = saliency_map_scaled*saliency_map_false_scaled
        probability = sess.run(self.probability, feed_dict={self.images:image, self.labels_1hot:onehot})
        probabilities = sess.run(self.probabilities, feed_dict={self.images:image, self.labels_1hot:onehot})
        correct_prediction = sess.run(self.correct_prediction, feed_dict={self.images:image, self.labels_1hot:onehot})
        #diff = saliency_map * (saliency_map_scaled > saliency_map_false_scaled
        plt.imsave('hathaway' + str(i) + '.png', saliency_map*5E4)
        plt.imsave('hathaway+hines' + str(i)  + '.png', combined_saliency*5E3)
        print(saliency_map)
        print(combined_saliency)
        #print(diff)
        print(sess.run(self.cnn_layers.fc3l, feed_dict={self.images:image, self.labels_1hot:onehot}))
        print(probability)
        print(probabilities)
        print(correct_prediction)
        

input_shape = [224, 224, 3]
no_classes = 10
images = loadmat('data/image_set_16faces.mat')

del images['__header__']
del images['__version__']
del images['__globals__']

names = ['James Marsden', 'Summer Glau', 'Marcia Cross', 'Courteney Cox', 'Hayden Christensen', 'Adam Brody']
for name in names:
    del images[name]

onehots={}
for i, k in enumerate(sorted(images.keys())):
    onehot_vec = np.zeros(no_classes)
    onehot_vec[i] = 1
    onehots[k] = onehot_vec

gb = guided_backpropagator(input_shape, no_classes)
gb.classifier_graph()
gb.backprop_graph()

with tf.Session() as sess:
    gb.load_weights('10faces/vgg16_weights_10faces_2.npy', sess)
    i=98
    onehot = [onehots['Anne Hathaway']]
    onehot_false = [onehots['Cheryl Hines']]
    image = [images['Anne Hathaway'][i]]
    gb.run_backprop(image, onehot, onehot_false, i, sess)



