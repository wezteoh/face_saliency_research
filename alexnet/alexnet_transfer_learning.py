import numpy as np
import tensorflow as tf
import os
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import _pickle as cPickle

from scipy.io import savemat
from scipy.io import loadmat

from alexnet_face_classifier import * 
from utils import *


class transfer_learning_graph:
    def __init__(self, num_classes, nhid, cnn):
        self.num_classes = num_classes
        self.f_maps = tf.placeholder(tf.float32, shape = [None, 43264])
        self.keep_prob = tf.placeholder(tf.float32)
        self.labels_1hot = tf.placeholder(tf.float32, shape=[None, self.num_classes])
        self.cnn = cnn(None, self.f_maps, self.num_classes, self.keep_prob)
        self.cnn.fc_layers(transfer_learning=True, nhid=nhid) 
        
    def train_graph(self, rate, decay_lam=0):    
        # cost function
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = self.labels_1hot, logits=self.cnn.fc2))
        decay_penalty = decay_lam*(tf.reduce_sum(tf.square(self.cnn.fc1W))+tf.reduce_sum(tf.square(self.cnn.fc2W)))
        self.cost = cross_entropy + decay_penalty
        self.train_step = tf.train.AdamOptimizer(rate).minimize(self.cost)
        
    def predict_graph(self):
        correct_prediction = tf.equal(tf.argmax(self.cnn.fc2,1), tf.argmax(self.labels_1hot,1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        


def train(graph, batch_size, training_set, training_labels, validation_set, validation_labels, decay_lambda, rate, keep_prob, iter, sess):
    
    # create learning graph
    graph.train_graph(rate, decay_lambda)
    sess.run(tf.global_variables_initializer())   
    
    # keep track of best performance
    best_validation_cost = 1E10
    best_validation_accuracy = 0
    
    # keep track of learning curves        
    train_accuracies = []
    train_costs = []
            
    validation_accuracies = []
    validation_costs = []
    
    weight_names = ['fc1W', 'fc1b', 'fc2W', 'fc2b']
    
    for j in range(iter):
        
        batch_xs, batch_ys = get_batch(training_set, training_labels, batch_size, graph.num_classes)
        graph.train_step.run(feed_dict={graph.f_maps:batch_xs, graph.labels_1hot:batch_ys, graph.keep_prob:keep_prob})
        
        # if j%2 == 0:
        #     batch_xs, batch_ys = get_batch(training_set, training_labels, batch_size, graph.num_classes)
        #     graph.train_step.run(feed_dict={graph.f_maps:batch_xs, graph.labels_1hot:batch_ys, graph.keep_prob:keep_prob})
        # 
        # else:
        #     batch_xs, batch_ys = get_random_batch(training_set, training_labels, batch_size)
        #     graph.train_step.run(feed_dict={graph.f_maps:batch_xs, graph.labels_1hot:batch_ys, graph.keep_prob:keep_prob})
            
        
        # evaluate every 5 steps
        if (j+1)%5 == 0:
            print('iteration'+str(j+1))
            train_accuracy = sess.run(graph.accuracy, feed_dict={graph.f_maps:batch_xs, graph.labels_1hot:batch_ys, graph.keep_prob:1.0})
            train_cost = sess.run(graph.cost, feed_dict={graph.f_maps:batch_xs, graph.labels_1hot:batch_ys, graph.keep_prob:1.0})/batch_size
            validation_accuracy = sess.run(graph.accuracy, feed_dict={graph.f_maps:validation_set, graph.labels_1hot:validation_labels, graph.keep_prob:1.0})
            validation_cost = sess.run(graph.cost, feed_dict={graph.f_maps:validation_set, graph.labels_1hot:validation_labels, graph.keep_prob:1.0})
            print('est training accuracy is {}'.format(train_accuracy))
            print('est training cost is {}'.format(train_cost))
            print('validation accuracy is {}'.format(validation_accuracy))
            print('validation cost is {}'.format(validation_cost))
                    
            train_accuracies.append(train_accuracy)
            validation_accuracies.append(validation_accuracy)
            train_costs.append(train_cost)
            validation_costs.append(validation_cost)        
            
            # keep track of weight data for best performance
            if validation_accuracy >= best_validation_accuracy:
                best_validation_accuracy = validation_accuracy
                best_validation_cost = validation_cost
                best_weights = {}
                for i in range(len(graph.cnn.fc_parameters)):
                    best_weights[weight_names[i]] = sess.run(graph.cnn.fc_parameters[i])

        
    # plot learning curves
    cPickle.dump(best_weights, open('transfer_learning_fc_weights.pkl', 'wb')) 
    f1 = plt.figure(1)
    plt.plot(range(5, iter+1, 5), train_accuracies, color='blue', linestyle='solid')
    plt.plot(range(5, iter+1, 5), validation_accuracies, color='red', linestyle='solid')
    f1.savefig("tl_accuracies_10faces.pdf", bbox_inches='tight')
            
    f2 = plt.figure(2)
    plt.plot(range(5, iter+1, 5), train_costs, color='blue', linestyle='solid')
    plt.plot(range(5, iter+1, 5), validation_costs, color='red', linestyle='solid')
    f2.savefig("tl_costs_10faces.pdf", bbox_inches='tight')
         
    print('best validation accuracy is {}'.format(best_validation_accuracy))
    print('best validation cost is {}'.format(best_validation_cost))
    print('corresponding training accuracy is {}'.format(sess.run(graph.accuracy, feed_dict={graph.f_maps:batch_xs, graph.labels_1hot:batch_ys, graph.keep_prob:1.0})))
    print('corresponding training cost is {}'.format(sess.run(graph.cost, feed_dict={graph.f_maps:batch_xs, graph.labels_1hot:batch_ys, graph.keep_prob:1.0})))
    
    
    
def test(graph, test_set, test_labels, weight_file, sess):
    graph.cnn.load_weights(weight_file, sess, fc_only=True)
    test_accuracy = sess.run(graph.accuracy, feed_dict={graph.f_maps:test_set, graph.labels_1hot:test_labels, graph.keep_prob:1.0})
    print('test accuracy is {}'.format(test_accuracy))
    
    
    
    



###
full_sets, label_sets = get_data_and_labels('training_set_10_conv5.pkl', 'validation_set_10_conv5.pkl', 'test_set_10_conv5.pkl')
training_set, validation_set, test_set = full_sets
training_labels, validation_labels, test_labels = label_sets

    
tl_graph = transfer_learning_graph(10, 80, alexnet_face_classifier)
tl_graph.predict_graph()

with tf.Session() as sess:
    train(tl_graph, 30, training_set, training_labels, validation_set, validation_labels, 1E-2, 5E-4, 0.5, 1000, sess)
    
with tf.Session() as sess:
    test(tl_graph, test_set, test_labels, 'transfer_learning_fc_weights.pkl', sess)
    

        
        
        
        
    