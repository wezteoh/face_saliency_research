from pylab import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import time
import matplotlib.image as mpimg
import urllib
from numpy import random
import _pickle as cPickle
import os
t = int(time.time())
random.seed(t)
import tensorflow as tf
num_P = 13*13*256
num_N = 12

names = ['carell','drescher','ferrera','chenoweth','baldwin','hader']
names =['drescher', 'ferrera', 'chenoweth', 'baldwin', 'hader', 'carell', 'butler', 'radcliffe', 'vartan', 'bracco', 'gilpin', 'harmon']

def get_train_batch(M, N):
    # Get mini batches
    n = int(N/num_N)
    batch_xs = zeros((0, num_P))
    batch_y_s = zeros( (0, num_N))
    
    folder = 'training'
    for k in range(num_N):
        train_size = len(M[names[k]][folder])
        idx = array(random.permutation(train_size)[:n])
        batch_xs = vstack((batch_xs, ((array(M[names[k]][folder][idx]))  )))
        one_hot = np.zeros(num_N)
        one_hot[k] = 1
        batch_y_s = vstack((batch_y_s,   tile(one_hot, (n, 1))   ))
    return batch_xs, batch_y_s
    
def get_data(M, folder):
    # Get all data
    batch_xs = zeros((0, num_P))
    batch_y_s = zeros( (0, num_N))
    for k in range(num_N):
        batch_xs = vstack((batch_xs, ((array(M[names[k]][folder]))  )))
        one_hot = zeros(num_N)
        one_hot[k] = 1
        batch_y_s = vstack((batch_y_s,   tile(one_hot, (len(M[names[k]][folder]), 1))   ))
    return batch_xs, batch_y_s


def alexnet_train():
    f = open("images5.pkl","rb")
    M = cPickle.load(f)
    nhid = 10
    x = tf.placeholder(tf.float32, [None, num_P])
    keep_prob = tf.placeholder(tf.float32)
    
    init_std = 0.001
    
    W0 = tf.Variable(tf.random_normal([num_P, nhid], stddev=init_std))
    b0 = tf.Variable(tf.random_normal([nhid], stddev=init_std))
    
    W1 = tf.Variable(tf.random_normal([nhid, num_N], stddev=init_std))
    b1 = tf.Variable(tf.random_normal([num_N], stddev=init_std))
    
    layer1 = tf.nn.relu(tf.matmul(x, W0)+b0)
    drop_out = tf.nn.dropout(layer1, keep_prob)
    layer2 = tf.matmul(drop_out, W1) + b1
    #layer2 = tf.matmul(layer1, W1)+b1
    
    y = tf.nn.softmax(layer2)
    y_ = tf.placeholder(tf.float32, [None, num_N])
    
    lam = 0.01
    decay_penalty =lam*tf.reduce_sum(tf.square(W0))+lam*tf.reduce_sum(tf.square(W1))
    reg_NLL = -tf.reduce_sum(y_*tf.log(y))+decay_penalty
    
    stepsize = 0.001
    train_step = tf.train.AdamOptimizer(stepsize).minimize(reg_NLL)
    
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    
    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    test_x, test_y = get_data(M, 'test')
    v_x, v_y = get_data(M, 'validation')
    
    result_step_size = 25
    iteration = 200
    x_axis = np.arange(iteration//result_step_size) * result_step_size
    trainning_p = np.ones(iteration//result_step_size)
    valid_p = np.ones(iteration//result_step_size)
    test_p = np.ones(iteration//result_step_size)
    batchsize = 30
    for i in range(iteration):
        #print i  
        batch_xs, batch_ys = get_train_batch(M, num_N * batchsize)
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys, keep_prob : 1})
        
        
        if i % result_step_size == 0:
            print("i=",i)
            print( "Test:", sess.run(accuracy, feed_dict={x: test_x, y_: test_y, keep_prob : 1}))
            batch_xs, batch_ys = get_data(M, 'training')
        
            print("Train:", sess.run(accuracy, feed_dict={x: batch_xs, y_: batch_ys, keep_prob : 1}))
            print("Validation:", sess.run(accuracy, feed_dict={x: v_x, y_: v_y, keep_prob : 1}))
            print("Penalty:", sess.run(decay_penalty))
            v_perform = sess.run(accuracy, feed_dict={x: v_x, y_: v_y, keep_prob : 1})
            
            trainning_p[i//result_step_size] -= sess.run(accuracy, feed_dict={x: batch_xs, y_: batch_ys, keep_prob : 1})
            test_p[i//result_step_size] -= sess.run(accuracy, feed_dict={x: test_x, y_: test_y, keep_prob : 1})
            valid_p[i//result_step_size] -= v_perform
        
            snapshot = {}
            snapshot["W0"] = sess.run(W0)
            snapshot["W1"] = sess.run(W1)
            snapshot["b0"] = sess.run(b0)
            snapshot["b1"] = sess.run(b1)
            # print(np.linalg.norm(snapshot["W0"]))
            #cPickle.dump(snapshot,  open("tf/relu"+str(i)+".pkl", "wb"))
    
    # Plot the Performances
    plt.plot(x_axis, trainning_p)
    plt.plot(x_axis, test_p)
    plt.plot(x_axis,  valid_p)
    plt.xlabel('Iteration')
    plt.ylabel('Error Performance')
    plt.legend(['Training Set','Test Set','Validation Set'])
    plt.savefig('conv5_'+str(nhid)+'_test_0.001'+'.png')
    close()
    cPickle.dump(snapshot,  open("relu_"+str(valid_p[i//result_step_size])+str(i)+".pkl", "wb"))
    
    
alexnet_train()