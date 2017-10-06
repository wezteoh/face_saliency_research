from pylab import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import time
from scipy.misc import *
#from scipy.misc import imread
#from scipy.misc import imresize
import matplotlib.image as mpimg
from scipy.ndimage import filters
import urllib
from numpy import random
import _pickle as cPickle

import os
from scipy.io import loadmat
import tensorflow as tf

from Shan_alexnet_retrieve_conv4 import conv

t = int(time.time())
random.seed(t)

#ACT_NUM={'Fran Drescher':1, 'America Ferrera':2, 'Kristin Chenoweth':3, 'Alec Baldwin':4, 'Bill Hader':5, 'Steve Carell':0}

train_x = zeros((1, 227,227,3)).astype(float32)
train_y = zeros((1, 1000))
xdim = train_x.shape[1:]
ydim = train_y.shape[1]
net_data = load(open("bvlc_alexnet.npy", "rb"), encoding="latin1").item()

num_N = 6

def alex_nn(images, unit_num, labels,actnum,alpha = 0.001,epoch = 100):
    x = tf.placeholder(tf.float32, (None,) + xdim)


    #conv1
    #conv(11, 11, 96, 4, 4, padding='VALID', name='conv1')
    k_h = 11; k_w = 11; c_o = 96; s_h = 4; s_w = 4
    conv1W = tf.Variable(net_data["conv1"][0])
    conv1b = tf.Variable(net_data["conv1"][1])
    conv1_in = conv(x, conv1W, conv1b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=1)
    conv1 = tf.nn.relu(conv1_in)

    #lrn1
    #lrn(2, 2e-05, 0.75, name='norm1')
    radius = 2; alpha = 2e-05; beta = 0.75; bias = 1.0
    lrn1 = tf.nn.local_response_normalization(conv1,
                                                      depth_radius=radius,
                                                      alpha=alpha,
                                                      beta=beta,
                                                      bias=bias)

    #maxpool1
    #max_pool(3, 3, 2, 2, padding='VALID', name='pool1')
    k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'VALID'
    maxpool1 = tf.nn.max_pool(lrn1, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)


    #conv2
    #conv(5, 5, 256, 1, 1, group=2, name='conv2')
    k_h = 5; k_w = 5; c_o = 256; s_h = 1; s_w = 1; group = 2
    conv2W = tf.Variable(net_data["conv2"][0])
    conv2b = tf.Variable(net_data["conv2"][1])
    conv2_in = conv(maxpool1, conv2W, conv2b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
    conv2 = tf.nn.relu(conv2_in)


    #lrn2
    #lrn(2, 2e-05, 0.75, name='norm2')
    radius = 2; alpha = 2e-05; beta = 0.75; bias = 1.0
    lrn2 = tf.nn.local_response_normalization(conv2,
                                                      depth_radius=radius,
                                                      alpha=alpha,
                                                      beta=beta,
                                                      bias=bias)

    #maxpool2
    #max_pool(3, 3, 2, 2, padding='VALID', name='pool2')
    k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'VALID'
    maxpool2 = tf.nn.max_pool(lrn2, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)

    #conv3
    #conv(3, 3, 384, 1, 1, name='conv3')
    k_h = 3; k_w = 3; c_o = 384; s_h = 1; s_w = 1; group = 1
    conv3W = tf.Variable(net_data["conv3"][0])
    conv3b = tf.Variable(net_data["conv3"][1])
    conv3_in = conv(maxpool2, conv3W, conv3b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
    conv3 = tf.nn.relu(conv3_in)

    #conv4
    #conv(3, 3, 384, 1, 1, group=2, name='conv4')
    k_h = 3; k_w = 3; c_o = 384; s_h = 1; s_w = 1; group = 2
    conv4W = tf.Variable(net_data["conv4"][0])
    conv4b = tf.Variable(net_data["conv4"][1])
    conv4_in = conv(conv3, conv4W, conv4b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
    conv4 = tf.nn.relu(conv4_in)
    
    #conv5
    #conv(3, 3, 256, 1, 1, group=2, name='conv5')
    k_h = 3; k_w = 3; c_o = 256; s_h = 1; s_w = 1; group = 2
    conv5W = tf.Variable(net_data["conv5"][0])
    conv5b = tf.Variable(net_data["conv5"][1])
    conv5_in = conv(conv4, conv5W, conv5b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
    conv5 = tf.nn.relu(conv5_in)

    ##Change
    output_dim = num_N; nhid = 50; input_dim = 13*13*256 #CHANGE
    # define input x and target t
    fc1 = tf.reshape(conv5, [-1, int(prod(conv5.get_shape()[1:]))])
    t = tf.placeholder(tf.float32, [None, output_dim])

    # define weight matrix varaibles
    W0 = tf.Variable(tf.random_normal([input_dim, nhid], stddev=0.001))
    b0 = tf.Variable(tf.random_normal([nhid], stddev=0.001))

    W1 = tf.Variable(tf.random_normal([nhid, output_dim], stddev=0.001))
    b1 = tf.Variable(tf.random_normal([output_dim], stddev=0.001))
    f = open("relu9_25_conv5_50.pkl","rb")
    snapshot = cPickle.load(f)
    #snapshot = np.load('data/part10_80_60_5e-05_0.0_weights.npy').item()
    W0 = tf.Variable(snapshot["W0"])
    b0 = tf.Variable(snapshot["b0"])
    W1 = tf.Variable(snapshot["W1"])
    b1 = tf.Variable(snapshot["b1"])

    # define layers
    layer1 = tf.nn.relu(tf.matmul(fc1, W0)+b0)
    layer2 = tf.matmul(layer1, W1)+b1

    # define output
    y = tf.nn.softmax(layer2)

    # Guided Backprop
    dlayer1 = tf.gradients(layer2[0,actnum], layer1)[0]
    #most = tf.argmax(layer1)

    
    
    ## Conv5 neg
    dconv5_neg = tf.gradients(layer1[0,unit_num], conv5)[0]
    ZEROS = tf.zeros(tf.shape(dconv5_neg), tf.float32)
    dconv5_neg = tf.where(dconv5_neg > 0.0, ZEROS, dconv5_neg)
    dconv4_neg = tf.gradients(conv5, conv4, grad_ys=dconv5_neg)[0]
    
    ZEROS = tf.zeros(tf.shape(dconv4_neg), tf.float32)
    dconv4_neg = tf.where(dconv4_neg > 0.0, ZEROS, dconv4_neg)
    dconv3_neg = tf.gradients(conv4, conv3, grad_ys=dconv4_neg)[0]

    ZEROS = tf.zeros(tf.shape(dconv3_neg), tf.float32)
    dconv3_neg = tf.where(dconv3_neg > 0.0, ZEROS, dconv3_neg)
    dmaxpool2_neg = tf.gradients(conv3, maxpool2, grad_ys=dconv3_neg)[0]

    ZEROS = tf.zeros(tf.shape(dmaxpool2_neg), tf.float32)
    dmaxpool2_neg = tf.where(dmaxpool2_neg > 0.0, ZEROS, dmaxpool2_neg)
    dlrn2_neg = tf.gradients(maxpool2, lrn2, grad_ys=dmaxpool2_neg)[0]

    ZEROS = tf.zeros(tf.shape(dlrn2_neg), tf.float32)
    dlrn2_neg= tf.where(dlrn2_neg> 0.0, ZEROS, dlrn2_neg)
    dconv2_neg = tf.gradients(lrn2, conv2, grad_ys=dlrn2_neg)[0]

    ZEROS = tf.zeros(tf.shape(dconv2_neg), tf.float32)
    dconv2_neg= tf.where(dconv2_neg> 0.0, ZEROS, dconv2_neg)
    dmaxpool1_neg = tf.gradients(conv2, maxpool1, grad_ys=dconv2_neg)[0]

    ZEROS = tf.zeros(tf.shape(dmaxpool1_neg), tf.float32)
    dmaxpool1_neg= tf.where(dmaxpool1_neg> 0.0, ZEROS, dmaxpool1_neg)
    dlrn1_neg = tf.gradients(maxpool1, lrn1, grad_ys=dmaxpool1_neg)[0]

    ZEROS = tf.zeros(tf.shape(dlrn1_neg), tf.float32)
    dlrn1_neg= tf.where(dlrn1_neg> 0.0, ZEROS, dlrn1_neg)
    dconv1_neg = tf.gradients(lrn1, conv1, grad_ys=dlrn1_neg)[0]

    ZEROS = tf.zeros(tf.shape(dconv1_neg), tf.float32)    
    dconv1_neg = tf.where(dconv1_neg> 0.0, ZEROS, dconv1_neg)
    dx_neg = tf.gradients(conv1, x, grad_ys=dconv1_neg)[0]
    
    ZEROS = tf.zeros(tf.shape(dx_neg), tf.float32)
    dxt_neg = tf.where(dx_neg> 0.0, ZEROS, dx_neg)

    ##Continue
    init = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init)

    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(t,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    
    grad_x_neg = sess.run(dxt_neg, feed_dict = {x: images, t: labels})
    acc = sess.run(accuracy, feed_dict={x: images, t: labels})

    return acc, grad_x_neg
names = ['carell','drescher','ferrera','chenoweth','baldwin','hader']
#names =['drescher', 'ferrera', 'chenoweth', 'baldwin', 'hader', 'carell', 'butler', 'radcliffe', 'vartan', 'bracco', 'gilpin', 'harmon']
names =['drescher', 'ferrera', 'chenoweth', 'baldwin', 'hader', 'carell']
directs = ['test','validation','training']
def test(unit_num):
    test = {}
    validation = {}
    M = {}
    All = [test, validation, M]
    f = open("images_6_actors","rb")
    M = cPickle.load(f)
    
    for actor in range(6):
        batch_xs = zeros((0, 227,227,3))
        batch_y_s = zeros( (0, num_N))
        rand_int = random_integers(95)#(M[names[actor]]['training'].shape[0])
        batch_xs = vstack((batch_xs, ((array(M[names[actor]]['training'][rand_int].reshape((1,227,227,3))))  )))
        for k in range(6):
            batch_xs = vstack((batch_xs, ((array(M[names[actor]]['training'][rand_int].reshape((1,227,227,3))))  )))
            one_hot = zeros(num_N)
            one_hot[k] = 1
            batch_y_s = vstack((batch_y_s,   one_hot   ))
        
        images = batch_xs
        t = batch_y_s
        name = names[actor]
        if not os.path.exists(name+"/"):
            os.makedirs(name+"/")
        correctness = ['Incorrect','Correct']
        result = []
        grad_plots_neg = zeros((0, 227,227,3))
        
        for i in range(num_N):
            accuracy,grad_neg= alex_nn(images[i+1].reshape(1,227,227,3),unit_num, t[i].reshape(1,num_N),i,alpha = 0.0001, epoch = 100)
            accuracy = int(accuracy)
            grad_plots_neg = vstack((grad_plots_neg, grad_neg*50))
            result.append(correctness[accuracy])
            
        print('GOOD')
        fig, ax = plt.subplots(3,3, figsize=(60, 60))
        #fig = plt.figure(figsize=(16,6))
        #ax[0] = fig.add_subplot(121)
        ax[0, 0].imshow(images[0], label = names[actor]+str(rand_int), cmap = cm.gray)
        ax[0, 0].xaxis.set_visible(False)
        ax[0, 0].yaxis.set_visible(False)
        ax[0, 1].xaxis.set_visible(False)
        ax[0, 1].yaxis.set_visible(False)
        #ax1 = fig.add_subplot(122)
        ax[0, 1].imshow(grad_plots_neg[actor], label = "correct actor")
        #ax2 = fig.add_subplot(221)
        for temp_int in range(num_N):
            ax[(temp_int +2)//3, (temp_int +2)%3].imshow(grad_plots_neg[temp_int], label = names[temp_int])
            ax[(temp_int +2)//3, (temp_int +2)%3].xaxis.set_visible(False)
            ax[(temp_int +2)//3, (temp_int +2)%3].yaxis.set_visible(False)

        plt.tight_layout()
        plt.title(name + "_"+ str(rand_int)+"_backpropogation_" + result[actor])
        plt.savefig("Visualization_50/"+str(unit_num)+name+str(rand_int)+'_neg_plots_2.png')
        print('GOOD')
        
            #imsave("conv5_50_full_face_backprop/conv5_40_"+str(unit_num)+name+str(i)+'.png',grad[0]*100)

for i in range(0,10):
    unit_num = np.random.randint(50)
    test(unit_num)
