import numpy as np
import tensorflow as tf

from scipy.io import savemat
from scipy.io import loadmat

from fcl import * 

# load dataset
feature_maps = loadmat('vgg_pool5_outputs.mat')
del feature_maps['__header__']
del feature_maps['__version__']
del feature_maps['__globals__']

input_size = list(feature_maps.values())[0][0].shape[0]
no_classes = 10

# inputs
f_maps = tf.placeholder(tf.float32, shape = [None, input_size])
labels_1hot = tf.placeholder(tf.float32, shape=[None, no_classes])
train_layers = fcl(f_maps, no_classes, train_indicator=True)

# cost function
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = labels_1hot, logits=train_layers.fc3l))
lam = 0.003
decay_penalty =lam*tf.reduce_sum(tf.square(train_layers.fc1w))+lam*tf.reduce_sum(tf.square(train_layers.fc3w)) \
                #+ lam*tf.reduce_sum(tf.square(W_fc2))
cost = cross_entropy\
+ decay_penalty


# training
learning_rate = 0.001
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# accuracy test
correct_prediction = tf.equal(tf.argmax(train_layers.fc3l,1), tf.argmax(labels_1hot,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


# positions of ones in one-hot labels
labels = {}
for i, k in enumerate(feature_maps.keys()):
    labels[i] = k

# minibatch labels
train_batch_size = 50
csize_per_batch = train_batch_size//no_classes
train_classes_1hot = np.zeros([0,no_classes])
for i in range(no_classes):
    train_class_1hot = np.zeros([csize_per_batch, no_classes])
    train_class_1hot[:,i]=1 
    train_classes_1hot = np.vstack((train_classes_1hot, train_class_1hot))

# full training set
csize = 70
classes_1hot = np.zeros([0,no_classes])
for i in range(no_classes):
    class_1hot = np.zeros([csize, no_classes])
    class_1hot[:,i]=1 
    classes_1hot = np.vstack((classes_1hot, class_1hot))

trainset = np.zeros([0,input_size])
for i in range(no_classes):
    artist_data = feature_maps[labels[i]]
    train_class_set = artist_data[:csize]
    trainset = np.vstack((trainset, train_class_set))


# full test set
test_classes_1hot = np.zeros([0,no_classes])
for i in range(no_classes):
    test_class_1hot = np.zeros([feature_maps[labels[i]].shape[0]-csize, no_classes])
    test_class_1hot[:,i]=1 
    test_classes_1hot = np.vstack((test_classes_1hot, test_class_1hot))

testset = np.zeros([0,input_size])
for i in range(no_classes):
    artist_data = feature_maps[labels[i]]
    test_class_set = artist_data[csize:]
    testset = np.vstack((testset, test_class_set))


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    iter = 300
    for j in range(iter):
        
        # minibatch training
        fullbatch = np.zeros([0,input_size])
        for i in range(no_classes):
            artist_data = feature_maps[labels[i]]
            class_batch = artist_data[np.random.choice(csize, csize_per_batch)]
            fullbatch = np.vstack((fullbatch, class_batch))
            
        train_step.run(feed_dict={f_maps:fullbatch, labels_1hot:train_classes_1hot})
    
        
        #assess
        if (j+1)%10 == 0:
            print('iteration'+str(j+1))
            print(sess.run(accuracy, feed_dict={f_maps:trainset, labels_1hot:classes_1hot}))
            print(sess.run(accuracy, feed_dict={f_maps:testset, labels_1hot:test_classes_1hot}))
            print(sess.run(cross_entropy, feed_dict={f_maps:testset, labels_1hot:test_classes_1hot}))


