import numpy as np
import tensorflow as tf

from scipy.io import savemat
from scipy.io import loadmat

from vgg16_pool4_extractor import *

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