import numpy as np
import tensorflow as tf

from scipy.io import savemat
from scipy.io import loadmat

from vgg16_extractor import *

if __name__ == '__main__':
    image_set = loadmat('image_set.mat')
    del image_set['__header__']
    del image_set['__version__']
    del image_set['__globals__']
    
    inputs = tf.placeholder(tf.float32, shape = [None, 224, 224, 3])
    extractor = vgg16_extractor(inputs)
    outputs_flat = flatten(extractor.pool5)    
    
    with tf.Session() as sess:
        
        extractor.load_weights('vgg16_weights.npz', sess)
        
        batch_size = 25    
        for artist in image_set.keys():
            feature_maps[artist] = np.zeros([0, outputs_flat.shape[1]]) 
            for i in range(int(image_set[artist].shape[0]/batch_size)):
                batch = image_set[artist][i*batch_size:(i+1)*batch_size]
                feature_maps[artist] = np.vstack((feature_maps[artist], sess.run(outputs_flat, feed_dict={inputs: batch})))
                print(str((i+1)*100/(image_set[artist].shape[0]/batch_size)) + '% completed for ' + str(artist)) 
                
        savemat('test', feature_maps)