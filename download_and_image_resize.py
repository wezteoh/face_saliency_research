"""
1. Download, crop and resize artists' images obtained from FaceScrub dataset 
2. Save the processed images in numpy array format
""" 

import os
from pylab import *
import numpy as np
import time
from scipy.misc import imread
from scipy.misc import imresize
import urllib

from scipy.io import savemat
from scipy.io import loadmat
from hashlib import sha256


def timeout(func, args=(), kwargs={}, timeout_duration=1, default=None):
    '''From:
    http://code.activestate.com/recipes/473878-timeout-function-using-threading/'''
    import threading
    class InterruptableThread(threading.Thread):
        def __init__(self):
            threading.Thread.__init__(self)
            self.result = None

        def run(self):
            try:
                self.result = func(*args, **kwargs)
            except:
                self.result = default

    it = InterruptableThread()
    it.start()
    it.join(timeout_duration)
    if it.isAlive():
        return False
    else:
        return it.result


testfile = urllib.request.URLopener() 
image_set = {}
if not os.path.exists("images"):
    os.makedirs("images")
    

# 100 images are obtained for each of the artists here as initial dataset
actors = ['Daniel Radcliffe', 'Gerard Butler', 'Michael Vartan', 'Matt Damon', 'Nicolas Cage']
actresses = ['Cheryl Hines', 'Selena Gomez', 'Angie Harmon', 'Anne Hathaway', 'Jennifer Aniston']


for a in actors:
# change to actresses
    name = a.split()[1].lower()
    i = 1
    act_imgset = []
    with open("facescrub_actors.txt") as f:
    #change to actresses
        while i <= 100:
            line = f.readline()
            if a in line:
                filename = name + '_' + str(i) + '.' + line.split()[4].split('.')[-1]
                timeout(testfile.retrieve, (line.split()[4], "images/" + filename), {}, 30)


                print(filename)
                try:
                    with open('images/'+ filename, 'rb') as g:
                        sha256sum = sha256(g.read()).hexdigest()
                        
                    img = imread("images/" + filename)
                    if sha256sum == line.split()[6] and len(img.shape) == 3:
                        img = imread("images/" + filename, mode='RGB')
                        bound = line.split()[5].split(',')
                        img = img[int(bound[1]):int(bound[3]), int(bound[0]):int(bound[2])]
                        img_resized = imresize(img, [224,224,3])
                        act_imgset.append(img_resized)
                        i += 1
                    else:
                        os.remove('images/'+filename)
                    
                except:
                    if os.path.isfile('images/'+filename):
                        os.remove('images/'+filename)
                    
    image_set[a] = np.asarray(act_imgset)


# Repeat with actresses before this step

savemat("image_set", image_set)



