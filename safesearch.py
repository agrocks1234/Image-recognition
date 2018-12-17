import math
from collections import Counter

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from PIL import Image
import pickle
from tqdm import tqdm
import cv2
    
import glob, os
%matplotlib inline
name = []

def image_array(filename,size):
    array = cv2.imread(filename)
    if array is not None:
        array = cv2.resize(array, (size, size)) 
    else:
        array = np.resize(array, (size,size,3))
        
    return array
    
size = 224
os.chdir('/home/amit/Documents/Cnn model/train_set/non_nude')

data_x = []
non_nude = []
for file in glob.glob("*.jpg"):
    img = image_array(file , size)
    data_x.append(img)
    non_nude.append(img)
    name.append(file)
    
os.chdir('..')
os.chdir('nude')
nude = []
for file in glob.glob("*.jpg"):
    img = image_array(file,size)
    data_x.append(img)
    nude.append(img)
    name.append(file)

data_x = np.asarray(data_x)
non_nude = np.asarray(non_nude)
nude = np.asarray(nude)
label1 = []
for i in range(2861):
    label1.append('nude')
    
label = []
for i in range(2130):
    label.append('non_nude')
    
data_y = np.concatenate((np.asarray(label) , np.asarray(label1)), axis = 0)
label = np.resize(data_y , (2130,1))
data_y.shape
def one_hot_encoding(array):
    df = pd.DataFrame(array, index = None)
    data = pd.get_dummies(df)
    data = np.asarray(data)
    return data
    
import sklearn
from sklearn.model_selection import cross_validate
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(data_x, data_y , test_size = 0.3 , random_state = 100)
train_label = y_train
test_label = y_test
y_train = one_hot_encoding(y_train)
y_test = one_hot_encoding(y_test)
x_train = np.array(x_train , dtype = 'float')
x_test = np.array(x_test , dtype = 'float')
os.chdir('..')
os.getcwd()

def plot_images(imgs, labels, rows=3):
    # Set figure to 13 inches x 8 inches
    figure = plt.figure(figsize=(16, 8))

    cols = len(imgs) // rows + 1

    for i in range(len(imgs)):
        subplot = figure.add_subplot(rows, cols, i + 1)
        subplot.axis('Off')
        if labels is not None:
            subplot.set_title(labels[i], fontsize=16)
        plt.imshow(imgs[i] , cmap = 'gray')

from keras.models import Sequential
from scipy.misc import imread
get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
import numpy as np
import keras
from keras.layers import Dense
import pandas as pd

from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input
import numpy as np
from keras.applications.resnet50 import decode_predictions

train_X = preprocess_input(x_train)
test_X = preprocess_input(x_test)
model = ResNet50(weights='imagenet', include_top = False)
features_train=model.predict(train_X)
features_test=model.predict(test_X)
from keras.layers import Dense, Activation
train = features_train.reshape(features_train.shape[0] , -1)
test = features_test.reshape(features_test.shape[0] , -1)
training_epochs = 20
n_dim = train.shape[1]
n_class = 2

from keras.activations import softmax
def softMaxAxis1(x):
    return softmax(x , axis=1)

import types
import tempfile
import keras.models

def make_keras_picklable():
    def __getstate__(self):
        model_str = ""
        with tempfile.NamedTemporaryFile(suffix='.hdf5', delete=True) as fd:
            keras.models.save_model(self, fd.name, overwrite=True)
            model_str = fd.read()
        d = { 'model_str': model_str }
        return d

    def __setstate__(self, state):
        with tempfile.NamedTemporaryFile(suffix='.hdf5', delete=True) as fd:
            fd.write(state['model_str'])
            fd.flush()
            model = keras.models.load_model(fd.name)
        self.__dict__ = model.__dict__


    cls = keras.models.Model
    cls.__getstate__ = __getstate__
    cls.__setstate__ = __setstate__
    
make_keras_picklable()
model=Sequential()

model.add(Dense(100, input_dim= n_dim, activation='tanh',kernel_initializer='uniform'))
keras.layers.core.Dropout(0.3, noise_shape=None, seed=None)

model.add(Dense(50,input_dim=1000,activation='sigmoid'))
keras.layers.core.Dropout(0.4, noise_shape=None, seed=None)

model.add(Dense(15,input_dim=500,activation='sigmoid'))
keras.layers.core.Dropout(0.2, noise_shape=None, seed=None)

model.add(Dense(units=n_class))
model.add(Activation('softmax'))



model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])

model.fit(train, y_train, epochs=50,validation_data=(test,y_test))
os.chdir('..')
filename = 'RESNET50.h5'
model.save(filename)    

