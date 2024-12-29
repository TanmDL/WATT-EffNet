#################### Import relevant library packages ###############

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import keras
import tensorflow as tf
import tensorflow.keras.backend as K
import os
import cv2
import numpy as np
import glob
from os import listdir
from numpy import asarray
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
import re
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Conv2D, ReLU, Add, MaxPool2D, UpSampling2D, BatchNormalization, concatenate, Subtract
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, ZeroPadding2D, Add, Activation, Conv2DTranspose
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras import initializers
from tensorflow.python.keras.layers import Layer, InputSpec
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense, Lambda
from tensorflow.keras import backend as K
from tensorflow.keras.activations import relu
from PIL import Image
import re

#### Download the AIDER subset dataset from https://zenodo.org/records/3888300 and placed them in your directory of choice ############

def natural_sort_key(s, _nsre=re.compile('([0-9]+)')):
   return[int(text) if text.isdigit() else text.lower() for text in _nsre.split(s   )]


def load_images(path, size = (224,224)):
    data_list = list()# enumerate filenames in directory, assume all are images
    for filename in sorted(os.listdir(path),key=natural_sort_key):
      pixels = load_img(path + filename, target_size = size)# Convert to numpy array.
      pixels = img_to_array(pixels).astype('float32')
      pixels = cv2.resize(pixels,(224,224))# Need to resize images first, otherwise RAM will run out of space.
      pixels = pixels/255
      #pixels = cv2.threshold(pixels, 128, 128, cv2.THRESH_BINARY)
      data_list.append(pixels)
    return asarray(data_list)


path_building = 'your building image directory'
path_fire = 'your fire image directory'
path_flood = 'your flood image directory'
path_traffic = 'your traffic incident image directory'
path_normal = 'your normal image directory'

data_train_building = load_images(path_building)
data_train_fire= load_images(path_fire)
data_train_flood = load_images(path_flood)
data_train_traffic = load_images(path_traffic)
data_train_normal= load_images(path_normal)

####################### Prepare array for storing images and labels.###################################

imgarraybuilding =  []
labelarraybuilding = []

imgarrayfire =  []
labelarrayfire = []

imgarrayflood =  []
labelarrayflood = []

imgarraytraffic =  []
labelarraytraffic = []

imgarraynormal =  []
labelarraynormal = []

for a in data_train_building:
    imgarraybuilding.append(a)
    labelarraybuilding.append(0)

for  b  in  data_train_fire :
    imgarrayfire.append (b)
    labelarrayfire.append (1)

for c in data_train_flood:
    imgarrayflood.append(c)
    labelarrayflood.append(2)

for d in data_train_traffic:
   imgarraytraffic.append(d)
   labelarraytraffic.append(3)

for e in data_train_normal:
    imgarraynormal.append (e)
    labelarraynormal.append (4)

##################################### Preparing for train-test split ######################################################

from sklearn.model_selection import train_test_split

###################### Building #########################

(X_trainb, X_testb, y_trainb, y_testb) = train_test_split(imgarraybuilding,labelarraybuilding,test_size=0.2,random_state=0)
#print(np.shape(X_trainb), np.shape(X_testb))
(X_trainb, X_valb, y_trainb, y_valb) = train_test_split(X_trainb, y_trainb,test_size=0.1, random_state=0)
#print(np.shape(X_trainb), np.shape(X_valb))

###################### Fire #########################

print()
(X_trainf, X_testf, y_trainf, y_testf) = train_test_split(imgarrayfire,labelarrayfire,test_size=0.4,random_state=0)
#print(np.shape(X_trainf), np.shape(X_testf))
(X_trainf, X_valf, y_trainf, y_valf) = train_test_split(X_trainf, y_trainf,test_size=0.2, random_state=0)
#print(np.shape(X_trainf), np.shape(X_valf))


###################### Flood #########################

print()
(X_trainF, X_testF, y_trainF, y_testF) = train_test_split(imgarrayflood,labelarrayflood,test_size=0.4,random_state=0)
#print(np.shape(X_trainF), np.shape(X_testF))
(X_trainF, X_valF, y_trainF, y_valF) = train_test_split(X_trainF, y_trainF,test_size=0.2, random_state=0)
#print(np.shape(X_trainF), np.shape(X_valF))

###################### Traffic #########################

print()
(X_traint, X_testt, y_traint, y_testt) = train_test_split(imgarraytraffic,labelarraytraffic,test_size=0.4,random_state=0)
#print(np.shape(X_traint), np.shape(X_testt))
(X_traint, X_valt, y_traint, y_valt) = train_test_split(X_traint, y_traint,test_size=0.2, random_state=0)
#print(np.shape(X_traint), np.shape(X_valt))

###################### Normal #########################

print()
(X_trainn, X_testn, y_trainn, y_testn) = train_test_split(imgarraynormal,labelarraynormal,test_size=0.4,random_state=0)
#print(np.shape(X_trainn), np.shape(X_testn))
(X_trainn, X_valn, y_trainn, y_valn) = train_test_split(X_trainn, y_trainn,test_size=0.2, random_state=0)
#print(np.shape(X_trainn), np.shape(X_valn))

####################### Prepare array for storing images and labels.(TRAINING) #######################################

imgarraybuilding_train =  []
labelarraybuilding_train = []

imgarrayfire_train =  []
labelarrayfire_train = []

imgarrayflood_train =  []
labelarrayflood_train = []

imgarraytraffic_train =  []
labelarraytraffic_train = []

imgarraynormal_train =  []
labelarraynormal_train = []

for a in X_trainb:
    imgarraybuilding_train.append(a)
    labelarraybuilding_train.append(0)

for b in X_trainf:
    imgarrayfire_train.append(b)
    labelarrayfire_train.append(1)

for c in X_trainF:
    imgarrayflood_train.append(c)
    labelarrayflood_train.append(2)

for d in X_traint:
   imgarraytraffic_train.append(d)
   labelarraytraffic_train.append(3)

for e in X_trainn:
    imgarraynormal_train.append(e)
    labelarraynormal_train.append(4)

####################### Prepare array for storing images and labels.(VALID) #######################################

# Prepare array for storing images and labels.(Validation)

imgarraybuilding_valid = []
labelarraybuilding_valid = []

imgarrayfire_valid =   []
labelarrayfire_valid =   []

imgarrayflood_valid = []
labelarrayflood_valid = []

imgarraytraffic_valid = []
labelarraytraffic_valid = []

imgarraynormal_valid =   []
labelarraynormal_valid =   []

for a in X_valb:
    imgarraybuilding_valid.append(a)
    labelarraybuilding_valid.append(0)

for b in X_valf:
    imgarrayfire_valid.append(b)
    labelarrayfire_valid.append(1)

for c in X_valF:
    imgarrayflood_valid.append(c)
    labelarrayflood_valid.append(2)

for d in X_valt:
   imgarraytraffic_valid.append(d)
   labelarraytraffic_valid.append(3)

for e in X_valn:
    imgarraynormal_valid.append(e)
    labelarraynormal_valid.append(4)

####################### Prepare array for storing images and labels.(TEST) #######################################

# Prepare array for storing images and labels.(test)

imgarraybuilding_test = []
labelarraybuilding_test = []

imgarrayfire_test =   []
labelarrayfire_test =   []

imgarrayflood_test = []
labelarrayflood_test= []

imgarraytraffic_test = []
labelarraytraffic_test = []

imgarraynormal_test =   []
labelarraynormal_test=   []

for a in X_testb:
    imgarraybuilding_test.append(a)
    labelarraybuilding_test.append(0)

for b in X_testf:
    imgarrayfire_test.append(b)
    labelarrayfire_test.append(1)

for c in X_testF:
    imgarrayflood_test.append(c)
    labelarrayflood_test.append(2)

for d in X_testt:
   imgarraytraffic_test.append(d)
   labelarraytraffic_test.append(3)

for e in X_testn:
    imgarraynormal_test.append(e)
    labelarraynormal_test.append(4)

############### Combine respective class arrays into one whole train,valid,test arrays ###################

trainarray = []
trainlabel = []

validarray = []
validlabel = []

testarray =  []
testlabel =  []

for  A1  in  imgarraybuilding_train :
    trainarray.append ( A1 )

for A2 in imgarraybuilding_valid:
    validarray.append(A2)

for A3 in imgarraybuilding_test:
    testarray.append(A3)

for A4 in labelarraybuilding_train:
    trainlabel.append(A4)

for A5 in labelarraybuilding_valid:
    validlabel.append(A5)

for A6 in labelarraybuilding_test:
    testlabel.append(A6)

###############################################

for B1 in imgarrayfire_train:
    trainarray.append(B1)

for B2 in imgarrayfire_valid:
    validarray.append(B2)

for B3 in imgarrayfire_test:
    testarray.append(B3)

for B4 in labelarrayfire_train:
    trainlabel.append(B4)

for B5 in labelarrayfire_valid:
    validlabel.append(B5)

for B6 in labelarrayfire_test:
    testlabel.append(B6)

###############################################

for  C1  in  imgarrayflood_train :
    trainarray.append ( C1 )

for C2 in imgarrayflood_valid:
    validarray.append(C2)

for C3 in imgarrayflood_test:
    testarray.append(C3)

for C4 in labelarrayflood_train:
    trainlabel.append(C4)

for C5 in labelarrayflood_valid:
    validlabel.append(C5)

for C6 in labelarrayflood_test:
    testlabel.append(C6)

###############################################

for  D1  in  imgarraytraffic_train :
    trainarray.append ( D1 )

for D2 in imgarraytraffic_valid:
    validarray.append(D2)

for D3 in imgarraytraffic_test:
    testarray.append(D3)

for D4 in labelarraytraffic_train:
    trainlabel.append(D4)

for D5 in labelarraytraffic_valid:
    validlabel.append(D5)

for D6 in labelarraytraffic_test:
    testlabel.append(D6)

###############################################

for  E1  in  imgarraynormal_train :
    trainarray.append ( E1 )

for E2 in imgarraynormal_valid:
    validarray.append(E2)

for E3 in imgarraynormal_test:
    testarray.append(E3)

for E4 in labelarraynormal_train:
    trainlabel.append(E4)

for E5 in labelarraynormal_valid:
    validlabel.append(E5)

for E6 in labelarraynormal_test:
    testlabel.append(E6)

################ Array tuple for train, valid and test set ########################

training_AIDER =  []
valid_AIDER =  []
test_AIDER = []

for a, b in zip(trainarray,trainlabel):
    training_AIDER.append([a,b])

for c, d in zip(validarray,validlabel):
    valid_AIDER.append([c,d])

for e, f in zip(testarray,testlabel):
    test_AIDER.append([e,f])

####### Shuffle img array and label (Important to randomize order) ##########

from sklearn.utils import shuffle

finaltraining = shuffle (training_AIDER)
finalval = shuffle (valid_AIDER)
finaltest = shuffle(test_AIDER)

new_X = [x[0] for x in finaltraining]
new_y = [x[1] for x in finaltraining]

new_X_valid = [w[0] for w in finalval]
new_y_valid  = [w[1] for w in finalval]

new_X_test = [v[0] for v in finaltest]
new_y_test  = [v[1] for v in finaltest]
