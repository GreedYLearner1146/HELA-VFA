import scipy
from scipy import spatial
from scipy.spatial.distance import cdist
import numpy as np
from tqdm import tqdm
from math import floor
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch as F
from torch.utils.data import Dataset
from torch import optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import resnet18
from torch.utils.data import DataLoader
import torch as T
from torch import nn, optim
from tqdm import tqdm

import matplotlib.pyplot as plt
import re
import os
import cv2
import PIL
import numpy as np
import torchvision
import glob
from random import shuffle
from PIL import Image
from os import listdir
from numpy import asarray
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from imblearn.under_sampling import RandomUnderSampler
import random
from tensorflow.keras.losses import MSE
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from numpy import array
from numpy import argmax
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

!pip install easyfsl  # Install easyfsl. IMPORTANT AS WE WILL BE USING THIS USEFUL FEW-SHOT LIBRARY PACKAGE.
from easyfsl.samplers import TaskSampler
from easyfsl.utils import plot_images, sliding_average

###############################################################################################
# DATA PREPARATION PART.
# We use miniImageNet as an example as the subsequent codes for the
# respective outputs can be illustrated in a smooth manner. The same steps goes for the other datasets.

def natural_sort_key(s, _nsre=re.compile('([0-9]+)')):
   return[int(text) if text.isdigit() else text.lower() for text in _nsre.split(s)]

path = 'YOUR PATH FOR THE MINIIMAGENET DATA HERE'

files_list_miniImageNet = []
for filename in sorted(os.listdir(path),key=natural_sort_key):
    files_list_miniImageNet.append(filename)

# Shuffle the list for randomization of the dataset.

shuffled = random.sample(files_list_miniImageNet,len(files_list_miniImageNet))

# For training and validation data splitting (64 meta-training, 16 meta-validation).

def get_training_and_valid_sets(file_list):
    split = 0.64
    split_index = floor(len(file_list) * split)
    # Training.
    training = file_list[:split_index]
    # Valid.
    validation = file_list[split_index:]
    return training, validation

trainlist_final,_ = get_training_and_valid_sets(shuffled)
_,vallist = get_training_and_valid_sets(shuffled)

# For validation and test data splitting (16 meta-valid, 20 meta-test).

def get_validation_and_testing_sets(file_list):
    split = 0.5 
    split_index = floor(len(file_list) * split)
    # Final valid.
    valid = file_list[:split_index]
    # Final test.
    test = file_list[split_index:]
    return valid, test

vallist_final,_ = get_validation_and_testing_sets(vallist)
_,testlist_final = get_validation_and_testing_sets(vallist)

######################## Load Meta-train Images ###############################

def natural_sort_key(s, _nsre=re.compile('([0-9]+)')):
   return[int(text) if text.isdigit() else text.lower() for text in _nsre.split(s   )]

def load_images(path, size = (84,84)):
    data_list = list()# enumerate filenames in directory, assume all are images
    for filename in sorted(os.listdir(path),key=natural_sort_key):
      pixels = load_img(path + filename, target_size = size)# Convert to numpy array.
      pixels = img_to_array(pixels).astype('float32')
      pixels = cv2.resize(pixels,(84,84)) 
      pixels = pixels/255  # Size of 84 x 84 x 3.
      data_list.append(pixels)
    return asarray(data_list)

train_img = []
for train in trainlist_final:
   data_train_img = load_images(path + '/' + train + '/')   # Training path as: 'YOUR PATH FOR THE MINIIMAGENET DATA HERE/train/' 
   train_img.append(data_train_img)


############# Train, valid, test images in array list format ##################

TRAIN = []

for i in range (len(train_img)):
   TRAIN.append(train_img[i][0:600])       # Each class has 600 images.

val_img = []       # Load valid images.

for val in vallist_final:
   data_val_img = load_images(path + '/' + val + '/')        # Valid path as: 'YOUR PATH FOR THE MINIIMAGENET DATA HERE/val/' 
   val_img.append(data_val_img)

test_img = []            # Load test images.

for test in testlist_final:        
   data_test_img = load_images(path + '/' + test + '/')     # Test path as: 'YOUR PATH FOR THE MINIIMAGENET DATA HERE/test/' 
   test_img.append(data_test_img)

############# Train, valid, test images + labels in array list format ##################

train_img_final = []
val_img_final = []
test_img_final = []

train_label_final = []
val_label_final = []
test_label_final = []


for a in range (len(TRAIN)):
   for b in range (600):
      train_img_final.append(train_img[a][b])
      train_label_final.append(a)  # 64 classes.

for c in range (len(val_img)):
   for d in range (600):
      val_img_final.append(val_img[c][d])
      val_label_final.append(c+64)  # 16 classes.

for e in range (len(test_img)):
  for f in range (600):
      test_img_final.append(test_img[e][f])
      test_label_final.append(e+80)  # Remaining 20 classes.

############# Reassemble in tuple format. ##################

train_array = []
val_array = []
test_array = []

for a,b in zip(train_img_final,train_label_final):
  train_array.append((a,b))

for c,d in zip(val_img_final,val_label_final):
  val_array.append((c,d))

for e,f in zip(test_img_final,test_label_final):
  test_array.append((e,f))

new_X_train = [x[0] for x in train_array]
new_y_train = [x[1] for x in train_array]

new_X_val = [x[0] for x in val_array]
new_y_val = [x[1] for x in val_array]

new_X_test = [x[0] for x in test_array]
new_y_test = [x[1] for x in test_array]

################### Check shape of each image and label array ########################

print(np.shape(new_X_train), np.shape(new_y_train))
print(np.shape(new_X_val), np.shape(new_y_val))
print(np.shape(new_X_test), np.shape(new_y_test))


############ USE the Gaussian Noise code component of 'Data_Augmentation.py ################

noisyI = []       # For train images. 

# Mean = 0, std = 0.005.

for f1 in range (len(new_X_train)):
  img = new_X_train[f1]
  mean = 0.0   # some constant
  std = 0.005   # some constant (standard deviation)
  noisy_imgI = img + np.random.normal(mean, std, img.shape)
  noisy_img_clippedI = np.clip(noisy_imgI, 0, 255)  # we might get out of bounds due to noise
  noisy_img_clippedI  = np.asarray(noisy_img_clippedI) # REMEMBER TO ADD CONVERT TO ASARRAY FIRST BEFORE APPENDING!!!!!!
  noisyI.append(noisy_img_clippedI)


noisy1test = []     # For test images. 

for f1t in range (len(new_X_test)):
  imgt = new_X_test[f1t]
  mean = 0.0   # some constant
  std = 0.005   # some constant (standard deviation)
  noisy_img1t = imgt + np.random.normal(mean, std, img.shape)
  noisy_img_clipped1t = np.clip(noisy_img1t, 0, 255)  # we might get out of bounds due to noise
  noisy_img_clipped1t  = np.asarray(noisy_img_clipped1t) # REMEMBER TO ADD CONVERT TO ASARRAY FIRST BEFORE APPENDING!!!!!!
  noisy1test.append(noisy_img_clipped1t)

######################### Arrays after inclusion of the noisy images ########################################

train_img_FINAL = []
val_img_FINAL = []
test_img_FINAL = []

train_label_FINAL = []
val_label_FINAL = []
test_label_FINAL = []


for a in range (len(TRAIN)):
   for b in range (600):
      train_img_FINAL.append(train_img[a][b])
      train_label_FINAL.append(a)  # 60 classes.

for AI in range (len(noisyI)): # Add the gaussian noise 0.005.
      train_img_FINAL.append(noisyI[AI])
      train_label_FINAL.append(a)  # 60 classes.


for e in range (len(test_img)):
  for f in range (600):
      test_img_FINAL.append(test_img[e][f])
      test_label_FINAL.append(e+80)  # Remaining 20 classes.

for E in range (len(noisy1test)):  # Add the gaussian noise 0.005.
      test_img_FINAL.append(noisy1test[E])
      test_label_FINAL.append(e+80)   # Remaining 20 classes.


############# Reassemble in tuple format. ##################

train_array = []
val_array = []
test_array = []

for a,b in zip(train_img_FINAL,train_label_FINAL):
  train_array.append((a,b))

#for c,d in zip(val_img_final,val_label_final):
  #val_array.append((c,d))

for e,f in zip(test_img_FINAL,test_label_FINAL):
  test_array.append((e,f))

################## shuffle #############################
from sklearn.utils import shuffle
train_array = shuffle(train_array)
test_array = shuffle(test_array)

new_X_train = [x[0] for x in train_array]
new_y_train = [x[1] for x in train_array]

new_X_test = [x[0] for x in test_array]
new_y_test = [x[1] for x in test_array]

################### Check new shape after adding gaussian noise ###########################

print(np.shape(new_X_train))
print(np.shape(new_y_train))

print(np.shape(new_X_test))
print(np.shape(new_y_test))
