# Import relevant libraries

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

############ We use the easyfsl library mainly for the work ###############

!pip install easyfsl  # Install easyfsl.
from easyfsl.samplers import TaskSampler
from easyfsl.utils import plot_images, sliding_average

###############################################################################################
# DATA PREPARATION PART.
# We use miniImageNet as an example as the subsequent codes for the
# respective outputs can be illustrated in a smooth manner. 
# The same steps goes for the other datasets.

def natural_sort_key(s, _nsre=re.compile('([0-9]+)')):
   return[int(text) if text.isdigit() else text.lower() for text in _nsre.split(s   )]

path = 'path to miniimagenet files here'

files_list_miniImageNet = []
for filename in sorted(os.listdir(path),key=natural_sort_key):
    files_list_miniImageNet.append(filename)

# Shuffle the list for randomization of the dataset.

random.seed(500)  # Set to __ first. Change to __ after training as part of the fine-tuning steps.
shuffled = random.sample(files_list_miniImageNet,len(files_list_miniImageNet))

def get_training_and_test_sets(file_list):
    split = 0.80                                    # This simulation fused the training and validation class together.
    split_index = floor(len(file_list) * split)     # Testing classes remains at 20.
    # Training.
    training = file_list[:split_index]
    # Valid.
    validation = file_list[split_index:]
    return training, validation

trainlist_final, testlist_final = get_training_and_test_sets(shuffled)

########################### Load Images (size 84 x 84) ##############################

def load_images(path, size = (84,84)):
    data_list = list()# enumerate filenames in directory, assume all are images
    for filename in sorted(os.listdir(path),key=natural_sort_key):
      pixels = load_img(path + filename, target_size = size) 
      pixels = img_to_array(pixels).astype('float32')
      pixels = cv2.resize(pixels,(84,84)) 
      pixels = pixels/255
      data_list.append(pixels)
    return asarray(data_list)

############# Train, test images in array list format ##################

train_img = []
for train in trainlist_final:
   data_train_img = load_images(path + '/' + train + '/')
   train_img.append(data_train_img)

TRAIN = []

for i in range (len(train_img)):
   TRAIN.append(train_img[i][0:600])   # Each train, valid and test class has 600 images. 

test_img = []

for test in testlist_final:
   data_test_img = load_images(path + '/' + test + '/')
   test_img.append(data_test_img)

############# Train, test images + labels in array list format ##################

train_img_final = []
test_img_final = []

train_label_final = []
test_label_final = []


for a in range (len(TRAIN)):
   for b in range (600):
      train_img_final.append(train_img[a][b])
      train_label_final.append(a)

for e in range (len(test_img)):
  for f in range (600):
      test_img_final.append(test_img[e][f])
      test_label_final.append(e+80)

############# Reassemble in tuple format. ##################

train_array = []
test_array = []

for a,b in zip(train_img_final,train_label_final):
  train_array.append((a,b))

for e,f in zip(test_img_final,test_label_final):
  test_array.append((e,f))

################## shuffle #############################

from sklearn.utils import shuffle
train_array = shuffle(train_array)
test_array = shuffle(test_array)

new_X_train = [x[0] for x in train_array]
new_y_train = [x[1] for x in train_array]

new_X_test = [x[0] for x in test_array]
new_y_test = [x[1] for x in test_array]

################### Check shape of each image and label array ########################

print(np.shape(new_X_train), np.shape(new_y_train))
print(np.shape(new_X_test), np.shape(new_y_test))
