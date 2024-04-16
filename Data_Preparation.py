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
# respective outputs can be illustrated in a smooth manner. The same steps goes for the other datasets.

def natural_sort_key(s, _nsre=re.compile('([0-9]+)')):
   return[int(text) if text.isdigit() else text.lower() for text in _nsre.split(s   )]

path = 'path to miniImageNet here'

files_list_miniImageNet = []
for filename in sorted(os.listdir(path),key=natural_sort_key):
    files_list_miniImageNet.append(filename)

# Shuffle the list for randomization of the dataset.
shuffled = random.sample(files_list_miniImageNet,len(files_list_miniImageNet))

# For training and validation data splitting.

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


########################### Load Images (size 84 x 84) ##############################
def load_images(path, size = (84,84)):
    data_list = list()# enumerate filenames in directory, assume all are images
    for filename in sorted(os.listdir(path),key=natural_sort_key):
      pixels = load_img(path + filename, target_size = size)# Convert to numpy array.
      pixels = img_to_array(pixels).astype('float32')
      pixels = cv2.resize(pixels,(84,84))
      pixels = pixels/255
      data_list.append(pixels)
    return asarray(data_list)

train_img = []
for train in trainlist_final:
   data_train_img = load_images(path + '/' + train + '/')
   train_img.append(data_train_img)

############# Train, valid images in array list format ##################

TRAIN = []

for i in range (len(train_img)):
   TRAIN.append(train_img[i][0:600])

val_img = []

for val in vallist:
   data_val_img = load_images(path + '/' + val + '/')
   val_img.append(data_val_img)


############# Train, valid images + labels in array list format ##################

train_img_final = []
val_img_final = []

train_label_final = []
val_label_final = []

for a in range (len(TRAIN)):
   for b in range (600):  # Each class has 600 images.
      train_img_final.append(train_img[a][b])
      train_label_final.append(a)

for c in range (len(val_img)):
   for d in range (600):   # Each class has 600 images.
      val_img_final.append(val_img[c][d])
      val_label_final.append(c+60)


############# Reassemble in tuple format. ##################

train_array = []
val_array = []

for a,b in zip(train_img_final,train_label_final):
  train_array.append((a,b))

for c,d in zip(val_img_final,val_label_final):
  val_array.append((c,d))

################## shuffle #############################

from sklearn.utils import shuffle
train_array = shuffle(train_array)
val_array = shuffle(val_array)

new_X_train = [x[0] for x in train_array]
new_y_train = [x[1] for x in train_array]

new_X_val = [x[0] for x in val_array]
new_y_val = [x[1] for x in val_array]
