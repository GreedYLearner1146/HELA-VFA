

import PIL
import numpy as np
import torch
import torchvision

######## Adding Gaussian Noise as part of our data augmentation #####################

noisyI = []       # Training datasets.

# Mean = 0, std = 0.005.  # You can change the value of std to 0.1, 0.15, 0.20, etc.

for f1 in range (len(new_X_train)):  # new_X_train is from Data_Loading.py
  img = new_X_train[f1]
  mean = 0.0   # some constant (Gaussian noise mean)
  std = 0.005   # some constant (standard deviation) (Gaussian noise std)
  noisy_imgI = img + np.random.normal(mean, std, img.shape)
  noisy_img_clippedI = np.clip(noisy_imgI, 0, 255)  # we might get out of bounds due to noise
  noisy_img_clippedI  = np.asarray(noisy_img_clippedI) # REMEMBER TO ADD CONVERT TO ASARRAY FIRST BEFORE APPENDING!!!!!!
  noisyI.append(noisy_img_clippedI)


noisyItest = []       # Testing datasets.

# Mean = 0, std = 0.005.

for f1t in range (len(new_X_test)):    # new_X_test is from Data_Loading.py
  imgt = new_X_test[f1t]
  mean = 0.0   # some constant
  std = 0.10   # some constant (standard deviation)
  noisy_imgIt = imgt + np.random.normal(mean, std, img.shape)
  noisy_img_clippedIt = np.clip(noisy_imgIt, 0, 255)  # we might get out of bounds due to noise
  noisy_img_clippedIt  = np.asarray(noisy_img_clippedIt) # REMEMBER TO ADD CONVERT TO ASARRAY FIRST BEFORE APPENDING!!!!!!
  noisyItest.append(noisy_img_clippedIt)


################# Remaining data augmentation for training (Top) and testing (Bottom) #################

# The data augmentation selected for the train datasets are as below.

data_transform = transforms.Compose(

     [
            transforms.ToTensor(),
            transforms.Resize((84,84)),
            transforms.CenterCrop((84,84)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
       ])

############################ No augmentation for test dataset ############################
data_transform_test = transforms.Compose([transforms.ToTensor(),
                                          transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

