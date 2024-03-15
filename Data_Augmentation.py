import PIL
import numpy as np
import torch
import torchvision


################# Remaining data augmentation for training (Top) and testing (Bottom) #################

data_transform = transforms.Compose(

     [
            transforms.ToTensor(),
            transforms.Resize((84,84)),
            transforms.CenterCrop((84,84)),
            transforms.RandomVerticalFlip(0.5),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomApply([RandomRotation((90, 90))], p=0.5),
            transforms.RandomApply([RandomRotation((270, 270))], p=0.5),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
       ])

data_transform_test = transforms.Compose(

      [
        transforms.ToTensor(),
        transforms.Resize(84),
        transforms.CenterCrop(84),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
       ])
