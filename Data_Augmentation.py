########################## Data Augmentation steps #############################

from torchvision import transforms
from torchvision.transforms import v2

################################ The main ART-HELANet Codes #################################
################# We emphasize the role of data augmentation here. ##########################

# The data augmentation selected for the train, valid and test datasets are as below.

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

data_transform_valid = transforms.Compose(

      [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
       ])

data_transform_test = transforms.Compose(

      [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
       ])

import torchvision.transforms as T

# Based on the Augmentation procedure laid out in the simCLR paper, as our Hesim loss function has an analogous structure. #

class Augment:
   """
   A stochastic data augmentation module
   Transforms any given data example randomly
   resulting in two correlated views of the same example,
   denoted x ̃i and x ̃j, which we consider as a positive pair.
   """

   def __init__(self, img_size, s=1):
       color_jitter = T.ColorJitter(
           0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s
       )
       # 10% of the image
       blur = T.GaussianBlur((3, 3), (0.1, 2.0))

       self.train_transform = torch.nn.Sequential(
           T.RandomResizedCrop(size=img_size),
           T.RandomHorizontalFlip(p=0.5), 
           T.RandomApply([color_jitter], p=0.5),
           T.RandomApply([blur], p=0.5),
           T.RandomGrayscale(p=0.5), 
           # imagenet stats
           T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
       )

   def __call__(self, x):
       return self.train_transform(x), self.train_transform(x)
