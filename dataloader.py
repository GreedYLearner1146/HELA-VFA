from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from torch.utils.data import default_collate
from torchvision.transforms import v2

#################### Cutmix or MixUp data augmentation ################################

cutmix = v2.CutMix(num_classes=100)
mixup = v2.MixUp(num_classes=100)
cutmix_or_mixup = v2.RandomChoice([cutmix, mixup])

def collate_fn(batch):
    return cutmix_or_mixup(*default_collate(batch))

########################### miniImageNet dataloader ###################################

class miniImageNet_CustomDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = data_transform
    def __getitem__(self, idx):
        label = self.labels[idx]
        image = self.images[idx]
        image = self.transform(np.array(image))
        return image, label
    def __len__(self):
        return len(self.labels)

#################################### Dataloader ##############################

train_dataset = miniImageNet_CustomDataset(new_X_train,new_y_train, transform=[data_transform, Augment]) # Combined data transform. Augment is from Data_Augmentation.py
val_dataset =  miniImageNet_CustomDataset(new_X_val,new_y_val, transform=[data_transform_valtest])

train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn) # Collate_fn called on here.
val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=True) # Collate_fn called on here.
