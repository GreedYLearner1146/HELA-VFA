from torch.utils.data import Dataset
from torch.utils.data import DataLoader

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

# new_X_train,new_y_train, new_X_test, new_y_test from 'Data_Preparation.py'

train_dataset = miniImageNet_CustomDataset(new_X_train,new_y_train, transform=data_transform)
test_dataset =  miniImageNet_CustomDataset(new_X_test, new_y_test, transform=data_transform_test)

train_dataloader = DataLoader(train_dataset, batch_size=1024, shuffle=True)  
test_dataloader = DataLoader(test_dataset, batch_size=1024, shuffle=True)   # 
