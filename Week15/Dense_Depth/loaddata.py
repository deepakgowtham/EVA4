import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image
import random
from nyu_transform import *


class depthDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_file, transform=None):
        self.frame = pd.read_csv(csv_file, header=None,skiprows=[0])
        self.transform = transform

    def __getitem__(self, idx):
        image_name = self.frame.iloc[idx, 0]
        depth_name = self.frame.iloc[idx, 1]

        image = Image.open(image_name).resize((64,64))
        depth = Image.open(depth_name).resize((64,64))

        sample = {'image': image, 'depth': depth}

        if self.transform:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return len(self.frame)


def getTrainingData(batch_size=128):
    __imagenet_pca = {
        'eigval': torch.Tensor([0.2175, 0.0188, 0.0045]),
        'eigvec': torch.Tensor([
            [-0.5675,  0.7192,  0.4009],
            [-0.5808, -0.0045, -0.8140],
            [-0.5836, -0.6948,  0.4203],
        ])
    }
    __imagenet_stats = {'mean': [0.3931, 0.3785, 0.3606],
                        'std': [0.1965, 0.1813, 0.1779]}

    transformed_training = depthDataset(csv_file='/content/Data/Mask_RCNN/train.csv',
                                        transform=transforms.Compose([
                                            #Scale(240),
                                            #RandomHorizontalFlip(),
                                            #RandomRotate(5),
                                            #CenterCrop([304, 228], [152, 114]),
                                            ToTensor(),
                                            #Lighting(0.1, __imagenet_pca[
                                             #   'eigval'], __imagenet_pca['eigvec']),
                                            #ColorJitter(
                                             #   brightness=0.4,
                                              #  contrast=0.4,
                                               # saturation=0.4,
                                            #),
                                            Normalize(__imagenet_stats['mean'],
                                                      __imagenet_stats['std'])
                                        ]))

    dataloader_training = DataLoader(transformed_training, batch_size,
                                     shuffle=True, num_workers=6, pin_memory=True)

    return dataloader_training



def getTestingData(batch_size=64):

    __imagenet_stats = {'mean': [0.3931, 0.3785, 0.3606],
                        'std': [0.1965, 0.1813, 0.1779]}
    # scale = random.uniform(1, 1.5)
    transformed_testing = depthDataset(csv_file='/content/Data/Mask_RCNN/test1.csv',
                                       transform=transforms.Compose([
                                           #Scale(240),
                                           #CenterCrop([304, 228], [152, 114]),
                                           ToTensor(is_test=True),
                                           Normalize(__imagenet_stats['mean'],
                                                     __imagenet_stats['std'])
                                       ]))

    dataloader_testing = DataLoader(transformed_testing, batch_size,
                                    shuffle=False, num_workers=4, pin_memory=True)

    return dataloader_testing

