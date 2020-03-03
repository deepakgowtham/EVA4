import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import OneCycleLR
import matplotlib.pyplot as plt
#%matplotlib inline
import numpy as np
import torchvision
import torchsummary
from torchsummary import summary

class Net(nn.Module):
  
  def __init__(self):
    super(Net, self).__init__()
    
    self.conv_block1= nn.Sequential(
        nn.Conv2d(in_channels=3, out_channels=16, kernel_size=(3,3),bias=False, padding=1, padding_mode='same'), #RF 3
        nn.BatchNorm2d(16),
        nn.ReLU(),
		nn.Dropout2d(0.1),
        nn.Conv2d(in_channels=16, out_channels=32 , kernel_size=(3,3), bias=False, padding=1, padding_mode='same'), #RF 5
        nn.BatchNorm2d(32),
        nn.ReLU(),
		nn.Dropout2d(0.1),
        nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3,3), bias=False, padding=1, padding_mode='same',dilation=2), #RF7
        nn.BatchNorm2d(64),
        nn.ReLU(),
		nn.Dropout2d(0.1),
        nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3,3), bias=False, padding=1, padding_mode='same'), #RF 9
        nn.BatchNorm2d(128),
        nn.ReLU(),
		nn.Dropout2d(0.1),
        nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3,3),bias=False, padding=1, padding_mode='same'), #RF 11
        nn.BatchNorm2d(256),
        nn.ReLU(),
		nn.Dropout2d(0.1))
    
    self.trans_block1= nn.Sequential(
        nn.MaxPool2d(2,2),   #RF12
        nn.Conv2d(in_channels=256, out_channels=32, kernel_size=(1,1), bias=False)) #RF 12
    
    self.conv_block2= nn.Sequential(
        nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3,3), bias=False, padding=1, padding_mode='same'), #RF 16
        nn.BatchNorm2d(64),
        nn.ReLU(),
		nn.Dropout2d(0.1),
        nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3,3), bias=False, padding=1, padding_mode='same'), #RF 20
        nn.BatchNorm2d(128),
        nn.ReLU(),
		nn.Dropout2d(0.1),
        nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3,3), bias=False, padding=1, groups=128, padding_mode='same'), # 24
        nn.BatchNorm2d(256),
        nn.ReLU(), 
		nn.Dropout2d(0.1))
    
    self.trans_block2= nn.Sequential(
        nn.MaxPool2d(2,2),  #RF 26
        nn.Conv2d(in_channels=256, out_channels=32, kernel_size=(1,1), bias=False)) #RF26
    
    self.conv_block3= nn.Sequential(
        nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3,3), bias=False, padding=1, padding_mode='same'), #RF34
        nn.BatchNorm2d(64),
        nn.ReLU(),
		nn.Dropout2d(0.1),
        nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3,3), bias=False, padding=1, padding_mode='same'), #RF 42
        nn.BatchNorm2d(128),
        nn.ReLU(),
		nn.Dropout2d(0.1),
        nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3,3), bias=False, padding=1,groups=128, padding_mode='same'), #RF 50
        nn.BatchNorm2d(256),
        nn.ReLU(),
		nn.Dropout2d(0.1))
    
    self.trans_block3= nn.Sequential(
        nn.MaxPool2d(2,2), #54
        nn.Conv2d(in_channels=256, out_channels=32, kernel_size=(1,1), bias=False)) #54

    self.conv_block4=nn.Sequential(
        nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3,3),bias=False, padding=1, padding_mode='same'), #RF 66
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.AvgPool2d(3),  #RF74
        nn.Conv2d(in_channels=64,out_channels=10, kernel_size=(1,1),bias=False)
    )
    
  def forward(self, x):
    x=self.conv_block1(x)
    x= self.trans_block1(x)
    x=self.conv_block2(x)
    x=self.trans_block2(x)
    x=self.conv_block3(x)
    x=self.trans_block3(x)
    x=self.conv_block4(x)
    x=x.view(-1,10)
    return F.log_softmax(x)
	

def disp_summary(model):
	#use_cuda= torch.cuda.is_available()
	#device=torch.device('cuda' if use_cuda else 'cpu')
	##model=Net().to(device)
	summary(model, input_size=(3,32,32))
