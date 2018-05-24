# -*- coding: utf-8 -*-
"""
Created on Wed May 23 18:57:41 2018

@author: Bob
"""

import torch
import math
import os

from torch import optim
from torch import Tensor
from torch.autograd import Variable
from torch import nn
import torchvision
from torchvision import datasets
import torchvision.transforms as transforms
import seaborn as sns
from sklearn.model_selection import KFold
from torch.utils.data.sampler import SubsetRandomSampler
import matplotlib.pyplot as plt
import numpy as np

from torch.nn import functional as F


"""
def create_mnist_model():
  return nn.Sequential(
      nn.Linear(784, 100),
      nn.ReLU(),
      nn.Linear(100, 10)
  )
  """

class create_mnist_model(nn.Module):
    def __init__(self):
        super(create_mnist_model, self).__init__()
        self.fc1 = nn.Linear(784, 100)
        self.fc2 = nn.Linear(100, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x.view(-1,784)))
        x = self.fc2(x)
        return x
    
    
def get_datasets(mini_batch_size=100):
  transform = transforms.ToTensor()
  transform=transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((0.1307,), (0.3081,)),])  #these are the mean and std of MNIST
  root = './data'
  if not os.path.exists(root):
    os.mkdir(root)
  # Load and transform data
  train_dataset = torchvision.datasets.MNIST(root, train=True, download=True, transform=transform)
  train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=mini_batch_size, shuffle=True, num_workers=2)
  
  

  test_dataset = torchvision.datasets.MNIST(root, train=False, download=True, transform=transform)
  test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=mini_batch_size, shuffle=False, num_workers=2)
  return  train_dataset, train_loader, test_dataset, test_loader
    