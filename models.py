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