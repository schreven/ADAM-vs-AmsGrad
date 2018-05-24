# -*- coding: utf-8 -*-
"""
Created on Wed May 23 18:16:08 2018

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

from visualizing import plot_acc_loss
from models import create_mnist_model


from training_and_validating import train_validate_kfold

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


model = create_mnist_model()
kfold = 5
nb_epochs = 40
lr = 1e-1
mini_batch = 100
train_dataset, train_loader, test_dataset, test_loader = get_datasets(mini_batch_size = mini_batch)
interstates = True

optimizer_ = optim.SGD
opt_parameters_SGD = [lr]

train_loss_kfold, val_loss_kfold, train_acc_kfold, val_acc_kfold = train_validate_kfold(\
        model, optimizer_, opt_parameters_SGD, train_dataset, kfold=kfold, shuffle=True, nb_epochs = nb_epochs, mini_batch_size = mini_batch, interstates = interstates)

plot_acc_loss(train_loss_kfold, val_loss_kfold, train_acc_kfold, val_acc_kfold)

save_array = [train_loss_kfold, val_loss_kfold, train_acc_kfold, val_acc_kfold]
np.save(os.path.join('arrays_and_images','1st_kf=5_epo=40_lr=1e-1_btch=100_SGD_inter'),save_array)


"""
model = create_mnist_model()
kfold = 2
nb_epochs = 5
lr = 1e-1
beta1, beta2 = 0.9, 0.99
amsgrad = False
mini_batch = 1000
train_dataset, train_loader, test_dataset, test_loader = get_datasets(mini_batch_size = mini_batch)
interstates = True

optimizer_ = optim.Adam
opt_parameters_Adam = [lr, (0.9, 0.99), amsgrad]

train_loss_kfold, val_loss_kfold, train_acc_kfold, val_acc_kfold = train_validate_kfold(\
        model, optimizer_, opt_parameters_Adam, train_dataset, kfold=kfold, shuffle=True, nb_epochs = nb_epochs, mini_batch_size = mini_batch, interstates = interstates)

plot_acc_loss(train_loss_kfold, val_loss_kfold, train_acc_kfold, val_acc_kfold)
"""