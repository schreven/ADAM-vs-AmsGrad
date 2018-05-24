# -*- coding: utf-8 -*-
"""
Created on Wed May 23 18:19:12 2018

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
  
from sklearn.model_selection import KFold
from torch.utils.data.sampler import SubsetRandomSampler

from models import create_mnist_model


def train_model(train_loader, model, criterion, optimizer):
    # Set model for training
    model.train()
    
    # Initialize counters to 0
    nb_correct = 0
    nb_elem = 0
    loss_epoch = 0
    
    # Iterate over batches
#     print("before first loop")
    for i, data in enumerate(train_loader):
        # Create Variable
        train_data, train_labels = data
        train_data = train_data.view(train_data.size(0),-1).float()
        
        inputs = train_data
        targets = train_labels
       # inputs = Variable(train_data)
       # targets = Variable(train_labels)
        if torch.cuda.is_available():
          inputs = inputs.cuda()
          targets = targets.cuda()

        # Clear gradients
        model.zero_grad()

        # Forward pass        
        outputs = model(inputs)

        # Predicted labels (the one with highest probability)
        pred_label = outputs.data.max(1)[1]

         # Compute and store the loss
        loss = criterion(outputs, targets)
        loss_epoch += loss.data[0]
        
        # Update nb. correct and nb of elem
        nb_correct += (pred_label == targets.data).sum()
        nb_elem += len(pred_label)

        # Backward pass
        loss.backward()
        optimizer.step()

    loss_epoch /=nb_elem
    acc_epoch = float(nb_correct)/nb_elem
    return loss_epoch, acc_epoch


def validate_model(val_loader, model, criterion):
    # Switch to evaluate mode
    model.eval()

    # Initialize counters
    nb_correct = 0
    nb_elem = 0
    loss_epoch = 0
    
    # Iterate over batches
    for data in val_loader:
        # Create Variable
        val_data, val_labels = data
        val_data = val_data.view(val_data.size(0),-1).float()
        #val_data.sub_(mean).div_(std)
        
        inputs = val_data
        targets = val_labels
        inputs = Variable(val_data)
        targets = Variable(val_labels)
        if torch.cuda.is_available():
          inputs = inputs.cuda()
          targets = targets.cuda()
        
        # Obtain predictions
        outputs = model(inputs)
        
        # Predicted label (highest probability)
        pred_label = outputs.data.max(1)[1]

        # Loss
        loss = criterion(outputs, targets)
        loss_epoch += loss.data[0]

        # Update nb. correct and nb. total
        nb_correct += (pred_label == targets.data).sum()
        nb_elem += len(pred_label)
        
    loss_epoch/=nb_elem
    acc_epoch = float(nb_correct)/nb_elem
    return loss_epoch, acc_epoch

def train_validate_kfold(model, optimizer_, opt_parameters, train_dataset, kfold=5, shuffle=True, nb_epochs = 150, mini_batch_size = 100, lr = 1e-1):
  criterion = nn.CrossEntropyLoss()
  kf = KFold(n_splits = kfold, shuffle=shuffle)

  # Define vectors to store results for each fold
  train_loss_kfold = []
  val_loss_kfold = []
  train_acc_kfold = []
  val_acc_kfold = []
  
  for train_index, val_index in kf.split(train_dataset.train_data):
    print("New split of data")
    train_sampler = SubsetRandomSampler(train_index)
    val_sampler = SubsetRandomSampler(val_index)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=mini_batch_size, sampler=train_sampler, drop_last=False)
#     print(len(train_loader))
    
    val_loader = torch.utils.data.DataLoader(train_dataset, batch_size=mini_batch_size, sampler=val_sampler, drop_last=False)
#     print(len(val_loader))
    
    model = create_mnist_model()
    if torch.cuda.is_available():
      model = model.cuda()
      #train_dataset.input, val_dataset.input = train_input.cuda(), train_target.cuda(),test_input.cuda(), test_target.cuda()

    # Store loss and accuracy per each epoch
    train_e_loss = []
    val_e_loss = []
    train_e_acc = []
    val_e_acc = []
    
    if torch.cuda.is_available():
      criterion = nn.CrossEntropyLoss().cuda()
    #optimizer = optim.SGD(model.parameters(), lr)
    #optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(beta1, beta2))
    optimizer = optimizer_(model.parameters(), *opt_parameters)
    
    for epoch in range(0, nb_epochs):
      if (epoch%10==0):
        print('At epoch number: {}'.format(epoch+1))
      # for this epoch calculate train loss, accuracy
      train_loss, train_acc = train_model(train_loader, model, criterion, optimizer)
      # Store them in list to be able to plot
      train_e_loss.append(train_loss)
      train_e_acc.append(train_acc)
      

      # Evaluate for epoch val loss and accuracy
      val_loss, val_acc = validate_model(val_loader, model, criterion)
      # Store them in the list to be able to plot
      val_e_loss.append(val_loss)
      val_e_acc.append(val_acc)
      print('epoch: {}'.format(epoch)) 
    
    # for k-fold sets, store loss and accuracy through epochs 
    train_loss_kfold.append(train_e_loss)
    val_loss_kfold.append(val_e_loss)
    train_acc_kfold.append(train_e_acc)
    val_acc_kfold.append(val_e_acc)
    
  return train_loss_kfold, val_loss_kfold, train_acc_kfold, val_acc_kfold


def train_test(model, optimizer_, opt_parameters, train_dataset, test_dataset, shuffle=True, nb_epochs = 150, mini_batch_size = 100, lr = 1e-1):
  criterion = nn.CrossEntropyLoss()
  optimizer = optim.SGD(model.parameters(), lr)
  
  from torch.utils.data.sampler import SubsetRandomSampler

  train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=mini_batch_size, sampler=train_sampler, drop_last=False)
#     print(len(train_loader))

  test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=mini_batch_size, sampler=val_sampler, drop_last=False)
#     print(len(val_loader))

  model = create_mnist_model()
  if torch.cuda.is_available():
    model = model.cuda()
 
  # Store loss and accuracy per each epoch
  train_e_loss = []
  test_e_loss = []
  train_e_acc = []
  test_e_acc = []

  if torch.cuda.is_available():
    criterion = nn.CrossEntropyLoss().cuda()
  #optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(beta1, beta2))
  optimizer = optimizer_(model.parameters(), *opt_parameters)

  for epoch in range(0, nb_epochs):
    if (epoch%10==0):
      print(epoch)
    # here we normalize data on train set and take mean and std for test set
    train_loss, train_acc, mean, std = train_model(train_loader, model, criterion, optimizer)
    # Store them in list to be able to plot
    train_e_loss.append(train_loss)
    train_e_acc.append(train_acc)

    # Evaluate for epoch val loss and accuracy
    test_loss, test_acc = validate_model(test_loader, model, criterion, mean, std)
    # Store them in the list to be able to plot
    test_e_loss.append(test_loss)
    test_e_acc.append(test_acc)
    
    # for k-fold sets, store loss and accuracy through epochs 
    train_loss_kfold.append(train_e_loss)
    test_loss_kfold.append(test_e_loss)
    train_acc_kfold.append(train_e_acc)
    test_acc_kfold.append(test_e_acc)
    
  return train_loss_kfold, test_loss_kfold, train_acc_kfold, test_acc_kfold