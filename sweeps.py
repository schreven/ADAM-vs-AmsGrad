# -*- coding: utf-8 -*-
"""
Created on Thu May 24 15:35:30 2018

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
from models import create_1layer_model, create_convex_model, get_datasets
from training_and_validating import train_validate_kfold

def grid_search_lr(model_ = create_1layer_model):   
    results = {'lr':[], 'train_loss':[], 'train_acc':[], 'val_loss':[], 'val_acc':[]}
    
    # Define minibatch size
    kfold = 5
    nb_epochs = 2
    mini_batch = 100
    train_dataset, train_loader, test_dataset, test_loader = get_datasets(mini_batch_size = mini_batch)
    interstates = False
    run_once = True
    
    amsgrad = False
    beta1 = 0.91
    beta2 = 0.999
    
    optimizer_ = optim.Adam
    
    # this should be best lr
    
    lr_list = [0.00001, 0.0001, 0.001, 0.01, 0.1]
    
    for lr in lr_list:
        results['lr'].append(lr)
        
        print("lr = ",lr)
        title = "lr = "+str(lr)
        
        opt_parameters_Adam = [lr, (beta1, beta2), amsgrad]
        
        train_loss_kfold, val_loss_kfold, train_acc_kfold, val_acc_kfold = train_validate_kfold(\
            model_, optimizer_, opt_parameters_Adam, train_dataset, kfold=kfold, shuffle=True, nb_epochs = nb_epochs, mini_batch_size = mini_batch, interstates = interstates, run_once = run_once)
        # Plot
        
        plot_acc_loss(train_loss_kfold, val_loss_kfold, train_acc_kfold, val_acc_kfold , title)
        print("Train accuracy =  ",train_acc_kfold)
        print("Val accuracy =  ",val_acc_kfold)
        
        results['train_loss'].append(train_loss_kfold)
        results['train_acc'].append(train_acc_kfold)
        results['val_loss'].append(val_loss_kfold)
        results['val_acc'].append(val_acc_kfold)
    np.save(os.path.join('arrays_and_images','grid_kf=5_epo=120_b1=0.91_b2=0.999_btch=100_Adam_lr_ro_convex'),results)
        
        

    
def grid_search_beta(model_ = create_1layer_model):   
    results = {'beta1': [], 'beta2':[], 'train_loss':[], 'train_acc':[], 'val_loss':[], 'val_acc':[]}
    
    # Define minibatch size
    kfold = 5
    nb_epochs = 40
    lr = 1e-3
    run_once = True
    
    mini_batch = 100
    train_dataset, train_loader, test_dataset, test_loader = get_datasets(mini_batch_size = mini_batch)
    interstates = False
    amsgrad = False
    
    optimizer_ = optim.Adam
    
    # this should be best lr
    
    
    beta1_list = [0.89, 0.9, 0.91, 0.92]
    beta2_list = [0.99, 0.994, 0.998, 0.999]
    for beta1 in beta1_list:
      results['beta1'].append(beta1)
      for beta2 in beta2_list:
        results['beta2'].append(beta2)
        
        print("beta1 = ",beta1, " beta2 = ", beta2)
        title = "beta1 = "+str(beta1)+" beta2 = "+str(beta2)
    
        opt_parameters_Adam = [lr, (beta1, beta2), amsgrad]
        
        train_loss_kfold, val_loss_kfold, train_acc_kfold, val_acc_kfold = train_validate_kfold(\
            model_, optimizer_, opt_parameters_Adam, train_dataset, kfold=kfold, shuffle=True, nb_epochs = nb_epochs, mini_batch_size = mini_batch, interstates = interstates, run_once = run_once)
        # Plot
    
        plot_acc_loss(train_loss_kfold, val_loss_kfold, train_acc_kfold, val_acc_kfold , title)
        #train_loss_kfold = np.mean(train_loss_kfold, axis = 0)
        #val_loss_kfold = np.mean(val_loss_kfold, axis = 0)
        #train_acc_kfold = np.mean(train_acc_kfold, axis = 0)
        #val_acc_kfold = np.mean(val_acc_kfold, axis = 0)
        #train_acc_kfold = np.mean(train_acc_kfold, axis = 0)
        print("Train accuracy =  ",train_acc_kfold)
    
        #te_acc = np.mean(te_acc, axis = 0)
        print("Val accuracy =  ",val_acc_kfold)
    
        results['train_loss'].append(train_loss_kfold)
        results['train_acc'].append(train_acc_kfold)
        results['val_loss'].append(val_loss_kfold)
        results['val_acc'].append(val_acc_kfold)
    np.save(os.path.join('arrays_and_images','grid_kf=5_epo=40_lr=1e-3_btch=100_Adam_beta1_beta2_ro_convex'),results)
