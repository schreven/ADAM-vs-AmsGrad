# -*- coding: utf-8 -*-
"""
Created on Thu May 24 15:59:04 2018

@author: Bob
"""

import os
import pylab
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


### SGD

save_array = np.load(os.path.join('arrays_and_images','1st_kf=5_epo=40_lr=1e-1_btch=100_SGD_inter.npy'))


train_loss_kfold, val_loss_kfold, train_acc_kfold, val_acc_kfold = save_array


plt.figure()
title="MNIST loss"
sns.tsplot(np.array(train_loss_kfold)).set_title(title)
sns.tsplot(np.array(val_loss_kfold), color = 'r')
plt.legend(['Train', 'Validation'])
plt.xlabel('Epoch')
plt.ylabel('Loss')

pylab.savefig(os.path.join('arrays_and_images','first_vis__.png'))



plt.figure()
title="MNIST accuracy"
sns.tsplot(np.array(train_acc_kfold)).set_title(title)
sns.tsplot(np.array(val_acc_kfold), color = 'r')
plt.legend(['Train', 'Validation'])
plt.xlabel('Epoch')
plt.ylabel('Accuracy')

pylab.savefig(os.path.join('arrays_and_images','second_vis__.png'))

print('SGD: {}' .format(np.mean(val_acc_kfold, axis=0)[-1]))

### SIMPLE ADAM

save_array = np.load(os.path.join('arrays_and_images','2nd_kf=4_epo=40_lr=1e-3_btch=100_Adam_inter.npy'))


train_loss_kfold, val_loss_kfold, train_acc_kfold, val_acc_kfold = save_array


plt.figure()
title="MNIST loss"
sns.tsplot(np.array(train_loss_kfold)).set_title(title)
sns.tsplot(np.array(val_loss_kfold), color = 'r')
plt.legend(['Train', 'Validation'])
plt.xlabel('Epoch')
plt.ylabel('Loss')

pylab.savefig(os.path.join('arrays_and_images','third_vis.png'))



plt.figure()
title="MNIST accuracy"
sns.tsplot(np.array(train_acc_kfold)).set_title(title)
sns.tsplot(np.array(val_acc_kfold), color = 'r')
plt.legend(['Train', 'Validation'])
plt.xlabel('Epoch')
plt.ylabel('Accuracy')

pylab.savefig(os.path.join('arrays_and_images','fourth_vis.png'))

print('ADAM: {}' .format(np.mean(val_acc_kfold, axis=0)[-1]))

### Updated ADAM: AMSGRAD

save_array = np.load(os.path.join('arrays_and_images','3rd_kf=4_epo=40_lr=1e-3_btch=100_AMSGRAD_inter.npy'))


train_loss_kfold, val_loss_kfold, train_acc_kfold, val_acc_kfold = save_array


plt.figure()
title="MNIST loss"
sns.tsplot(np.array(train_loss_kfold)).set_title(title)
sns.tsplot(np.array(val_loss_kfold), color = 'r')
plt.legend(['Train', 'Validation'])
plt.xlabel('Epoch')
plt.ylabel('Loss')

pylab.savefig(os.path.join('arrays_and_images','fifth_vis.png'))



plt.figure()
title="MNIST accuracy"
sns.tsplot(np.array(train_acc_kfold)).set_title(title)
sns.tsplot(np.array(val_acc_kfold), color = 'r')
plt.legend(['Train', 'Validation'])
plt.xlabel('Epoch')
plt.ylabel('Accuracy')

pylab.savefig(os.path.join('arrays_and_images','sixth_vis.png'))

print('Amsgrad: {}' .format(np.mean(val_acc_kfold, axis=0)[-1]))

### GRID  beta
res = np.load(os.path.join('arrays_and_images','grid_kf=4_epo=40_lr=1e-3_btch=100_Adam_beta1_beta2.npy'))
res = dict(res.tolist())
beta1_list = [0.88, 0.89, 0.9, 0.91, 0.92]
beta2_list = [0.986, 0.99, 0.994, 0.998, 0.999, 0.9999]
num_b1 = len(beta1_list)
num_b2 = len(beta2_list)

val_acc = np.reshape(np.mean(res['val_acc'], axis=1),(num_b1,num_b2))
##correcting mistake in beta1_list
tmp = val_acc[2].copy()
val_acc[2] = val_acc[3]
val_acc[3] = tmp

plt.figure()
sns.heatmap(val_acc, xticklabels=np.reshape(np.array(beta2_list),(-1,1)), yticklabels=np.reshape(np.array(beta1_list),(-1,1))).set_title('Validating accuracy mean')

pylab.savefig(os.path.join('arrays_and_images','grid_vis.png'))


### GRID LR
res = np.load(os.path.join('arrays_and_images','grid_kf=4_epo=120_b1=0.91_b2=0.999_btch=100_Adam_lr.npy'))
res = dict(res.tolist())
lr_list = [0.00001, 0.0001, 0.001, 0.01, 0.1]

num_lr = len(lr_list)
val_acc = np.mean(res['val_acc'], axis=1)

plt.figure()
plt.semilogx(lr_list, val_acc)
plt.title('Validating accuracy mean')
pylab.savefig(os.path.join('arrays_and_images','lr_vis_1.png'))

plt.figure()
plt.semilogx(lr_list[:-1], val_acc[:-1])
plt.title('Validating accuracy mean')
pylab.savefig(os.path.join('arrays_and_images','lr_vis_2.png'))

########### CONVEX ##########

### SGD CONVEX
save_array = np.load(os.path.join('arrays_and_images','1st_kf=5_epo=40_lr=1e-1_btch=100_SGD_inter_ro_convex.npy'))


train_loss_kfold, val_loss_kfold, train_acc_kfold, val_acc_kfold = save_array


plt.figure()
title="MNIST loss"
sns.tsplot(np.array(train_loss_kfold)).set_title(title)
sns.tsplot(np.array(val_loss_kfold), color = 'r')
plt.legend(['Train', 'Validation'])
plt.xlabel('Epoch')
plt.ylabel('Loss')

pylab.savefig(os.path.join('arrays_and_images','first_vis_convex.png'))



plt.figure()
title="MNIST accuracy"
sns.tsplot(np.array(train_acc_kfold)).set_title(title)
sns.tsplot(np.array(val_acc_kfold), color = 'r')
plt.legend(['Train', 'Validation'])
plt.xlabel('Epoch')
plt.ylabel('Accuracy')

pylab.savefig(os.path.join('arrays_and_images','second_vis_convex.png'))

print('SGD: {}' .format(np.mean(val_acc_kfold, axis=0)[-1]))


### GRID LR CONVEX 
res = np.load(os.path.join('arrays_and_images','grid_kf=5_epo=120_b1=0.91_b2=0.999_btch=100_Adam_lr_ro_convex.npy'))
res = dict(res.tolist())
lr_list = [0.00001, 0.0001, 0.001, 0.01, 0.1]

num_lr = len(lr_list)
val_acc = np.mean(res['val_acc'], axis=1)

plt.figure()
plt.semilogx(lr_list, val_acc)
plt.title('Validating accuracy mean')
pylab.savefig(os.path.join('arrays_and_images','lr_vis_1_convex.png'))


############# REPORT ####################

### LOSS EPOCH SGD, ADAM, AMSGRAD  ONE_LAYER
save_array_sgd = np.load(os.path.join('arrays_and_images','1st_kf=5_epo=40_lr=1e-1_btch=100_SGD_inter.npy'))
save_array_adam = np.load(os.path.join('arrays_and_images','2nd_kf=4_epo=40_lr=1e-3_btch=100_Adam_inter.npy'))
save_array_amsgrad = np.load(os.path.join('arrays_and_images','3rd_kf=4_epo=40_lr=1e-3_btch=100_AMSGRAD_inter.npy'))

train_loss_kfold_sgd, val_loss_kfold_sgd, train_acc_kfold_sgd, val_acc_kfold_sgd = save_array_sgd
train_loss_kfold_adam, val_loss_kfold_adam, train_acc_kfold_adam, val_acc_kfold_adam = save_array_adam
train_loss_kfold_amsgrad, val_loss_kfold_amsgrad, train_acc_kfold_amsgrad, val_acc_kfold_amsgrad = save_array_amsgrad

plt.figure()
title="MNIST, one layer:"
plt.title(title)

sns.tsplot(np.array(val_loss_kfold_adam), color = 'b', linestyle = '--')
sns.tsplot(np.array(val_loss_kfold_amsgrad), color = 'r')
sns.tsplot(np.array(val_loss_kfold_sgd), color = 'g', linestyle = ':')

sns.tsplot(np.array(train_loss_kfold_adam), color = 'm', linestyle = '--')
sns.tsplot(np.array(train_loss_kfold_amsgrad), color = 'y')
sns.tsplot(np.array(train_loss_kfold_sgd), color = 'c', linestyle = ':')


plt.legend(['test Adam', 'test Amsgrad', 'test SGD', 'train Adam', 'train Amsgrad', 'train SGD'])
plt.xlabel('Epoch')
plt.ylabel('Loss')

pylab.savefig(os.path.join('arrays_and_images','report_loss_SGD_Adam_AmsGrad_valid.png'))

### LOSS EPOCH SGD, ADAM, AMSGRAD  CONV

save_array_sgd = np.load(os.path.join('arrays_and_images','1st_kf=5_epo=40_lr=1e-1_btch=100_SGD_inter_ro_convex.npy'))
save_array_adam = np.load(os.path.join('arrays_and_images','2nd_kf=4_epo=40_lr=1e-3_btch=100_Adam_inter_ro_CONV.npy'))
save_array_amsgrad = np.load(os.path.join('arrays_and_images','3rd_kf=4_epo=40_lr=1e-3_btch=100_AMSGRAD_inter_ro_CONV.npy'))

train_loss_kfold_sgd, val_loss_kfold_sgd, train_acc_kfold_sgd, val_acc_kfold_sgd = save_array_sgd
train_loss_kfold_adam, val_loss_kfold_adam, train_acc_kfold_adam, val_acc_kfold_adam = save_array_adam
train_loss_kfold_amsgrad, val_loss_kfold_amsgrad, train_acc_kfold_amsgrad, val_acc_kfold_amsgrad = save_array_amsgrad

plt.figure()
title="MNIST, convex:"
plt.title(title)

sns.tsplot(np.array(val_loss_kfold_adam), color = 'b', linestyle = '--')
sns.tsplot(np.array(val_loss_kfold_amsgrad), color = 'r')
sns.tsplot(np.array(val_loss_kfold_sgd), color = 'g', linestyle = ':')

sns.tsplot(np.array(train_loss_kfold_adam), color = 'm', linestyle = '--')
sns.tsplot(np.array(train_loss_kfold_amsgrad), color = 'y')
sns.tsplot(np.array(train_loss_kfold_sgd), color = 'c', linestyle = ':')


plt.legend(['test Adam', 'test Amsgrad', 'test SGD', 'train Adam', 'train Amsgrad', 'train SGD'])
plt.xlabel('Epoch')
plt.ylabel('Loss')

pylab.savefig(os.path.join('arrays_and_images','report_loss_SGD_Adam_AmsGrad_valid_CONV.png'))








