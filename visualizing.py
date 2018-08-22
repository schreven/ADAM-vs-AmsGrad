# -*- coding: utf-8 -*-
"""
Created on Wed May 23 18:21:45 2018

@authors: Musluoglu Cem Ates, Novakovic Milica, van Schreven Cyril
"""

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def plot_acc_loss(tr_loss, te_loss, tr_acc, te_acc, title="MNIST"):   
    plt.figure()
    title="MNIST loss"
    sns.tsplot(np.array(tr_loss)).set_title(title)
    sns.tsplot(np.array(te_loss), color = 'r')
    plt.legend(['Train', 'Validation'])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.figure()
    title="MNIST accuracy"
    sns.tsplot(np.array(tr_acc)).set_title(title)
    sns.tsplot(np.array(te_acc), color = 'r')
    plt.legend(['Train', 'Validation'])
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')