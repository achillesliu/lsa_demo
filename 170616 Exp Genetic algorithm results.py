# -*- coding: utf-8 -*-
"""
Created on Fri Jun 17 11:35:07 2016

@author: bo
"""

import os, json, re, random, sys, copy
from os.path import join, dirname, basename, split, splitext
from collections import Counter, defaultdict, OrderedDict

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

sys.path.append('../tools')
import general as ge

from operator import add
import pickle
from operator import itemgetter

#%%
[fitnesses, best_individuals] = pickle.load(open('Results/genetic evolution.p'))

#%%
def smooth(seq, smooth_window):
    half_window = smooth_window / 2
    smoothed = []
    for data in seq[half_window : -half_window]:
        ind = np.where(seq == data)[0][0]
        slice = seq[ind - half_window : ind + 1 + half_window]
        mean = np.mean(slice)
        smoothed.append(mean)
    return smoothed


smootheds = []
for seq in best_individuals:
    
    seq = np.array(seq)
    smootheds.append(smooth(seq, 100))

#%%
fig = plt.figure(figsize=[16, 6])
ax = fig.add_subplot(111)
for ind, seq in enumerate(smootheds[-10:]):
    ax.plot(seq, label=str(ind), alpha=.5)
ax.legend()