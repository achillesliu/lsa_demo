# -*- coding: utf-8 -*-
"""
Created on Mon Jun 20 15:05:49 2016

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
distances = pickle.load(open('Results/distances.p'))


#%%
distances_d = defaultdict(list)
length = distances.shape[1]
for p in range(length):
    for p1 in range(length):
        if p == p1:
            continue
        else:
            distances_d[np.abs(p - p1)].append(distances[p, p1])

range_avg_distances = np.zeros([len(distances_d), 2])
range_avg_distances[:, 0] = np.arange(1, len(distances_d)+1)
for rg in distances_d:
    range_avg_distances[rg-1, 1] = np.mean(np.array(distances_d[rg]))