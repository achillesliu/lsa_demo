# -*- coding: utf-8 -*-
"""
Created on Thu May 12 09:29:53 2016

@author: bo
"""

import os, json, re, random
from os.path import join, dirname, basename, split, splitext
from collections import Counter, defaultdict, OrderedDict

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

import general as ge

#%%
def normalization(v):
    s = np.sum(v)
    return v / s

def information(i, j, m):
    # m: a np matrix, i and j: numeric
    row_i = normalization(m[i, :])
    pij = row_i[j]
    Iij = - pij * np.log(pij)
    row_i_without_i = row_i[row_i != 0.0]
    Ii = sum(- row_i_without_i * np.log(row_i_without_i))
    return Iij / Ii


#%%
# Read the network.
distances = np.loadtxt('sub_text.txt')
g = nx.from_numpy_matrix(distances)


#%%
# Calculate for the path in the book.
informations_b = .0
for ind, node in enumerate(range(distances.shape[1]-1)):
    for ind1, node1 in enumerate(range(ind+1, distances.shape[1])):
        Iij = information(node, node1, distances)
        informations_b += Iij


#%%
# Calculate for the random path.

# Generate a random path.
infos = []
for count in range(20):
    nodes = range(distances.shape[1])
    path = []
    while len(nodes) != 0:
        next_node = random.sample(nodes, 1)[0]
        path.append(next_node)
        nodes.remove(next_node)
    
    informations = 0.0
    for ind in range(len(path) - 1):
        node = path[ind]
        for ind1 in range(ind+1, len(path)):
            node1 = path[ind1]
            Iij = information(node, node1, distances)
            informations += Iij
    infos.append(informations)


#%%



138.7645053






    





















