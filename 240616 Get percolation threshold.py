# -*- coding: utf-8 -*-
"""
Created on Wed May 11 14:48:52 2016

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
def get_avg_comp_size(m2, weights, ind):
    m = m2.copy()
    weight = weights[ind]
    m[m <= weight] = 0.0
    
    g = nx.from_numpy_matrix(m)
    subgraphs_iter = nx.connected_component_subgraphs(g)
    count = 0
    size_l = []
    for c in subgraphs_iter:
        count += 1
        size_l.append(len(c.nodes()))
    return count, np.mean(np.array(size_l))


def right_is_small(size_l, size_r):
    return (size_l - size_r) >= (0.33 * size_l)


def get_left_half(ind_l, ind_r, len_weights, gen):
    interval = int(len_weights / (2 ** gen))
    new_ind_r = ind_l + interval
    return ind_l, new_ind_r


def get_right_half(ind_l, ind_r, len_weights, gen):
    interval = int(len_weights / (2 ** gen))
    new_ind_l = ind_r - interval
    return new_ind_l, ind_r
    

def get_percolation_threshold(distances2):
    distances = distances2.copy()
    weights = np.sort(np.unique(distances.reshape([1, distances.shape[0] ** 2])))
    l_weights = len(weights)
    ind_l = 0
    ind_r = l_weights - 1
    gen = 1

    # Grant grains.
    while ind_r - ind_l >= 30:
                
        # Check the left half of the current interval.
        left_half_ind_l, left_half_ind_r = get_left_half(ind_l, ind_r, l_weights, gen)
        _, size_l = get_avg_comp_size(distances, weights, left_half_ind_l)
        _, size_r = get_avg_comp_size(distances, weights, left_half_ind_r)
        
        print left_half_ind_l, left_half_ind_r
        print size_l, size_r
        
        # Choose the next interval from the condition.
        if right_is_small(size_l, size_r):
            # When the size right is small, the condition is fullfilled,
            # which means the percolation threshold is in the left half
            # interval. So the new interval is the left half interval.
            ind_l = left_half_ind_l
            ind_r = left_half_ind_r
        else:
            # The other condition.
            right_half_ind_l, right_half_ind_r = get_right_half(ind_l, ind_r, l_weights, gen)
            ind_l = right_half_ind_l
            ind_r = right_half_ind_r
        
        # Goes into the next generation.
        gen += 1
        print ' '

    # Small loop. Fine grain.
    comps = []
    for ind in range(ind_l, ind_r + 1):
        num, avg_size = get_avg_comp_size(distances, weights, ind)
        comps.append([ind, weights[ind], num, avg_size])
    comps = np.array(comps)
    
    
    # Get the percolation threshold.
    if (1 in comps[:, 2]) and (2 in comps[:, 2]):
        print 'Everything is good'
    else:
        print 'Be aware of this book!'
    
    ind_thres = np.where(comps[:, 2] == 2)[0][0]
    threshold = comps[ind_thres, 1]

    return threshold
    
    
def get_graph_from_percolation(distances):
    weight = get_percolation_threshold(distances)
    m = distances.copy()
    m[m <= weight] = 0.0
    g = nx.from_numpy_matrix(m)
    return g

#%%
fps = ge.fp_list_dir('Results/Results for Dickens', 'p')

for fp in fps:
    d = pickle.load(open(fp))
    
    for key in d:
        
        print 'Name of the book is %s' %key
        distances = d[key]
        g = get_graph_from_percolation(distances)
        
        name = splitext(key)[0].lower()
        nx.write_weighted_edgelist(g, name)
        
        print 'Get the graph.'
        
