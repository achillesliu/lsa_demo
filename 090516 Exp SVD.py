# -*- coding: utf-8 -*-
"""
Created on Mon May 09 15:05:51 2016

@author: bo
"""

# -*- coding: utf-8 -*-

import os, json, re, random
from os.path import join, dirname, basename, split, splitext
from collections import Counter, defaultdict, OrderedDict

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

import general as ge

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
porter = PorterStemmer()

#%%
# Reading and pre-treatment the text.
# The words are filtered for multiple times.

_, words = ge.read_file('1813 PRIDE AND PREJUDICE.txt', use_zip=0, zf='none')

length_original_text = len(words)

# Pre-treatment.
# stopwords
words = [w for w in words if not w in stopwords.words('english')]
# stem
words = [porter.stem(w) for w in words]
vocab = list(set(words))


#%%
# Map strings (words, vocab) into numbers for speed.
# For any string list, convert to numpy arrays, then feed them into the function.

# test
# vocab = [1, 2, 3, 4]
# words = [1,3,2,2,4,1,3,2,3]

words_numbers = [vocab.index(w) + 1 for w in words] # +1 since 0 will be mixed with no appearance.
words_numbers = np.array(words_numbers)
vocab_numbers = np.arange(len(vocab)) + 1


#%%
# Parameters.

window_length = 200
num_windows = len(words_numbers) / window_length + 1


#%%
# Get the vocab-subtext matrix.

m = np.zeros([np.size(vocab_numbers),num_windows])
for ind in range(num_windows):
    tmp = words_numbers[ind*window_length:(ind+1)*window_length]
    intersection = np.intersect1d(vocab_numbers, tmp, assume_unique=True)
    c = Counter(intersection)
    for k,v in c.items():
        m[:, ind][k-1] = v


#%%
# Try SVD.
u, s, v = np.linalg.svd(m, full_matrices=0)

# Use the first 200 singular values.
number_features = 200
s = s[:number_features]
v = v[:number_features, :]
texts_dis = np.dot(np.diag(s), v).transpose()
texts_dis = np.absolute(texts_dis)

# normalization
def normalization(v):
    s = np.sum(v)
    return v / s

normed = np.zeros([num_windows, number_features])
for ind in range(texts_dis.shape[0]):
    normed[ind, :] = np.array(normalization(texts_dis[ind, :]))


#%%
# Jensen Shannon divergence
def shannon_entropy(dis):
    dis = [-pi * np.log(pi) for pi in dis]
    return sum(dis)

def jensen_shannon_divergence(dis, dis1):
    term = shannon_entropy(0.5 * dis + 0.5 * dis1)
    term1 = 0.5 * shannon_entropy(dis)
    term2 = 0.5 * shannon_entropy(dis1)
    return term - term1 - term2


#%%
# Calculate distance between each part.
distances = np.zeros([num_windows, num_windows])
for ind in range(normed.shape[0]-1):
    dis = normed[ind, :]
    
    print 'calculating the %d text' %ind
    
    for ind1 in range(ind+1, normed.shape[0]):
        dis1 = normed[ind1, :]
        distances[ind, ind1] = jensen_shannon_divergence(dis, dis1)

distances = distances + distances.transpose()


#%%
edgelist = []
for ind in range(distances.shape[0]-1):
    node = ind + 1
    for ind1 in range(ind+1, distances.shape[0]):
        node1 = ind1 + 1
        tmp = [node, node1, distances[ind, ind1]]
        edgelist.append(tmp)













