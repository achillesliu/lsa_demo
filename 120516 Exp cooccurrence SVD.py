# -*- coding: utf-8 -*-
"""
Created on Thu May 12 16:08:48 2016

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

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
porter = PorterStemmer()


#%%
def count_cooccurrrence(word, word1, window_length, words_numbers):
    indices_v = np.where(words_numbers == word)
    indices_v1 = np.where(words_numbers == word1)
    distances = np.zeros([np.size(indices_v), np.size(indices_v1)])
    for ind in np.arange(np.size(indices_v)):
        index_v = indices_v[0][ind]
        distances[ind, :] = np.absolute(indices_v1 - index_v)
    threshold = window_length / 2
    return np.sum(distances <= threshold)

def cooccurrence_m(vocab_numbers, words_numbers, window_length):
    m = np.zeros([len(vocab_numbers), len(vocab_numbers)])
    for ind,v in list(enumerate(vocab_numbers))[:-1]:
        
        print 'calculating %d word' %ind
        
        for ind1 in range(ind+1, len(vocab_numbers)):
            v1 = vocab_numbers[ind1]
            m[ind, ind1] = count_cooccurrrence(v, v1, window_length, words_numbers)
    m = m + m.transpose()
    return m


#%%
# Pre-parameters.
threshold = 4


#%%
# Reading and pre-treatment the text.
# The words are filtered for multiple times.

_, words = ge.read_file('1813 PRIDE AND PREJUDICE.txt', use_zip=0, zf='none')

length_original_text = len(words)

# Pre-treatment.
# stopwords
stops = stopwords.words('english')
words = [w for w in words if not w in stops]
# stem
words = [porter.stem(w) for w in words]
vocab = list(set(words))
# threshold for number of appearances
c = Counter(words)
vocab = [w for w,count in c.items() if count >= threshold]
words = [w for w in words if w in vocab]


#%%
# Map strings (words, vocab) into numbers for speed.
# For any string list, convert to numpy arrays, then feed them into the function.

# test
# vocab = [1, 2, 3, 4]
# words = [1,3,2,2,4,1,3,2,3]

words_numbers = [vocab.index(w) for w in words]
words_numbers = np.array(words_numbers)
vocab_numbers = np.arange(len(vocab))


#%%
# Calculate only half the length for test.
words_numbers = words_numbers[:10000]


#%%
# Parameters.
window_length = 200
if len(words_numbers) % window_length:
    num_windows = len(words_numbers) / window_length + 1
else:
    num_windows = len(words_numbers) / window_length


#%%
# Map the text into a cooccurrence matrix.
m = cooccurrence_m(vocab_numbers, words_numbers, window_length)


#%%
# Build the random matrix.
c = Counter(words_numbers)
r = np.zeros([np.size(vocab_numbers), np.size(vocab_numbers)])
for ind,w in list(enumerate(vocab_numbers))[:-1]:
    for ind1 in range(ind+1, len(vocab_numbers)):
        w1 = vocab_numbers[ind1]
        r[ind, ind1] = float(window_length) * c[w] * c[w] / length_original_text
r = r + r.transpose()


#%%
# Get the normalized matrix.
n = m - r
n[n < 0] = 0


#%%
# svd
u, s, v = np.linalg.svd(n, full_matrices=False)


#%%
# After checking, the first 500 deminsions are enough.
word_vectors = np.dot(u[:, :200], np.diag(s[:200]))
word_vectors = np.absolute(word_vectors)


#%%
def get_svd_basis_subtext(word_vectors, subtext):
    c = Counter(subtext)
    parag_v = np.zeros([1, 200])
    for w, count in c.items():
        tmpv = word_vectors[w, :] * count
        parag_v += tmpv
    return parag_v[0]
    

def normalizaton(v):
    summ = np.sum(v)
    return v / summ


#%%
paragraph_vs = np.zeros([num_windows, 200])
for wind in range(num_windows):
    subtext = words_numbers[wind * window_length: (wind+1) * window_length]
    paragraph_v = get_svd_basis_subtext(word_vectors, subtext)
    paragraph_v = normalizaton(paragraph_v)
    paragraph_vs[wind, :] = paragraph_v


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
for ind in range(paragraph_vs.shape[0] - 1):
    dis = paragraph_vs[ind, :]
    
    print 'calculating the %d text' %ind
    
    for ind1 in range(ind+1, paragraph_vs.shape[0]):
        dis1 = paragraph_vs[ind1, :]
        distances[ind, ind1] = jensen_shannon_divergence(dis, dis1)

distances = distances + distances.transpose()


#%%
g = nx.from_numpy_matrix(distances)


#%%
def information(i, j, m):
    # m: a np matrix, i and j: numeric
    row_i = m[i, :]
    pij = row_i[j]
    Iij = - pij * np.log(pij)
    row_i_without_i = row_i[row_i != 0.0]
    Ii = sum(- row_i_without_i * np.log(row_i_without_i))
    return Iij / Ii


#%%
information_test = .0
for ind, node in enumerate(range(distances.shape[1]-1)):
    for ind1, node1 in enumerate(range(ind+1, distances.shape[1])):
        Iij = information(node, node1, distances)
        information_test += Iij
        

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





























