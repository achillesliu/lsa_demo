# -*- coding: utf-8 -*-
"""
Created on Fri May 13 09:45:57 2016

@author: bo
"""

import os, json, re, random, sys
from os.path import join, dirname, basename, split, splitext
from collections import Counter, defaultdict, OrderedDict

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

sys.path.append('../tools')
import general as ge

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
porter = PorterStemmer()


#%%
####################################################
# 1. Read original text.
####################################################


#%%
_, origianl_words = ge.read_file('../example texts/1813 PRIDE AND PREJUDICE.txt', use_zip=0, zf='none')

origianl_words = origianl_words[:40000]

lenght_original_text = len(origianl_words)


#%%
####################################################
# 2. For speed, shrink vocabulary.
####################################################


#%%
####################################################
# Entire precedure for word-doc matrix and path entropy and random paths.
####################################################


stops = stopwords.words('english')
words_0stop = [w for w in origianl_words if not w in stops]
words_0stop_stem = [porter.stem(w) for w in words_0stop]
c = Counter(words_0stop_stem)
threshold = 4
words_0stop_stem_threshold = [w for w, count in c.items() if count >= threshold]
final_vocab = words_0stop_stem_threshold
words_stem = [porter.stem(w) for w in origianl_words]

window_length = 200
if len(words_stem) % window_length:
    num_windows = len(words_stem) / window_length + 1
else:
    num_windows = len(words_stem) / window_length

finalwords_inwindows = []
for ind in range(num_windows):
    window_words = words_stem[ind*window_length:(ind+1)*window_length]
    tmp = [w for w in window_words if w in final_vocab]
    finalwords_inwindows.append(tmp)

m = np.zeros([len(final_vocab), num_windows])
for ind_window, finalwords_inwindow in enumerate(finalwords_inwindows):
    c = Counter(finalwords_inwindow)
    for word, count in c.items():
        ind_word = final_vocab.index(word)
        m[ind_word, ind_window] = count

u, s, v = np.linalg.svd(m, full_matrices=False)
window_vectors = np.dot(np.diag(s), v)
window_vectors = np.absolute(window_vectors)

def normalization(v):
    return v / np.sum(v)
for ind in range(window_vectors.shape[1]):
    window_vectors[:, ind] = normalization(window_vectors[:, ind])
    
distances = np.zeros([num_windows, num_windows])
for ind in range(window_vectors.shape[1]-1):
    dis = window_vectors[:, ind]
    for ind1 in range(ind+1, window_vectors.shape[1]):
        dis1 = window_vectors[:, ind1]
        distances[ind, ind1] = jensen_shannon_divergence(dis, dis1)
distances = distances + distances.transpose()

def information(i, j, m):
    # m: a np matrix, i and j: numeric
    row_i = m[i, :]
    pij = row_i[j]
    Iij = - pij * np.log(pij)
    row_i_without_i = row_i[row_i != 0.0]
    Ii = sum(- row_i_without_i * np.log(row_i_without_i))
    return Iij / Ii


path_entropy = .0
for node in range(num_windows-1):
    node1 = node+1
    Iij = information(node, node1, distances)
    path_entropy += Iij


# 20 random paths.
infos = []
for count in range(50):
    
    print count
    
    nodes = range(distances.shape[1])
    path = []
    while len(nodes) != 0:
        next_node = random.sample(nodes, 1)[0]
        path.append(next_node)
        nodes.remove(next_node)
    
    informations = 0.0
    for ind in range(len(path) - 1):
        node = path[ind]
        node1 = path[ind+1]
        Iij = information(node, node1, distances)
        informations += Iij
    infos.append(informations)

#%%
plt.figure(figsize=(8,6))
plt.plot(range(1, 51), infos, marker='o', linewidth=2)
plt.plot(range(1, 51), path_entropy*np.ones(50), 'b--', linewidth=2)
plt.legend(['random path', 'real path'], loc='upper left')
plt.ylabel('path entropy', fontsize=24)
plt.xlabel('simulations', fontsize=24)
plt.tick_params(labelsize=13)


#%%
####################################################
# Entire code for cooccurrence mathod.
####################################################

_, origianl_words = ge.read_file('1813 PRIDE AND PREJUDICE.txt', use_zip=0, zf='none')
origianl_words = origianl_words[:20000]
lenght_original_text = len(origianl_words)

stops = stopwords.words('english')
words_0stop = [w for w in origianl_words if not w in stops]
words_0stop_stem = [porter.stem(w) for w in words_0stop]
c = Counter(words_0stop_stem)
threshold = 4
words_0stop_stem_threshold = [w for w, count in c.items() if count >= threshold]
vocab = words_0stop_stem_threshold
words_stem = [porter.stem(w) for w in origianl_words]
original_vocab = list(set(words_stem))

# Convert into numbers.
words_stem_numeric = np.array([original_vocab.index(w) for w in words_stem])
vocab_numeric = np.array([original_vocab.index(w) for w in vocab])

def count_cooccurrrence(word, word1, window_length, words_numbers):
    indices_v = np.where(words_numbers == word)
    indices_v1 = np.where(words_numbers == word1)
    distances = np.zeros([np.size(indices_v), np.size(indices_v1)])
    for ind in np.arange(np.size(indices_v)):
        index_v = indices_v[0][ind] # indices_v is tuple
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


window_length = 200
m = cooccurrence_m(vocab_numeric, words_stem_numeric, window_length)

c = Counter(words_stem_numeric)
r = np.zeros([vocab_numeric.shape[0], vocab_numeric.shape[0]])
for ind, w in list(enumerate(vocab_numeric))[:-1]:
    for ind1 in range(ind+1, len(vocab_numeric)):
        w1 = vocab_numeric[ind1]
        r[ind, ind1] = float(window_length) * c[w] * c[w1] / lenght_original_text
r = r + r.transpose()

n = m - r
n[n < 0] = 0

u, s, v = np.linalg.svd(n, full_matrices=False)

# After plotting, use the first 200 dimensions as the desired.
word_vectors = np.dot(u[:, :400], np.diag(s[:400]))
word_vectors = np.absolute(word_vectors)

def normalization(v):
    return v / np.sum(v)
    
window_length = 200
if len(words_stem) % window_length:
    num_windows = len(words_stem) / window_length + 1
else:
    num_windows = len(words_stem) / window_length

window_vectors = np.zeros([400, num_windows])
for ind in range(num_windows):
    tmp_words_numeric = words_stem_numeric[ind*window_length:(ind+1)*window_length]
    goodwords = [w for w in tmp_words_numeric if w in vocab_numeric]
    window_vector = np.zeros([400])
    c = Counter(goodwords)
    for w in c:
        ind_vocab = np.where(vocab_numeric == w)[0][0]
        word_vector = word_vectors[ind_vocab, :]
        window_vector += word_vector * c[w]
    window_vector = normalization(window_vector)
    window_vectors[:, ind] = window_vector

# Jensen Shannon divergence
def shannon_entropy(dis):
    dis = [-pi * np.log(pi) for pi in dis]
    return sum(dis)

def jensen_shannon_divergence(dis, dis1):
    term = shannon_entropy(0.5 * dis + 0.5 * dis1)
    term1 = 0.5 * shannon_entropy(dis)
    term2 = 0.5 * shannon_entropy(dis1)
    return term - term1 - term2

distances = np.zeros([num_windows, num_windows])
for ind in range(window_vectors.shape[1]-1):
    dis = window_vectors[:, ind]
    
    print 'calculating the %d text' %ind
    
    for ind1 in range(ind+1, window_vectors.shape[1]):
        dis1 = window_vectors[:, ind1]
        distances[ind, ind1] = jensen_shannon_divergence(dis, dis1)

distances = distances + distances.transpose()

def information(i, j, m):
    # m: a np matrix, i and j: numeric
    row_i = m[i, :]
    pij = row_i[j]
    Iij = - pij * np.log(pij)
    row_i_without_i = row_i[row_i != 0.0]
    Ii = sum(- row_i_without_i * np.log(row_i_without_i))
    return Iij / Ii

path_entropy = .0
for node in range(num_windows-1):
    node1 = node+1
    Iij = information(node, node1, distances)
    path_entropy += Iij
ps = []
for node in range(num_windows-1):
    node1 = node + 1
    Iij = information(node, node1, distances)
    ps.append(Iij)
    
# 20 random paths.
infos = []
for count in range(50):
    
    print count
    
    path = range(distances.shape[1])
    random.shuffle(path)
    
    informations = 0.0
    for ind in range(len(path) - 1):
        node = path[ind]
        node1 = path[ind+1]
        Iij = information(node, node1, distances)
        informations += Iij
    infos.append(informations)

plt.figure(figsize=(8,6))
plt.plot(range(1, 51), infos, marker='o', linewidth=2)
plt.plot(range(1, 51), path_entropy*np.ones(50), 'b--', linewidth=2)
plt.legend(['random path', 'real path'], loc='center left')
plt.ylabel('path entropy', fontsize=24)
plt.xlabel('simulations', fontsize=24)
plt.tick_params(labelsize=13)

print 'path entropy: %f' %path_entropy
print 'mean of random path entropy: %f' %np.mean(infos)

#%%
########################################################
# 3. Build cooccurrence.
########################################################


#%%
def count_cooccurrrence(word, word1, window_length, words_numbers):
    indices_v = np.where(words_numbers == word)
    indices_v1 = np.where(words_numbers == word1)
    distances = np.zeros([np.size(indices_v), np.size(indices_v1)])
    for ind in np.arange(np.size(indices_v)):
        index_v = indices_v[0][ind] # indices_v is tuple
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
# Cooccurrence matrix, with window length.
window_length = 200
m = cooccurrence_m(stem_vocab_numeric, stem_original_words_numeric, window_length)


#%%
# Random cooccurrence matrix.
c = Counter(stem_original_words_numeric)
r = np.zeros([stem_vocab_numeric.shape[0], stem_vocab_numeric.shape[0]])
for ind,w in list(enumerate(stem_vocab_numeric))[:-1]:
    for ind1 in range(ind+1, len(stem_vocab_numeric)):
        w1 = stem_vocab_numeric[ind1]
        r[ind, ind1] = float(window_length) * c[w] * c[w] / lenght_original_text
        
r = r + r.transpose()


#%%
# Normalized cooccurrence matrix.
n = m - r
n[n < 0] = 0


#%%
########################################################
# 4. SVD.
########################################################


#%%
u, s, v = np.linalg.svd(n, full_matrices=False)


#%%
def get_abs(a):
    return np.absolute(a)
    
    
#%%
# After plotting, use the first 200 dimensions as the desired.
word_vectors = np.dot(u[:, :200], np.diag(s[:200]))
word_vectors = get_abs(word_vectors)


#%%
########################################################
# 5. Transform word numeric vectors into new vectors from the svd result.
########################################################


#%%
def normalization(v):
    return v / np.sum(v)
    
    
#%%
length_window = 200
if len(stem_original_words_numeric) % len(stem_vocab_numeric):
    num_windows = len(stem_original_words_numeric) / len(stem_vocab_numeric)
else:
    num_windows = len(stem_original_words_numeric) / len(stem_vocab_numeric) + 1


#%%
window_vectors = np.zeros([num_windows, 200])
for ind in range(num_windows):
    tmp_stem_original_words_numeric = stem_original_words_numeric[length_window * ind: length_window * (ind+1)]
    words = [w for w in tmp_stem_original_words_numeric if w in stem_vocab_numeric]
    window_v = np.zeros([1, 200])
    for w in words:
        ind_vocab = np.where(stem_vocab_numeric == w)[0][0]
        v = word_vectors[ind_vocab-1, :]
        window_v += v
    window_v = normalization(window_v)
    window_vectors[ind, :] = window_v


#%%
########################################################
# 6. Get the Jensen Shannon Divergence for each pair of subtext.
########################################################


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
distances = np.zeros([num_windows, num_windows])
for ind in range(window_vectors.shape[0]-1):
    dis = window_vectors[ind, :]
    
    print 'calculating the %d text' %ind
    
    for ind1 in range(ind+1, window_vectors.shape[0]):
        dis1 = window_vectors[ind1, :]
        distances[ind, ind1] = jensen_shannon_divergence(dis, dis1)

distances = distances + distances.transpose()


#%%
########################################################
# 7. Calculate the path entropy.
########################################################


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
path_entropy = .0
for ind, node in enumerate(range(distances.shape[1]-1)):
    for ind1, node1 in enumerate(range(ind+1, distances.shape[1])):
        Iij = information(node, node1, distances)
        path_entropy += Iij
        
        
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
###################################################################
# Random text model.
###################################################################
r_stem_original_words_numeric = np.copy(stem_original_words_numeric)
np.random.shuffle(r_stem_original_words_numeric)

window_length = 200
m = cooccurrence_m(stem_vocab_numeric, r_stem_original_words_numeric, window_length)

# Random cooccurrence matrix.
c = Counter(r_stem_original_words_numeric)
r = np.zeros([stem_vocab_numeric.shape[0], stem_vocab_numeric.shape[0]])
for ind,w in list(enumerate(stem_vocab_numeric))[:-1]:
    for ind1 in range(ind+1, len(stem_vocab_numeric)):
        w1 = stem_vocab_numeric[ind1]
        r[ind, ind1] = float(window_length) * c[w] * c[w] / lenght_original_text
r = r + r.transpose()

# Normalized cooccurrence matrix.
n = m - r
n[n < 0] = 0

u, s, v = np.linalg.svd(n, full_matrices=False)

# After plotting, use the first 200 dimensions as the desired.
word_vectors = np.dot(u[:, :200], np.diag(s[:200]))
word_vectors = get_abs(word_vectors)

length_window = 200
if len(r_stem_original_words_numeric) % len(stem_vocab_numeric):
    num_windows = len(r_stem_original_words_numeric) / len(stem_vocab_numeric)
else:
    num_windows = len(r_stem_original_words_numeric) / len(stem_vocab_numeric) + 1

window_vectors = np.zeros([num_windows, 200])
for ind in range(num_windows):
    tmp_stem_original_words_numeric = r_stem_original_words_numeric[length_window * ind: length_window * (ind+1)]
    words = [w for w in tmp_stem_original_words_numeric if w in stem_vocab_numeric]
    window_v = np.zeros([1, 200])
    for w in words:
        ind_vocab = np.where(stem_vocab_numeric == w)[0][0]
        v = word_vectors[ind_vocab-1, :]
        window_v += v
    window_v = normalization(window_v)
    window_vectors[ind, :] = window_v
    
distances = np.zeros([num_windows, num_windows])
for ind in range(window_vectors.shape[0]-1):
    dis = window_vectors[ind, :]
    
    print 'calculating the %d text' %ind
    
    for ind1 in range(ind+1, window_vectors.shape[0]):
        dis1 = window_vectors[ind1, :]
        distances[ind, ind1] = jensen_shannon_divergence(dis, dis1)

distances = distances + distances.transpose()

path_entropy = .0
for ind, node in enumerate(range(distances.shape[1]-1)):
    for ind1, node1 in enumerate(range(ind+1, distances.shape[1])):
        Iij = information(node, node1, distances)
        path_entropy += Iij
#%%
###################################################################
# Random text model, word-doc matrix.
###################################################################

r_stem_original_words_numeric = np.copy(stem_original_words_numeric)
np.random.shuffle(r_stem_original_words_numeric)

window_length = 200
if len(stem_original_words_numeric) % window_length:
    num_windows = len(stem_original_words_numeric) / window_length + 1
else:
    num_windows = len(stem_original_words_numeric) / window_length

m = np.zeros([stem_vocab_numeric.shape[0], num_windows])
for ind in range(num_windows):
    tmp = r_stem_original_words_numeric[ind*window_length:(ind+1)*window_length]
#    intersection = np.intersect1d(stem_vocab_numeric, tmp)
    intersection = [w for w in tmp if w in stem_vocab_numeric]
    c = Counter(intersection)
    for w, count in c.items():
        ind_w = np.where(stem_vocab_numeric == w)[0][0]
        m[:, ind][ind_w] = count


u, s, v = np.linalg.svd(m, full_matrices=False)

# Use the first 200 dimensions as the desired.
para_vectors = np.dot(np.diag(s[:100]), v[:100, :])
para_vectors = np.absolute(para_vectors)

subtext_vectors = np.copy(para_vectors)
for ind in range(subtext_vectors.shape[1]):
    subtext_vectors[:, ind] = normalization(subtext_vectors[:, ind])


distances = np.zeros([num_windows, num_windows])
for ind in range(subtext_vectors.shape[1]-1):
    dis = subtext_vectors[:, ind]
    for ind1 in range(ind+1, subtext_vectors.shape[1]):
        dis1 = subtext_vectors[:, ind1]
        distances[ind, ind1] = jensen_shannon_divergence(dis, dis1)
distances = distances + distances.transpose()

path_entropy = .0
for ind, node in enumerate(range(distances.shape[1])):
    for ind1, node1 in enumerate(range(ind+1, distances.shape[1])):
        Iij = information(node, node1, distances)
        path_entropy += Iij

#%%




















