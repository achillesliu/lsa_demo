# -*- coding: utf-8 -*-

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
def information(i, j, m):
    # Calculate the information needed given you are
    # on node i to node j.
    # The graph matrix is m.
    # m: a np matrix, i and j: numeric
    row_i = m[i, :]
    pij = row_i[j]
    Iij = - pij * np.log(pij)
    row_i_without_i = row_i[row_i != 0.0]
    Ii = sum(- row_i_without_i * np.log(row_i_without_i))
    return Iij / Ii

def fitness_get_path_entropy(mtrx, path):
    # Bo, 160616
    # The program is altered since now I have calculated all
    # the information entropy.
    # Now this function can read directly from the information
    # entropy matrix.
    # This saves time for the evolution process.

    path_entropy = .0
    for ind in range(len(path) - 1):
        node = path[ind]
        node1 = path[ind + 1]
        Iij = mtrx[node, node1]
#        Iij = information(node, node1, mtrx)
        path_entropy += Iij
    return path_entropy

def get_individual(length_path):
    # Get the random paths.
    # Nodes in the paths are integers.
    path = range(length_path)
    random.shuffle(path)
    return path
    
def get_population(num_indis, length_path):
    return [get_individual(length_path) for i in range(num_indis)]

def get_fitness_list(pop, m):
    # Gets the score for all individuals in the population.
    # Returns the sorted list with the most fitted individual foremost.
    graded = [ (fitness_get_path_entropy(m, path), path) for path in pop]
    return [x[1] for x in sorted(graded, key=itemgetter(0))]

def get_average_fitness(pop, m):
    grades = np.array([ fitness_get_path_entropy(m, path) for path in pop])
    return np.mean(grades)
    
def crossover_ox(father, mother):
    point1, point2 = sorted(random.sample(range(len(father)), 2))
    slice_father = father[point1:point2]
    slice_mother = mother[point1:point2]
    child_father = remove_dup(slice_mother, father)
    child_mother = remove_dup(slice_father, mother)
    return point1, point2, child_father, child_mother

def remove_dup(slc, parent):
    diff = np.array([x for x in parent if x not in slc])
    return np.concatenate((slc, diff))

def mutation_change(parent):
    point1, point2 = sorted(random.sample(range(len(parent)), 2))
    parent[point1], parent[point2] = parent[point2], parent[point1]
    return parent
    
def evolve(pop, m, retain_prob=.2, random_select_prob=.05, mutate_prob=.01):
    
    # Retains some as the parents.
    graded = get_fitness_list(pop, m)
    best_current = graded[0]
    retain_length = int(len(graded) * retain_prob)
    parents = graded[:retain_length]
    
    # Add low preformance indi for diversity.
    for indi in graded[retain_length:]:
        if random_select_prob > random.random():
            parents.append(indi)
    
    # Crossover.
    parents_length = len(parents)
    desired_length = len(pop) - parents_length
    children = []
    while len(children) < desired_length:
        father_ind = random.randint(0, parents_length - 1)
        mother_ind = random.randint(0, parents_length - 1)
        if father_ind != mother_ind:
            father = parents[father_ind]
            mother = parents[mother_ind]
            _, _, child1, child2 = crossover_ox(father, mother)
            children.append(child1)
            children.append(child2)
    if len(children) > desired_length:
        children.pop()
    parents.extend(children)
    
    # Mutation
    for indi_ind in range(len(parents)):
        if mutate_prob > random.random():
            parents[indi_ind] = mutation_change(parents[indi_ind])
    
    return best_current, parents

#%% test
informations = pickle.load(open('informations.p'))

pop_size = 100
pop = get_population(pop_size, informations.shape[1])
fitness_history = [get_average_fitness(pop, informations), ]
best_indi_history = [pop[0], ]
for i in xrange(50000):
    
#    print '%d generation...' %i
    best_current, pop = evolve(pop, informations)
    fitness_history.append(get_average_fitness(pop, informations))
    if i % 100 == 0:
        best_indi_history.append(best_current)
