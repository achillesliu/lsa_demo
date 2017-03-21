# -*- coding: utf-8 -*-
"""
Created on Fri Jun 24 16:34:26 2016

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
import powerlaw

#%%
fps = ge.fp_list_dir('Results/Semantic distances networks Dickens', 'edgelist')
fp = 'Results/Semantic distances networks Dickens/hard times.edgelist'
g = nx.read_weighted_edgelist(fp)
assortativity = nx.degree_assortativity_coefficient(g)
degrees = nx.degree(g).values()
powerlaw.plot_pdf(degrees)
