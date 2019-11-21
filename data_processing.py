#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 20 19:37:30 2019

@author: HectorMg
"""

import numpy as np
from population_utilities import initialize_groups
from graph_utilities import *

def normalize_feature_set(X):
    return X / X.max(axis = 0)

def positive(ex_class):
    return ex_class == 1 or ex_class == "positive" or ex_class == "yes"

def split_dataset_by_class(X, Y, normalization_flag):
    if normalization_flag: X = normalize_feature_set(X)
    
    positive_examples = list()
    negative_examples = list()
    
    for i in range(len(X)):
        if positive(Y[i]):
            positive_examples.append(X[i])
        else:
            negative_examples.append(X[i])
    return np.array(positive_examples), np.array(negative_examples)

# Creates a prototype from the average feature values of a group
def create_prototype(data, indices):
    ## THIS SHOULD USE EVOLUTIONARY OPERATORS 
    prototype = np.zeros((1, data[0].size))
    
    ## Takes examples that match the group indices
    group_examples = np.take(data, indices, axis=0)
    
    ## Get the mean of each attribute considering all examples in group to create prototype example
    prototype = np.mean(group_examples, axis=0, keepdims=True)
    return prototype


def prepare_data_for_training(X, Y, class_examples, complement_examples, individual, minority_flag, species_label, initialization):
    #Crea los grupos. 
    if(initialization):
        individual, groups = initialize_groups(individual)
        graph = create_graph_from_individual(individual)
        groups = graph_traversal(graph)
    else:
        graph = create_graph_from_individual(individual)
        groups = graph_traversal(graph)
    prototypes = np.zeros((len(groups), len(X[0])))
    # Una vez que tengamos todos los indices de cada grupo, haremos los nuevos 
    # prototipos. 
    for j in range(0,len(groups)):
        if (len(groups[j]) > 0):
            prototype = create_prototype(X, groups[j])
            prototypes[j, :] = prototype  
        
    #Agrego nuevos prototypos.
    if (minority_flag):
        X = np.append(class_examples, prototypes, axis=0)
        Y_prototypes = np.full((len(prototypes),), species_label)
        Y = np.append(Y, Y_prototypes, axis=0)
        Y = np.transpose(Y)
    else:
        if complement_examples is not None:
            X = np.append(complement_examples, prototypes, axis = 0)
            complement_classes = np.full(len(complement_examples), not species_label)
            prototype_classes = np.full(len(prototypes), species_label)
            Y = np.append(complement_classes, prototype_classes)
        else:
            X = prototypes
            Y = np.full(len(prototypes), species_label)
    return X, Y