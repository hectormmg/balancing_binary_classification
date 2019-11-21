#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 23 10:40:20 2019

@author: German
"""

from sklearn.datasets import load_breast_cancer
import numpy as np
import seaborn as sns
import random
from sklearn import preprocessing
    


def shuffle_individual(individual, divisions):
    N = len(individual)
    individual = np.reshape(individual, (int(len(individual) / divisions) ,divisions))
        
    np.random.shuffle(individual)

    individual = np.reshape(individual, (N))
    return individual
    
           
# Funcion que hace oversampling para minority class y undersampling para majority class.
def preprocess_imbalance(dataset):
    
    positive_class, negative_class = separate_by_class(dataset)
    
    # Para ambas clases sacamos un arreglo en donde los indices son nodos y el valor
    # es al nodo al que estan conectados. 
    
    positive_individual, positive_graph = create_random_individual(len(positive_class))
    negative_individual, negative_graph = create_random_individual(len(negative_class))

    positive_groups = graph_traversal(positive_graph)
    negative_groups = graph_traversal(negative_graph)
    
    
    positive_prototypes = np.zeros((len(positive_groups), len(positive_class[0])))
    negative_prototypes = np.zeros((len(negative_groups), len(negative_class[0])))

    # Una vez que tengamos todos los indices de cada grupo, haremos los nuevos 
    # prototipos. 
    for i in range(0,len(positive_groups)):
        prototype = create_prototype(positive_class, positive_groups[i])
        positive_prototypes[i, :] = prototype
    
    for i in range(0,len(negative_groups)):
        prototype = create_prototype(negative_class, negative_groups[i])
        negative_prototypes[i, :] = prototype
        
    if(len(positive_class) > len(positive_class) ):
        positive_class = positive_prototypes
        negative_class = np.append(negative_class , negative_prototypes, axis = 0)
    else:
        negative_class = negative_prototypes
        positive_class = np.append(positive_class , positive_prototypes, axis = 0)
        
    positive_Y = np.ones((len(positive_class), 1))
    negative_Y = np.zeros((len(negative_class), 1))

    ### Esto lo pongo por mientras para que podamos ver como jalo.
#    fig = plt.figure()
#    ax1 = fig.add_subplot(111)
#    ax2 = fig.add_subplot(111)
#    
#    ax1.scatter(x = positive_class[:,2], y = positive_class[:,8], c='blue')
#    ax1.scatter(x = positive_prototypes[:,2], y = positive_prototypes[:,8], c='green')
#    
#    ax2.scatter(x = negative_class[:,2], y = negative_class[:,8], c='red')
#    ax2.scatter(x = negative_prototypes[:,2], y = negative_prototypes[:,8], c='orange')
#    
#    plt.show()
    ### 
    
    
    return positive_class, negative_class, positive_Y, negative_Y 
        
    
## Test de las clases para ver como funcionan. 
# Plot everything
#colors = ['red', 'black']
#plt.scatter(x = data.data[:,2], y = data.data[:,8], c = data.target, cmap=matplotlib.colors.ListedColormap(colors))
#plt.show()

