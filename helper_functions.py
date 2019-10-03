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
import matplotlib.pyplot as plt
from sklearn import preprocessing
import matplotlib


## Funcion que separa el dataset por sus clases y regresa dos datasets separados.
def separate_by_class(X, Y, normalize):
    positive_class = []
    negative_class = []
    
    if(normalize):
        X = X / X.max(axis = 0)
    
    for i in range(0,len(X)):
        if(Y[i] == 1 or Y[i] == "positive" or Y[i] == "yes"):
            positive_class.append(X[i])
        else:
            negative_class.append(X[i])
        
    return np.array(positive_class), np.array(negative_class)
        


# Funcion que crea las conexiones de los nodos (sample)
# regresa un arreglo con las relaciones y un grafo en donde el indice del dict
# es el indice del sample y su valor es un arreglo con el indice de todos los 
# nodos a los que esta conectado. 
    
def create_random_individual(size):
    individual = np.zeros((1,size))
    
    for i in range(0,size):
        individual[0][i] = random.randint(0,size - 1)
                
    return individual
        
def create_graph_from_individual(individual):
    graph = {}
    
    for i in range(0, len(individual)):
        if i in graph and (individual[i] not in graph[i]) :
            graph[i].append(individual[i])
        else:
            graph[i] = [int(individual[i])]
        
        
        if individual[i] in graph and (i not in graph[individual[i]]) :
            graph[individual[i]].append(i)
        else:
            graph[individual[i]] = [int(i)]
            
    return graph 
        
# Funcion que crea los grupos cuando se inicializa la poblacion.
def initialize_groups(individual):
    r = random.uniform(0,1)
    number_of_examples = len(individual)
    num_groups = 0
    number_of_groups = int(r * number_of_examples) + 1
    groups = [[] for i in range(number_of_groups)]
    
    for example in individual:
        random_group = random.randint(0, number_of_groups - 1)
        
        groups[random_group].append(example)
        
    # Aqui "hacemos al individuo" solo lo ponemos a como estan los grupos.
    for i in range(0, number_of_groups):
        for j in range(0, len(groups[i]) - 1):
            cur = int(groups[i][j])
            nxt = int(groups[i][j + 1])
            individual[cur] = nxt
        if len(groups[i]) > 0:
            individual[int(groups[i][len(groups[i]) - 1])] = int(groups[i][len(groups[i]) - 1])

    return individual, np.array(groups)
    
# Funcion que crea un prototipo, le pasas el dataset y los indices.
def create_prototype(data, indexes):
    ## Aqui aplicamos los operadores, por mientras solo pondre un promedio. 
    prototype = np.zeros((1, data[0].size))
    
    #prototype = np.mean(data[indexes])
    for i in indexes:
        prototype = prototype + data[int(i),:] 
        
    prototype = prototype / len(indexes)
    
    return prototype
    


def shuffle_individual(individual, divisions):
    N = len(individual)
    individual = np.reshape(individual, (int(len(individual) / divisions) ,divisions))
        
    np.random.shuffle(individual)

    individual = np.reshape(individual, (N))
    return individual
    
    

#Funcion para ayudar con la recursividad de graph traversal
# Va poniendo los nodos que ya vio como True en arreglo de "visited"
# Agrega a arreglo group cada vector que este conectado.
def dfs_util(v, visited, graph, group):
    visited[v] = True
    group.append(v)
    
    for node in graph[v]:
        if(not visited[int(node)]):
            dfs_util(int(node), visited, graph, group)
    

# Funcion que pasa por todos los nodos del grafo.
# Al terminar de guardar todos los nodos conectados de un "grupo"
# pone sus indices en un arreglo y ese arreglo en la matriz de componentes conectados.
def graph_traversal(graph):
    visited = [False] * len(graph)
    connected_components = []
    
    for i in range(0, len(graph)):
        if(not visited[i]):
            group = []
            dfs_util(int(i), visited, graph, group)
            connected_components.append(group)
        
    
    return connected_components
           
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

