#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 14:14:34 2019

@author: HectorMg
"""
import sys

# Funcion que crea las conexiones de los nodos (sample)
# regresa un arreglo con las relaciones y un grafo en donde el indice del dict
# es el indice del sample y su valor es un arreglo con el indice de todos los 
# nodos a los que esta conectado. 
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

#Funcion para ayudar con la recursividad de graph traversal
# Va poniendo los nodos que ya vio como True en arreglo de "visited"
# Agrega a arreglo group cada vector que este conectado.
def dfs_util(v, visited, graph, group):
    visited[v] = True
    group.append(v)
    
    for node in graph[v]:
        if(not visited[int(node)]):
            dfs_util(int(node), visited, graph, group)