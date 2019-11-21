#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 11:50:48 2019

@author: HectorMg
"""

import random
import numpy as np
import sys

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
    for i in range(number_of_groups):
        for j in range(0, len(groups[i]) - 1):
            cur = int(groups[i][j])
            nxt = int(groups[i][j + 1])
        if len(groups[i]) > 0:
            individual[int(groups[i][len(groups[i]) - 1])] = int(groups[i][len(groups[i]) - 1])
    return individual, np.array(groups)