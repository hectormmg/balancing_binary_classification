#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 20 22:20:27 2019

@author: HectorMg
"""
import random
import numpy as np
from fitness import evaluate
from data_processing import prepare_data_for_training


class Species:
    def __init__(self, X, Y, class_examples, complement_examples, species_label, N, minority_flag):
        self.initialize_population(N, len(class_examples))
        self.population_fitnesses = np.zeros((N, 1))
        for i in range(0,N):
            Xf, Yf = prepare_data_for_training(X, Y, X, complement_examples, self.population[i], minority_flag, species_label, True)
            fitness = evaluate(Xf, Yf, X, Y, "SVM", "kappa")
            self.population_fitnesses[i] = fitness
    
    def initialize_population(self, N, M):
        population = np.zeros((N, M))
        for i in range(0, N):
            population[i, :] = self.create_random_individual(M)
        self.population = population
    
    def create_random_individual(self, size):
        individual = np.zeros((1,size))
        for i in range(size):
            individual[0][i] = random.randint(0,size - 1)
        return individual
    
            