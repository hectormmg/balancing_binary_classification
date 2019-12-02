#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  2 09:40:29 2019

@original_author: German Villacorta
@author: Héctor Morales

"""

from helper_functions import *
import sys
import numpy as np
np.seterr(divide='ignore', invalid='ignore')
from sklearn import svm
import pandas as pd
import random
from sklearn.model_selection import train_test_split

from fitness import evaluate
from fitness_statistics import FitnessStatistics
from data_processing import *
from species import Species
from species_factory import SpeciesFactory




def crossover(individual1, individual2):
    
    uniform = np.random.uniform(0,1,len(individual1))
    random_array = np.where(uniform > 0.5, 1, 0)
    
    new_individual = np.zeros(len(individual1))
    
    new_individual = np.where(random_array == 1,new_individual, individual1 )
    new_individual = np.where(random_array == 0,new_individual, individual2 )
    
    return new_individual

def mutation(individual, prob_mutation):
    prob = random.uniform(0,1)
    
    if prob < 0.5:
        randA = random.randint(0,len(individual) - 1)
        randB = random.randint(0,len(individual) - 1)
    
        individual[randA], individual[randB] = individual[randB], individual[randA]
    else:
        individual = np.where(random.uniform(0,1) < prob_mutation,
                              individual,
                              random.randint(0,len(individual) - 1))
    
    return individual



def binaryTournament(population, fitnesses, indexP1, indexP2):
    
    fitnessA = fitnesses[indexP1]
    fitnessB = fitnesses[indexP2]
    
    if(fitnessA > fitnessB):
        return population[indexP1], indexP1
    
    return population[indexP2], indexP2

def pick_parents(population, species_fitnesses):
    #Para que los que tengan mayor fitness tengan mas probabilidad de ser seleccionados.    
    parents = np.random.randint(0, len(population), 4)
   
    return parents[0], parents[1], parents[2], parents[3]

def train(X, Y, classifier, metric):
    
    # Configure hyperparameters
    N = 100
    iterations = 100
    prob_mutation = 0.05
    
    # Split dataset by class
    positive_class, negative_class = split_dataset_by_class(X, Y, classifier != "G")
    positive_count = positive_class.shape[0]
    negative_count = negative_class.shape[0]
    example_count = positive_count + negative_count
    print(f'Initial Positive Class Count: {positive_count} examples ({round(positive_count/example_count * 100, 2)}%)')
    print(f'Initial Negative Class Count: {negative_count} examples ({round(negative_count/example_count * 100, 2)}%)\n')
    
    # Instantiate Fitness Statistics for reporting
    fitness_statistics = FitnessStatistics()
    
    # Flag determines whether positive class is minority
    minority_flag = len(positive_class) < len(negative_class)
    
    # Initialize positive and negative "species"
    species = SpeciesFactory.initialize_species(X, Y, positive_class, negative_class, minority_flag)
    species_X = [positive_class, negative_class]
    species_Y = [np.full((len(positive_class),), 1), np.full((len(negative_class),), 0)]
    species_Y_value = [1, 0]
    
    # Outer training loop
    current_iteration = 1
    print(f'Starting iteration {current_iteration} out of {iterations}')
    while current_iteration <= iterations:
        if (current_iteration%10 == 0):
            print(f'Starting iteration {current_iteration} out of {iterations}')
        # Loop between species (Positive/Negative)
        for i in range(len(species)):
            # Select current population
            currentS = species[i]
            currentS_fitness = currentS.population_fitnesses
            
            # Check if current species is minority
            minority_flag = len(species_X[i]) < len(species_X[not i])
            
            # Create random index
            random_index_m = random.randint(0, N-1 )

            m_dataX, m_dataY = prepare_data_for_training(X, species_Y[not i], 
                                                         species_X[not i],
                                                         None,
                                                         species[not i].population[random_index_m],
                                                         (not minority_flag),
                                                         species_Y_value[not i],
                                                         False)
            
            for j in range(0, N):
                # Selecciona dos padres.
                indexP1, indexP2, indexP3, indexP4 = pick_parents(currentS.population, currentS_fitness)
                
                parentA, indexA = binaryTournament(currentS.population, currentS_fitness, indexP1, indexP2)
                parentB, indexB = binaryTournament(currentS.population, currentS_fitness, indexP3, indexP4)
                
                # Crea un offspring.
                child = crossover(parentA, parentB)
                child = mutation(child, prob_mutation)
                child_dataX, child_dataY = prepare_data_for_training(X, species_Y[i], 
                                                                     species_X[i],
                                                                     None,
                                                                     child,
                                                                     minority_flag,
                                                                     species_Y_value[i],
                                                                     False)
                
                # Juntamos data de ambas soluciones, nuevo hijo y indice m.
                # Entrenamos con data de soluciones y probamos con dataset completo.
                child_dataX = np.append(child_dataX, m_dataX, 0)
                child_dataY = np.append(child_dataY, m_dataY, 0)
                child_fitness = evaluate(child_dataX, child_dataY, X, Y, classifier, metric)

                # En caso de que el fitness del hijo sea mejor que alguno de los padres
                # entonces quita ese padre de la poblacion y pon al hijo en su 
                # posicion.
                if child_fitness >= currentS_fitness[indexA] and child_fitness >= currentS_fitness[indexB]:
                    if currentS_fitness[indexA] < currentS_fitness[indexB]:
                        minIndex = indexA
                    else:
                        minIndex = indexB
                        
                    currentS.population[minIndex] = child
                    currentS_fitness[minIndex] = child_fitness
                    
                elif child_fitness >= currentS_fitness[indexA]:
                    currentS.population[indexA] = child
                    currentS_fitness[indexA] = child_fitness
                elif child_fitness >= currentS_fitness[indexB]:
                    currentS.population[indexB] = child
                    currentS_fitness[indexB] = child_fitness
                    
            
            species[i].population_fitnesses = currentS_fitness
            species[i] = currentS
            
        current_iteration += 1
        fitness_statistics.max_positive.append(np.max(species[0].population_fitnesses))
        fitness_statistics.max_negative.append(np.max(species[1].population_fitnesses))
        fitness_statistics.avg_positive.append(np.mean(species[0].population_fitnesses))
        fitness_statistics.avg_negative.append(np.mean(species[1].population_fitnesses))

            
                
    positive_class_solution = species[0].population[np.argmax(species[0].population_fitnesses)]
    negative_class_solution = species[1].population[np.argmax(species[1].population_fitnesses)]
    

    # Regresamos el dataset de la solucion.
    X1, Y1 = prepare_data_for_training(X, species_Y[0], 
                                         species_X[0],
                                         None,
                                         positive_class_solution,
                                         len(positive_class) < len(negative_class),
                                         species_Y_value[0],
                                         False)
    
    X2, Y2 = prepare_data_for_training(X, species_Y[1], 
                                         species_X[1],
                                         None,
                                         negative_class_solution,
                                         len(negative_class) < len(positive_class),
                                         species_Y_value[1],
                                         False)
    
    X_new = np.append(X1, X2, 0)
    Y_new = np.append(Y1, Y2, 0)
    
    best_fitness = evaluate(X_new, Y_new, X, Y, classifier, metric)
    shapes = {"p_orig": positive_class.shape[0],
              "n_orig": negative_class.shape[0],
              "p_new": X1.shape[0],
              "n_new": X2.shape[0]}
    
    print("\nOriginal size: ", X.shape, "Balanced size: ", X_new.shape)
    print("Original Positive Count: ",
          shapes["p_orig"],
          f'examples ({round((positive_class.shape[0]/X.shape[0])*100, 2)}%)',
          "\nBalanced Positive Count: ",
          shapes["p_new"],
          f'examples ({round((X1.shape[0]/X_new.shape[0])*100, 2)}%)')
    print("Original Negative Count: ",
          shapes["n_orig"],
          f'examples ({round((negative_class.shape[0]/X.shape[0])*100, 2)}%)',
          "\nBalanced Negative Count: ",
          shapes["n_new"],
          f'examples ({round((X2.shape[0]/X_new.shape[0])*100, 2)}%)')
    print("Balanced Dataset Fitness: ", round(best_fitness, 2))
    
    fitness_statistics.plot_all()
    
    return X_new, Y_new, best_fitness, shapes
