#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  2 09:40:29 2019

@author: German
"""

from helper_functions import *
import sys
import numpy as np
np.seterr(divide='ignore', invalid='ignore')
from sklearn import svm
import pandas as pd
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import f1_score



def get_fitness(X_train, y_train, X_test, y_test, classifier, metric, originalN):
    
    lenP = np.count_nonzero(y_train)
    lenN = len(y_train) - lenP
    
    print("LenP: ", lenP, ", LenN: ", lenN, ", Total: ", len(y_train))
    
    if(classifier == "SVM"):
        clf = svm.SVC(gamma="auto")
    elif (classifier == "G"):
        clf = GaussianNB()

    clf.fit(X_train, y_train)
    prediction = clf.predict(X_test)

    tn, fp, fn, tp = confusion_matrix(y_test, prediction).ravel()
    
    if (metric == "accuracy"):
        accuracy_negative = tp / (tp + tn)
        accuracy_positive = fp / (fp + fn)

        fitness = 0.5 * (accuracy_negative + accuracy_positive)
    elif (metric == "kappa"):
        fitness = cohen_kappa_score(y_test,prediction)
    elif (metric == "auc"):
        fitness = roc_auc_score(y_test, prediction)
    elif (metric == "f1"):
        fitness = f1_score(y_test, prediction, average='macro')
        
    if np.isnan(fitness):
        fitness = 0.0
        
    return fitness - (0.1 * max(lenP,lenN) / min(lenP,lenN))

    
def initialize_population(N, M):
    #Pon random individuals en un grupo.
    population = np.zeros((N, M))
    for i in range(0, N):
        population[i, :] = create_random_individual(M)
        #shuffle_individual(population[i, :], int( N / 10))
#        # Prueba para que se hagan mas grupos
#        rand = random.randint(0,(N-1)/2)
#        indexes = np.random.randint(N, rand)
        
#        individual = np.where()
    return population


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
                              random.randint(0,len(individual) - 1)
                              )
    
    return individual


def prepare_data_for_training(X, Y, X_class, X_opposite_class, individual, is_minority, Y_value, initialization):
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
    if (is_minority):
        X = np.append(X_class, prototypes, axis = 0)
        Y_prototypes = np.full((len(prototypes),), Y_value)
        Y = np.append(Y, Y_prototypes, axis = 0)
        Y = np.transpose(Y)
    else:
        if X_opposite_class is not None:
            X = np.append(X_opposite_class, prototypes, axis = 0)
            Y_opposite = np.full(len(X_opposite_class), not Y_value)
            Y_prototypes = np.full(len(prototypes), Y_value)
            Y = np.append(Y_opposite, Y_prototypes)
        else:
            X = prototypes
            Y = np.full(len(prototypes), Y_value)
            
    return X, Y

    

def initialize_specie(X, Y, X_class, XP_class, Y_value, N, is_minority):
    
    population = initialize_population(N, len(X_class))
    population_fitnesses = np.zeros((N, 1))
    originalN = len(Y)
    for i in range(0,N):
        Xf, Yf = prepare_data_for_training(X, Y, X, XP_class, population[i], is_minority, Y_value, True)

        fitness = get_fitness(Xf, Yf, X, Y, "SVM", "kappa", originalN)
        population_fitnesses[i] = fitness

    return population, population_fitnesses


def binaryTournament(population, fitnesses, indexP1, indexP2):
    
    fitnessA = fitnesses[indexP1]
    fitnessB = fitnesses[indexP2]
    
    if(fitnessA > fitnessB):
        return population[indexP1], indexP1
    
    return population[indexP2], indexP2

def pick_parents(population, fitnesses):
    #Para que los que tengan mayor fitness tengan mas probabilidad de ser seleccionados.    
    parents= np.random.randint(0, len(population), 4)
   
    return parents[0], parents[1], parents[2], parents[3]

def main():
    import sys 
    sys.argv = [0,0,0,0,0,0]
    sys.argv[1] = "SVM"
    sys.argv[2] = "kappa"
    sys.argv[3] = "datasets/preprocessed/iris.csv"
    
    classifier =  sys.argv[1]  
    metric = sys.argv[2]       
    data_csv = pd.read_csv(sys.argv[3])              
    data_csv["Class"] = np.where(data_csv["Class"] == "yes", 1, 0)
    X = np.array(data_csv.iloc[:, 0:-1])
    Y = np.array(data_csv.iloc[:, -1])
    print(Y)
    N = 100
    iterations = 100
    prob_mutation = 0.05
    original_N = len(Y)
    positive_class, negative_class = separate_by_class(X, Y, classifier != "G")
    max_positive_fitness = []
    max_negative_fitness = []
    avg_positive_fitness = []
    avg_negative_fitness = []
    
    
    positive_P, positive_P_fitness = initialize_specie(X, Y, 
                                                       positive_class, negative_class, 1, 100, 
                                                       len(positive_class) < len(negative_class))
    
    negative_P, negative_P_fitness = initialize_specie(X, Y, 
                                                       negative_class, positive_class, 0, 100, 
                                                       len(negative_class) < len(positive_class))
    
    species = [positive_P, negative_P]
    species_fitnesses = [positive_P_fitness, negative_P_fitness]
    species_X = [positive_class, negative_class]
    species_Y = [np.full((len(positive_class),), 1), np.full((len(negative_class),), 0)]
    species_Y_value = [1, 0]
    
    while iterations:

        for i in range(0, 2):
            
            currentP = species[i]
            currentP_fitness = species_fitnesses[i]
            is_minority = len(species_X[i]) < len(species_X[not i])
            
            random_index_m = random.randint(0, N-1 )
            
            m_dataX, m_dataY = prepare_data_for_training(X, species_Y[not i], 
                                                         species_X[not i],
                                                         None,
                                                         species[not i][random_index_m],
                                                         (not is_minority),
                                                         species_Y_value[not i],
                                                         False)
            for j in range(0, N):
                # Selecciona dos padres.
                indexP1, indexP2, indexP3, indexP4 = pick_parents(currentP, currentP_fitness)
                
                parentA, indexA = binaryTournament(currentP, currentP_fitness, indexP1, indexP2)
                parentB, indexB = binaryTournament(currentP, currentP_fitness, indexP3, indexP4)
                
                # Crea un offspring.
                child = crossover(parentA, parentB)
                child = mutation(child, prob_mutation)
                child_dataX, child_dataY = prepare_data_for_training(X, species_Y[i], 
                                                                     species_X[i],
                                                                     None,
                                                                     child,
                                                                     is_minority,
                                                                     species_Y_value[i],
                                                                     False)
                
                # Juntamos data de ambas soluciones, nuevo hijo y indice m.
                # Entrenamos con data de soluciones y probamos con dataset completo.
                child_dataX = np.append(child_dataX, m_dataX, 0)
                child_dataY = np.append(child_dataY, m_dataY, 0)
                
                child_fitness = get_fitness(child_dataX, child_dataY, X, Y, classifier, metric, original_N)

                # En caso de que el fitness del hijo sea mejor que alguno de los padres
                # entonces quita ese padre de la poblacion y pon al hijo en su 
                # posicion.
                if child_fitness >= currentP_fitness[indexA] and child_fitness >= currentP_fitness[indexB]:
                    if currentP_fitness[indexA] < currentP_fitness[indexB]:
                        minIndex = indexA
                    else:
                        minIndex = indexB
                        
                    currentP[minIndex] = child
                    currentP_fitness[minIndex] = child_fitness
                    
                elif child_fitness >= currentP_fitness[indexA]:
                    currentP[indexA] = child
                    currentP_fitness[indexA] = child_fitness
                elif child_fitness >= currentP_fitness[indexB]:
                    currentP[indexB] = child
                    currentP_fitness[indexB] = child_fitness
                    
            
            species_fitnesses[i] = currentP_fitness
            species[i] = currentP
            
        iterations = iterations - 1
        max_positive_fitness.append(np.max(species_fitnesses[0]))
        max_negative_fitness.append(np.max(species_fitnesses[1]))
        avg_positive_fitness.append(np.mean(species_fitnesses[0]))
        avg_negative_fitness.append(np.mean(species_fitnesses[1]))

            
                
    positive_class_solution = species[0][np.argmax(species_fitnesses[0])]
    negative_class_solution = species[1][np.argmax(species_fitnesses[1])]
    

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
    
    best_fitness = get_fitness(X_new, Y_new, X, Y, classifier, metric, original_N)
    print("+ Era: ", positive_class.shape, "Es: ", X1.shape)
    print("- Era: ", negative_class.shape, "Es: ", X2.shape)

    print("Total Era: ", X.shape, "Es: ", X_new.shape)
    print("fitness: ", best_fitness)
    
     ### Esto lo pongo por mientras para que podamos ver como jalo.
    plt.plot(max_negative_fitness, label = "Negative fitness")
    plt.plot(max_positive_fitness, label = "Positive fitness")
    plt.plot(avg_positive_fitness, label = "Average Positive fitness")
    plt.plot(avg_negative_fitness, label = "Average Negative fitness")
    plt.legend()
    
    return X_new, Y_new


        

fullx, fy = main()
fy = np.reshape(fy, (len(fy), 1))
print(fullx.shape, fy.shape)
new_dataset = np.append(fullx, fy, axis=1)
pd.DataFrame(new_dataset).to_csv(sys.argv[3] + "_balanced.csv")
      





