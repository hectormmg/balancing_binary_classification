#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 18:58:39 2019

@author: HectorMg
"""

import sys
import pandas as pd
import numpy as np

from genetic_algorithm import train


#########################################
#                                       # 
#          Execution Options            #
#                                       #
#########################################

CLASSIFIERS = ["SVM", "G"]
METRICS = ["accuracy", "kappa", "auc", "f1"]
DATASETS = ["../datasets/preprocessed/iris.csv"]#,
            #"../datasets/preprocessed/abalone19.csv"]


def encode_csv_dataset(data_csv):
    test_class = data_csv["Class"][0]
    if(test_class == "yes" or test_class == "no" ):
        data_csv["Class"] = np.where(data_csv["Class"] == "yes", 1, 0)
    return data_csv

def extract_data(encoded_dataset):
    X = np.array(encoded_dataset.iloc[:, 0:-1])
    Y = np.array(encoded_dataset.iloc[:, -1])
    return X, Y
           


def execute_model(X, Y, classifier, metric, dataset_index):
    _, _, balanced_fitness, shapes = train(X, Y, classifier, metric)
    
    return [DATASETS[dataset_index],
            classifier,
            metric,
            round(balanced_fitness, 4),
            shapes["p_orig"],
            shapes["n_orig"],
            f'{round(abs(shapes["p_orig"] - shapes["n_orig"])/(shapes["p_orig"] + shapes["n_orig"]) * 100, 2)}%',
            shapes["p_new"],
            shapes["n_new"],
            f'{round(abs(shapes["p_new"] - shapes["n_new"])/(shapes["p_new"] + shapes["n_new"]) * 100, 2)}%']


#########################################
#                                       # 
#            Main Script                #
#                                       #
#########################################
    
results = list()

# Iterate datasets
for dataset_index in range(len(DATASETS)):
    data_csv = pd.read_csv(DATASETS[dataset_index])              
    encoded_dataset = encode_csv_dataset(data_csv)
    X, Y = extract_data(encoded_dataset)
    for classifier_index in range(len(CLASSIFIERS)):
        classifier = CLASSIFIERS[classifier_index]
        for metric_index in range(len(METRICS)):
            metric = METRICS[metric_index]
            result = execute_model(X, Y, classifier, metric, dataset_index)
            results.append(result)

results_df = pd.DataFrame(results, columns=["Dataset",
                                            "Classifier",
                                            "Fitness Metric",
                                            "Balanced Fitness",
                                            "Starting Positive Count",
                                            "Starting Negative Count",
                                            "Starting Unbalance",
                                            "Balanced Positive Count",
                                            "Balanced Negative Count",
                                            "Ending Unbalance"])
results_df.to_csv("experiment_results.csv")



