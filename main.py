#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 15:34:46 2019

@author: HectorMg
"""
import sys
import pandas as pd
import numpy as np

from genetic_algorithm import train

def print_configuration(target_config):
    print(f'Classifier: {target_config["classifier"]}')
    print(f'Fitness Metric: {target_config["metric"]}')
    print(f'Dataset Location: {target_config["dataset_filename"]}\n')

def read_system_params(default_config):
    if(len(sys.argv) == 4):   
        config = dict()
        config['classifier'] = sys.argv[1]
        config['metric'] = sys.argv[2]
        config['dataset_filename'] = sys.argv[3]
        print("Executing CO-evolutionary Algorithm with custom options:")
        print_configuration(config)
        return config
    else:
        print("Executing Co-evolutionary Algorithm with default options:")
        print_configuration(default_config)
        return default_config

def encode_csv_dataset(data_csv):
    test_class = data_csv["Class"][0]
    if(test_class == "yes" or test_class == "no" ):
        data_csv["Class"] = np.where(data_csv["Class"] == "yes", 1, 0)
    return data_csv

def extract_data(encoded_dataset):
    X = np.array(encoded_dataset.iloc[:, 0:-1])
    Y = np.array(encoded_dataset.iloc[:, -1])
    return X, Y


# Main Script
default_config = { 'classifier': "SVM",
                   'metric': "accuracy",
                   'dataset_filename': "../datasets/preprocessed/iris.csv" }

config = read_system_params(default_config) 
data_csv = pd.read_csv(config['dataset_filename'])              
encoded_dataset = encode_csv_dataset(data_csv)
X, Y = extract_data(encoded_dataset)



# Train Model
fullx, fy, best_fitness, shapes = train(X, Y, config['classifier'], config['metric'])

# Report results
reshaped_fy = np.reshape(fy, (len(fy), 1))
new_dataset = np.append(fullx, reshaped_fy, axis=1)
pd.DataFrame(new_dataset).to_csv(config["dataset_filename"] + "_balanced.csv")