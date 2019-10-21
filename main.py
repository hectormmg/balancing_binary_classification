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

def read_system_params():
    sys.argv = [0,0,0,0,0,0]
    sys.argv[1] = "SVM"
    sys.argv[2] = "accuracy"
    sys.argv[3] = "../datasets/preprocessed/iris.csv"
    return sys.argv[1], sys.argv[2], sys.argv[3]

def encode_csv_dataset(data_csv):
    data_csv["Class"] = np.where(data_csv["Class"] == "yes", 1, 0)
    return data_csv

def extract_data(encoded_dataset):
    X = np.array(encoded_dataset.iloc[:, 0:-1])
    Y = np.array(encoded_dataset.iloc[:, -1])
    return X, Y
    

# Main Script

classifier, metric, dataset_filename = read_system_params()  
data_csv = pd.read_csv(dataset_filename)              
encoded_dataset = encode_csv_dataset(data_csv)
X, Y = extract_data(encoded_dataset)

# Train Model
fullx, fy = train(X, Y, classifier, metric)

# Report results
reshaped_fy = np.reshape(fy, (len(fy), 1))
print(fullx.shape, reshaped_fy.shape)
new_dataset = np.append(fullx, reshaped_fy, axis=1)
pd.DataFrame(new_dataset).to_csv(sys.argv[3] + "_balanced.csv")