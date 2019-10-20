#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 19 21:02:00 2019

@author: HectorMg
"""

import numpy as np
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import f1_score

def accuracy_score(true_positive, false_positive, true_negative, false_negative):
        negative_accuracy = true_positive / (true_positive + true_negative)
        positive_accuracy = false_positive / (false_positive + false_negative)
        return 0.5 * (negative_accuracy + positive_accuracy)

def apply_metric(metric, y_test, prediction, true_negative, false_positive, false_negative, true_positive):
    if (metric == "accuracy"):
        fitness = accuracy_score(true_negative, false_positive, false_negative, true_positive)
    elif (metric == "kappa"):
        fitness = cohen_kappa_score(y_test,prediction)
    elif (metric == "auc"):
        fitness = roc_auc_score(y_test, prediction)
    elif (metric == "f1"):
        fitness = f1_score(y_test, prediction, average='macro')
    
    if np.isnan(fitness):
        fitness = 0.0
        
    return fitness

def evaluate(X_train, y_train, X_test, y_test, classifier, metric, originalN):
    
    positive_count = np.count_nonzero(y_train)
    negative_count = len(y_train) - positive_count
    
    print("positive_count: ", positive_count, ", negative_count: ", negative_count, ", Total: ", len(y_train))
    
    if(classifier == "SVM"):
        clf = svm.SVC(gamma="auto")
    elif (classifier == "G"):
        clf = GaussianNB()

    clf.fit(X_train, y_train)
    prediction = clf.predict(X_test)

    true_negative, false_positive, false_negative, true_positive = confusion_matrix(y_test, prediction).ravel()
    
    fitness = apply_metric(metric, y_test, prediction, true_negative, false_positive, false_negative, true_positive)
        
    return fitness - (0.1 * max(positive_count, negative_count) / min(positive_count, negative_count))