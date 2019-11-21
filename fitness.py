#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 19 21:02:00 2019

@author: HectorMg
"""
import sys
import numpy as np
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import f1_score

def custom_accuracy_score(y_test, prediction, tp, fp, tn, fn):
    accuracy_negative = tp / (tp + fn)
    accuracy_positive = tn / (fp + tn)

    fitness = 0.5 * (accuracy_negative + accuracy_positive)
    return fitness

def apply_metric(metric, y_test, prediction, tn, fp, fn, tp):
    if (metric == "accuracy"):
        fitness = custom_accuracy_score(y_test, prediction, tn, fp, fn, tp)
    elif (metric == "kappa"):
        fitness = cohen_kappa_score(y_test,prediction)
    elif (metric == "auc"):
        fitness = roc_auc_score(y_test, prediction)
    elif (metric == "f1"):
        fitness = f1_score(y_test, prediction, average='macro')
    
    if np.isnan(fitness):
        fitness = 0.0
        
    return fitness

def classify(X_train, y_train, X_test, y_test, classifier="SVM"):
    if(classifier == "SVM"):
        clf = svm.SVC(gamma="auto")
    elif (classifier == "G"):
        clf = GaussianNB()
    clf.fit(X_train, y_train)
     
    return clf.predict(X_test)

def evaluate(X_train, y_train, X_test, y_test, classifier, metric):
    
    positive_count = np.count_nonzero(y_train)
    negative_count = len(y_train) - positive_count
#   print("positive_count: ", positive_count, ", negative_count: ", negative_count, ", Total: ", len(y_train))
    prediction = classify(X_train, y_train, X_test, y_test, classifier)
    tn, fp, fn, tp = confusion_matrix(y_test, prediction).ravel()
    fitness = apply_metric(metric, y_test, prediction, tn, fp, fn, tp)
        
    return fitness - (0.1 * max(positive_count, negative_count) / min(positive_count, negative_count))