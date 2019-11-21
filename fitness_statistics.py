#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 20 21:11:05 2019

@author: HectorMg
"""

import matplotlib.pyplot as plt

class FitnessStatistics:
    def __init__(self):
        self.max_positive = list()
        self.max_negative = list()
        self.avg_positive= list()
        self.avg_negative = list()
    
    def plot_max_positive(self):
        plt.plot(self.max_positive, label = "Positive fitness")
    
    def plot_max_negative(self):
        plt.plot(self.max_negative, label = "Negative fitness")
    
    def plot_avg_positive(self):
        plt.plot(self.avg_positive, label = "Average Positive Fitness")
    
    def plot_avg_negative(self):
        plt.plot(self.avg_negative, label = "Average Negative Fitness")
    
    def plot_all(self):
        self.plot_max_positive()
        self.plot_max_negative()
        self.plot_avg_positive()
        self.plot_avg_negative()
        plt.xlabel("Iteration")
        plt.ylabel("Fitness")
        plt.legend()
    