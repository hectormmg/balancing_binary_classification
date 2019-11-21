#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 20 22:22:23 2019

@author: HectorMg
"""

from species import Species

class SpeciesFactory:
    def initialize_species(X, Y, positive_examples, negative_examples, minority_flag):
        # Positive Species
        species_a = Species(X, Y, positive_examples, negative_examples, 1, 100, minority_flag)
        # Negative Species
        species_b = Species(X, Y, negative_examples, positive_examples, 0, 100, not minority_flag)
        return [species_a, species_b]
    