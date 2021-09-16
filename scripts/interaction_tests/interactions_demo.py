# Program Name: interactions.py
# Author: Aravinthen Rajkumar
# Description: Example of the interactions class

import numpy as np
import time
import sys
sys.path.insert(0, '../../main')
from simulation import Simulation
from poly import PolyLattice

# polylattice object
box = PolyLattice(20.0, 15)
dmp = 0

def print_type():
    print(box.interactions.types)
    print(box.interactions.type_matrix)
    print(box.interactions.sigma_matrix)
    print(box.interactions.energy_matrix)
    print(box.interactions.cutoff_matrix)
    print(box.interactions.n_matrix)
    print(box.interactions.alpha_matrix)
    print(box.interactions.lmbda_matrix)
    
box.interactions.newType("mainbead", 0.5,
                         (0.1, 0.2, 1.5, (1, 1.0, 0.1)))

box.interactions.newType("subbead", 1.0,
                         (0.1, 0.5, 1.5, (1, 1.0, 0.1)),
                         ('mainbead,subbead', (0.1, 0.2, 1.5, (1, 1.0, 0.1))))


box.interactions.modify_sigma("mainbead", "mainbead", 0.5)
box.interactions.modify_sigma("subbead", "subbead", 0.5)
