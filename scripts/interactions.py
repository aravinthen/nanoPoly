# Program Name: interactions.py
# Author: Aravinthen Rajkumar
# Description: Example of the interactions class

import numpy as np
import time
import sys
sys.path.insert(0, '../main')
from simulation import Simulation
from poly import PolyLattice

# polylattice object
box = PolyLattice(10.0, 20)

def print_type():
    print(box.interactions.types)
    print(box.interactions.type_matrix)
    print(box.interactions.sigma_matrix)
    print(box.interactions.energy_matrix)
    print(box.interactions.cutoff_matrix)

box.interactions.newType("mainbead", 0.5,
                         (1.1, 0.2, 0.2))

box.interactions.newType("subbead", 1.0,
                         (1.1, 0.2, 0.2),
                         ('mainbead,subbead', (1.1, 0.2, 0.2)))

box.interactions.newType("crosslink", 3.0,
                         (1.1, 0.2, 0.2),
                         ('mainbead,crosslink', (0.3, 0.2, 0.2)),
                         ('subbead,crosslink', (10.0, 0.2, 0.2)))

print_type()

print(box.interactions.return_sigma('mainbead','crosslink'))
