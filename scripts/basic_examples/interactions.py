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
box = PolyLattice(30.0, 20)

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
                         (1, 0.2, 1.3, (3, 0.8, 0.9)))

box.interactions.newType("subbead", 1.0,
                         (1, 0.5, 1.3),
                         ('mainbead,subbead', (1.1, 0.2, 0.1, (3, 0.2, 0.2))))

print_type()
# -------------------------------------------------------------------------
# Running random walk
# -------------------------------------------------------------------------
rw_kval = 30
rw_cutoff = 1.5
rw_epsilon = 1.0
rw_sigma = 1.0

box.randomwalk(30,
               rw_kval,
               rw_cutoff,
               rw_epsilon,
               rw_sigma,
               bead_sequence = ['mainbead', 'subbead'],
               initial_failures= 10000,
               walk_failures = 10000,
               soften=True,
               termination="soften")

box.simulation.structure("int_structure.in")
box.simulation.settings("int_settings.in", nskin=2.0) 

# # -------------------------------------------------------------------------
# # Modifying interactions mid simulation
# # -------------------------------------------------------------------------
box.simulation.modify_interaction('mainbead', 'subbead',
                                  new_sigma=1.3, new_cutoff=2, new_lambda=0.9)

box.simulation.modify_interaction('mainbead', 'mainbead',
                                  new_energy=3, new_cutoff=2)


for i in box.simulation.pending_mods:
    print(i)

box.simulation.run_modifications(1000, 134, 1, 0.1)

# # view_path = "~/ovito/build/bin/ovito"
# # box.simulation.view(view_path, "int_structure.in")
