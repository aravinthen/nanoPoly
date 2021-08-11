# Program Name: soft_potential_test.py                                                                   
# Author: Aravinthen Rajkumar                                                                            
# Description: nanopoly configuration that demonstrate phase separation                                  
import numpy as np
import time                                                                                              
import sys                                                                                               
                                                                                                         
sys.path.insert(0, '../main')
from simulation import Simulation
from poly import PolyLattice
from analysis import Check
                                                                                                         
print("NANOPOLY SIMULATION")
box_size = 20.0
t0 = time.time()
# smaller cellnums ideal for coblock polymer structures

box = PolyLattice(box_size, cellnums=15)

t1 = time.time()
                                                                                                         
simname = "phase_separate"
                                                                                                         
dmp = 1000
print(f"Box generated, with {len(box.Cells)} cells in total. Time taken: {t1 - t0}")

# Order of properties: Sigma, energy, cutoff
box.interactions.newType("a", 1.0,
                         (1.0, 1.0, 1.5))

box.interactions.newType("b", 1.01,
                         (1.0, 0.5, 1.5),
                         ('a,b', (1.0, 0.2, 1.5)))
                                                                                                         
# following values determine the bonding of the random walks
size = 100
# size of the chain
rw_kval = 30
rw_cutoff = 1.5
rw_epsilon = 1.0
rw_sigma = 1.0

nums = 3
for i in range(nums):
    box.randomwalk(size,
                    rw_kval,
                    rw_cutoff,
                    rw_epsilon,
                    rw_sigma,
                    bead_sequence = ['a'],
                    initial_failures= 10000,
                    walk_failures = 10000,
                    soften=True,
                    termination="soften")
    
for i in range(nums):    
    box.randomwalk(size,
                    rw_kval,
                    rw_cutoff,
                    rw_epsilon,
                    rw_sigma,
                    bead_sequence = ['b'],
                    initial_failures= 10000,
                    walk_failures = 10000,
                    soften=True,
                    termination="soften")
    
print(f"Walks complete.")
t0 = time.time()
box.simulation.structure("test_structure.in")
t1 = time.time()
total_time = t1-t0
print(f"Structure file created. Total time: {total_time} seconds.")

view_path = "~/ovito/build/bin/ovito"
box.simulation.view(view_path, "test_structure.in")
