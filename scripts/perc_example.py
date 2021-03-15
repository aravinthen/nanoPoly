import numpy as np
import time
import sys

sys.path.insert(0, '../main')
from simulation import Simulation
from poly import PolyLattice
from analysis import Percolation

print("NANOPOLY SIMULATION")
pair_cutoff = 1.5
pair_sigma = 0.3 # lennard jones potential length, pair potential information
pair_epsilon = 0.05
box_size = 2.0
t0 = time.time()    
box = PolyLattice(box_size, pair_cutoff, pair_sigma, pair_epsilon)
t1 = time.time()
print(f"Box generated, with {len(box.Cells)} cells in total. Time taken: {t1 - t0}")

# types of atom in box-----------------------------------------------------------------------
# key = types, entry = masses

types={1 : 1.0,
       2 : 1.0,       
       3 : 1.0}

nums = 2 # number of random walks
size = 10 # size of the chain
rw_kval = 30.0
rw_cutoff = 3.5
rw_epsilon = 0.05
rw_sigma = 0.5

total_time = 0
for i in range(nums):
    t0 = time.time()    
    box.random_walk(size,
                    rw_kval,
                    rw_cutoff,
                    rw_epsilon,
                    rw_sigma,
                    bead_types = types,
                    termination="retract")
    
    t1 = time.time()
    total_time+= t1-t0
    print(f"Random walks: attempt {i+1} successful. Time taken: {t1 - t0}")
    
print(f"Total time taken for random walk configuration: {total_time}")

num_links = 1 # number of crosslinks
mass = 3.0 # mass of crosslinker bead
cl_kval = rw_kval
cl_epsilon = rw_epsilon
cl_sigma = rw_sigma
cl_cutoff = rw_cutoff
t0 = t1 = 0
t0 = time.time()

crosslinks = box.bonded_crosslinks(num_links,
                                   mass,
                                   cl_kval,
                                   cl_cutoff,
                                   cl_epsilon,
                                   cl_sigma,
                                   forbidden=[2],
                                   selflinking=5)

timestep = 0.01
t0 = time.time()
box.simulation.structure("test_structure.in")
t1 = time.time()
print(f"Structure file created.Time taken: {t1 - t0}")
box.simulation.settings("test_lattice.in", comms=1.9)

# box.check.distance_check()

# test_bead
bead = box.walk_data(1)[5]
coord = [bead[0], bead[1]]

paths = box.percolation.blaze(coord)
for i in paths:
    print(i)
    print(" ")

# box.simulation.view("test_structure.in")


