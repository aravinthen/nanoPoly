# Program Name: distribution.py
# Author: Aravinthen Rajkumar
# Description: A testing the see_distribution function.

# Program Name: walks_example.py
# Author: Aravinthen Rajkumar
# Description: nanopoly configuration that just runs the random walks

import numpy as np
import time
import random
import sys

sys.path.insert(0, '../main')
from simulation import Simulation
from poly import PolyLattice
from analysis import Check

print("NANOPOLY SIMULATION")
box_size = 100.0
t0 = time.time()    
box = PolyLattice(box_size, cellnums=60)
t1 = time.time()

simname = "big_test"
dmp = 100
print(f"Box generated, with {len(box.Cells)} cells in total. Time taken: {t1 - t0}")

# Order of properties: Sigma, energy, cutoff
box.interactions.newType("a", 1.0,
                         (1.0, 1.0, 1.5))

box.interactions.newType("b", 0.5,
                         (1.0, 0.5, 1.5),
                         ('a,b', (1.0, 0.2, 1.5)))

# following values determine the bonding of the random walks

num_walks = 4
size = 100
# size of the chain
rw_kval = 30
rw_cutoff = 1.5
rw_epsilon = 1.0
rw_sigma = 1.0


# block = 100
copolymer = []
for i in range(1500):
    copolymer.append('a')
for i in range(1500):
    copolymer.append('b')


random_copolymer = []

total_time = 0
for i in range(2):
    t0 = time.time()
    box.random_walk(size,
                    rw_kval,
                    rw_cutoff,
                    rw_epsilon,
                    rw_sigma,
                    bead_sequence = ['a'],
                    initial_failures= 10000,
                    walk_failures = 10000,
                    soften=True,
                    termination="soften")
    t1 = time.time()
    total_time+= t1-t0
    print(f"Walk {box.num_walks} completed in {t1-t0} seconds. Total time elapsed: {total_time}")
    
for i in range(2):
    low_density = box.find_low_density(region=2, quals=1)
    print(low_density)
    low_density_cell = low_density[0]
    t0 = time.time()
    box.random_walk(size,
                    rw_kval,
                    rw_cutoff,
                    rw_epsilon,
                    rw_sigma,
                    bead_sequence = ['b'],
                    initial_failures= 10000,
                    walk_failures = 1000,
                    cell_num = low_density_cell,
                    soften=True,
                    termination="soften")

    t1 = time.time()
    total_time+= t1-t0
    print(f"Walk {box.num_walks} completed in {t1-t0} seconds. Total time elapsed: {total_time}")

t1 = time.time()
total_time+= t1-t0


# box.see_distribution()
box.simulation.structure("test_structure.in")
view_path = "~/ovito/build/bin/ovito"
box.simulation.view(view_path, "test_structure.in")
