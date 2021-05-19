# Program Name: walks_example.py
# Author: Aravinthen Rajkumar
# Description: nanopoly configuration that just runs the random walks

# Program Name: polylattice_testing.py
# Author: Aravinthen Rajkumar
# Description: The actual use of the polylattice program will be confined to this file.

import numpy as np
import time
import sys

sys.path.insert(0, '../main')
from simulation import Simulation
from poly import PolyLattice
from analysis import Check

print("NANOPOLY SIMULATION")
box_size = 10.0
t0 = time.time()    
box = PolyLattice(box_size, cellnums=40)
t1 = time.time()
print(f"Box generated, with {len(box.Cells)} cells in total. Time taken: {t1 - t0}")

box.interactions.newType("a", 0.5,
                         (0.1, 2, 1))

box.interactions.newType("b", 1.0,
                         (0.2,3,1),
                         ('a,b', (0.1, 4,2)))

box.interactions.newType("c", 3.0, (0.25,3,1),
                         ('c,a', (0.19,4,2)),
                         ('c,b', (0.19,4,2)))

# following values determine the bonding of the random walks

num_walks = 100
size = 100
# size of the chain
rw_kval = 30.0
rw_cutoff = 3.5
rw_epsilon = 0.05
rw_sigma = 0.3


block = 100
copolymer = []
for i in range(block):
    copolymer.append('a')
for i in range(block):
    copolymer.append('b')

random_copolymer = []

total_time = 0
for i in range(num_walks):
    t0 = time.time()
    box.random_walk(size,
                    rw_kval,
                    rw_cutoff,
                    rw_epsilon,
                    rw_sigma,
                    bead_sequence = copolymer,
                    termination="retract")
    t1 = time.time()
    total_time+= t1-t0
    print(f"Walk {i} completed in {t1-t0} seconds. Total time elapsed: {total_time}")

    
total_time+= t1-t0
print(f"Walks complete. Total time: {total_time} seconds")

howmany = len(box.walk_data())
print(howmany)

# vector that controls nature of deformation.
strain = [-1e-2, -1e-2, -1e-2]
#         xx    yy    zz   xy  yz  yz
# box.simulation.deform(10000, timestep, strain, 0.3, reset=False, description=desc3)
box.simulation.structure("test_structure.in")
 
view_path = "~/ovito/build/bin/ovito"
box.simulation.view(view_path, "test_structure.in")

# box.simulation.run(folder="long_biax_test")
# add mpi=7 argument to run with mpi
# box.simulation.run(folder="comp_test")

