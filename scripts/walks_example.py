# Program Name: walks_example.py
# Author: Aravinthen Rajkumar
# Description: nanopoly configuration that just runs the random walks

import numpy as np
import time
import sys

sys.path.insert(0, '../main')
from simulation import Simulation
from poly import PolyLattice
from analysis import Check

print("NANOPOLY SIMULATION")
box_size = 50.0
t0 = time.time()    
box = PolyLattice(box_size, cellnums=100)
t1 = time.time()
print(f"Box generated, with {len(box.Cells)} cells in total. Time taken: {t1 - t0}")

# Order of properties: Sigma, energy, cutoff
box.interactions.newType("a", 1.0,
                         (0.3, 1.0, 1.5))

box.interactions.newType("b", 0.5,
                         (0.3, 0.5, 1.5),
                         ('a,b', (0.3, 0.2, 1.5)))

# following values determine the bonding of the random walks

num_walks = 600
size = 800
# size of the chain
rw_kval = 40.0
rw_cutoff = 3.5
rw_epsilon = 1.0
rw_sigma = 0.4


# block = 100
copolymer = []
for i in range(40):
    copolymer.append('a')
for i in range(720):
    copolymer.append('b')
for i in range(40):
    copolymer.append('a')

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
                    allowed_failures= 1000000,
                    termination="retract")
    t1 = time.time()
    total_time+= t1-t0
    print(f"Walk {i} completed in {t1-t0} seconds. Total time elapsed: {total_time}")

    
total_time+= t1-t0
print(f"Walks complete. Total time: {total_time} seconds")

# vector that controls nature of deformation.
strain = [-1e-2, -1e-2, -1e-2]
#         xx    yy    zz   xy  yz  yz
# box.simulation.deform(10000, timestep, strain, 0.3, reset=False, description=desc3)
t0 = time.time()
box.simulation.structure("test_structure.in")
t1 = time.time()
total_time = t1-t0
print(f"Structure file created. Total time: {total_time} seconds.")

box.simulation.settings("test_settings.in") 
desc1 = "testing"

timestep = 1e-3
box.simulation.equilibrate(50000,
                           timestep,
                           0.05,
                           'langevin',
                           output_steps=100,
                           description=desc1,
                           dump=1000)

strain = [-2e-2, -2e-2, -2e-2]
# box.simulation.deform(10000, timestep, strain, 0.1, reset=False, description=desc1)

view_path = "~/ovito/build/bin/ovito"

# box.simulation.view(view_path, "test_structure.in")

# add mpi=7 argument to run with mpi
box.simulation.run(folder="big_copolymer", mpi=7)

