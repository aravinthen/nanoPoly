# Program Name: graft_example.py
# Author: Aravinthen Rajkumar
# Description: test of the chain grafting mechanism


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
box = PolyLattice(box_size, cellnums=10)
t1 = time.time()
print(f"Box generated, with {len(box.Cells)} cells in total. Time taken: {t1 - t0}")

# Atom interactions
# TYPES
box.interactions.newType("a", 0.5,
                         (0.2, 2,1))

box.interactions.newType("b", 1.0,
                         (0.3,3,1),
                         ('a,b', (0.3,4,2)))

box.interactions.newType("c", 1.0,
                         (0.2, 3,1),
                         ('c,a', (0.2,4,2)),
                         ('c,b', (0.2,4,2)))

box.interactions.newType("d", 1.0,
                         (0.2, 3,1),
                         ('d,a', (0.1,4,2)),
                         ('d,b', (0.15,4,2)),
                         ('d,c', (0.2,4,2)))

# following values determine the bonding of the random walks
size = 100 # size of the chain
rw_kval = 30.0
rw_cutoff = 3.5
rw_epsilon = 0.05
rw_sigma = 0.3
graft_spacing = 20
graft_size = 10
graft_num = 4

total_time = 0
t0 = time.time()

box.random_walk(size,
                rw_kval,
                rw_cutoff,
                rw_epsilon,
                rw_sigma,
                bead_sequence = ['a','b'],
                starting_pos = [5.0, 5.0, 5.0],
                termination="retract")
print("Random walk completed.")

for i in range(1, size, graft_spacing):
    for j in range(graft_num):
        box.graft_chain([1, i],
                        graft_size,
                        rw_kval,
                        rw_cutoff,
                        rw_epsilon,
                        rw_sigma,
                        bead_sequence=['c','d'])
    print(f"Graft {i} completed.")


t1 = time.time()
total_time+= t1-t0
print(f"Walk completed in {t1-t0} seconds")

# box.random_walk(size,
#                 rw_kval,
#                 rw_cutoff,
#                 rw_epsilon,
#                 rw_sigma,
#                 bead_types = {"b": 1.1,},
#                 restart=True,
#                 termination="retract")

# box.random_walk(size,
#                 rw_kval,
#                 rw_cutoff,
#                 rw_epsilon,
#                 rw_sigma,
#                 bead_types = {"c": 1.2},
#                 restart=True,
#                 termination="break")

# for i in box.walk_data():
#    print(i)

# vector that controls nature of deformation.
strain = [-1e-2, -1e-2, -1e-2]
#         xx    yy    zz   xy  yz  yz
# box.simulation.deform(10000, timestep, strain, 0.3, reset=False, description=desc3)
box.simulation.structure("test_structure.in")
box.simulation.settings("test_settings.in")

view_path = "~/ovito/build/bin/ovito"
box.simulation.view(view_path, "test_structure.in")
# box.simulation.run(folder="long_biax_test")
# add mpi=7 argument to run with mpi
# box.simulation.run(folder="comp_test")

