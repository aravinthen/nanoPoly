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
                         (0.3, 3,1),
                         ('c,a', (0.2,4,3)),
                         ('c,b', (0.2,4,2)))

box.interactions.newType("d", 1.0,
                         (0.24, 3,1),
                         ('d,a', (0.1,4,2)),
                         ('d,b', (0.15,4,2)),
                         ('d,c', (0.25,4,2)))

box.interactions.newType("e", 1.0,
                         (0.14, 3,1),
                         ('e,a', (0.1,4,2)),
                         ('e,b', (0.15,4,2)),
                         ('e,c', (0.25,4,2)),
                         ('e,d', (0.15,4,2)))

# print(box.interactions.types)
# print(box.interactions.type_matrix)
# print(box.interactions.sigma_matrix)
# print(box.interactions.energy_matrix)
# print(box.interactions.cutoff_matrix)

# following values determine the bonding of the random walks
size = 1000 # size of the chain
rw_kval = 30.0
rw_cutoff = 3.5
rw_epsilon = 0.05
rw_sigma = 0.3
copolymer_frac = 10

# the next ones determine the bonding of the crosslinks
num_links = 1000
cl_kval = 1.1*rw_kval
cl_epsilon = 1.1*rw_epsilon
cl_sigma = 1.1*rw_sigma
cl_cutoff = 1.1*rw_cutoff

copolymer = []
for i in range(copolymer_frac):
    copolymer.append('a')
for i in range(copolymer_frac):
    copolymer.append('b')

random_copolymer = []

total_time = 0
t0 = time.time()    
box.randomwalk(size,
                rw_kval,
                rw_cutoff,
                rw_epsilon,
                rw_sigma,
                bead_sequence = copolymer,
                termination="retract")
t1 = time.time()
total_time+= t1-t0
print(f"Walk completed in {t1-t0} seconds")

t0 = time.time()

box.randomwalk(size,
                0.9*rw_kval,
                rw_cutoff,
                rw_epsilon,
                rw_sigma,
                bead_sequence = ['b', 'c'],
                termination="retract")
t1 = time.time()

print(f"Walk completed in {t1-t0} seconds")

box.bonded_crosslinks('d', # typeid
                      num_links,
                      cl_kval,
                      cl_cutoff,
                      cl_epsilon,
                      cl_sigma,
                      style='fene',
                      forbidden=['b'],
                      selflinking=5)


# box.randomwalk(size,
#                 rw_kval,
#                 rw_cutoff,
#                 rw_epsilon,
#                 rw_sigma,
#                 bead_types = {"b": 1.1,},
#                 restart=True,
#                 termination="retract")

# box.randomwalk(size,
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
timestep = 0.01
desc1 = "testing"
box.simulation.equilibrate(10000,
                           timestep,
                           2,
                           'langevin',
                           output_steps=1000,
                           description=desc1)

view_path = "~/ovito/build/bin/ovito"
box.simulation.view(view_path, "test_structure.in")
# box.simulation.run(folder="long_biax_test")
# add mpi=7 argument to run with mpi
# box.simulation.run(folder="comp_test")
