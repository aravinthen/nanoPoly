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
box = PolyLattice(box_size, cellnums=50)
t1 = time.time()
print(f"Box generated, with {len(box.Cells)} cells in total. Time taken: {t1 - t0}")

# Order of properties: Sigma, energy, cutoff
box.interactions.newType("a", 1.0,
                         (1.0, 1.0, 1.5))

box.interactions.newType("b", 0.5,
                         (1.0, 0.5, 1.5),
                         ('a,b', (0.3, 0.2, 1.5)))

# following values determine the bonding of the random walks

num_walks = 1
size = 20
# size of the chain
rw_kval = 1.0
rw_cutoff = 3.5
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

t0 = time.time()
box.simulation.structure("test_structure.in")
t1 = time.time()
total_time = t1-t0
print(f"Structure file created. Total time: {total_time} seconds.")

box.simulation.settings("test_settings.in") 
desc1 = "testing"

timestep = 1e-3
dmp = 1
box.simulation.equilibrate(100,
                           timestep,
                           1.0,
                           'langevin',
                           output_steps=100,
                           description=desc1,
                           dump=dmp)

# strain1 = [-2e-2, -2e-2, -2e-2]
# strain2 = [1e-1, 1e-1, 0]
# strain3 = [0, 0, 1]

# box.simulation.deform(5000, 
#                       timestep,
#                       strain1,
#                       0.5,
#                       reset=False,
#                       dump=dmp,
#                       description=desc1)

# box.simulation.equilibrate(5000,
#                            timestep,
#                            0.1,
#                            'nose-hoover',
#                            output_steps=100,
#                            description=desc1,
#                            reset=False,
#                            dump=dmp)
# box.simulation.deform(2500, 
#                       timestep,
#                       strain2,
#                       0.1,
#                       reset=False,
#                       dump=dmp,
#                       description=desc1)

# box.simulation.deform(500, 
#                       timestep,
#                       strain3,
#                       0.1,
#                       reset=False,
#                       dump=dmp,
#                       description=desc1)

# view_path = "~/ovito/build/bin/ovito"

# # box.simulation.view(view_path, "test_structure.in")

# # add mpi=7 argument to run with mpi
box.simulation.run(folder="testing")
