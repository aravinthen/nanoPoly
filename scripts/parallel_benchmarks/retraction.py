# Program Name: walks_example.py
# Author: Aravinthen Rajkumar
# Description: nanopoly configuration that just runs the random walks

import numpy as np
import time
import random
import sys

sys.path.insert(0, '../../main')

from mdsim import MDSim
from poly import PolyLattice
from analysis import Check
from meanfield import MeanField

print("NANOPOLY SIMULATION")
box_size = 10.0
t0 = time.time()    
box = PolyLattice(box_size, cellnums=5)
t1 = time.time()
print(f"Box generated, with {len(box.Cells)} cells in total. Time taken: {t1 - t0}")

# Atom interactions
# TYPES
box.interactions.newType("a", 1.0,
                         (1.0, 0.2, 1.5))

box.interactions.newType("b", 0.5,
                         (1.0, 1.0, 1.5),
                         ('a,b', (1.0, 0.2, 1.5)))

numwalks = 12
size = 30 # size of the chain
rw_kval = 30.0
rw_cutoff = 1.5
rw_epsilon = 1.0
rw_sigma = 1.0    

copolymer = []
for i in range(10):
    copolymer.append('a')
for i in range(10):
    copolymer.append('b')

random_copolymer = []    

for i in range(numwalks):
    box.randomwalk(size,
                   rw_kval,
                   rw_cutoff,
                   rw_epsilon,
                   rw_sigma,
                   copolymer,
                   mini=1.1,
                   suppress = False,
                   termination='retract',
                   sequence_range=3,
                   srate=0.9,
                   danger = 1.0, 
                   initial_failures = 100,
                   walk_failures = 50)

    print(f"{i+1} random walked")

print(f"Walk completed in {t1-t0} seconds")
print(f"Number of retractions: {box.retractions}")

# for i in box.walk_data():
#     print(i, box.mdsim.global_bead_num(i))

box.mdsim.structure("test_structure.in")
box.mdsim.settings("test_settings.in")

timestep = 0.01
desc1 = "testing"

dmp=0

box.mdsim.minimize(1e-5, 1e-5, 10000, 100000)

box.mdsim.equilibrate(20000, 
                      timestep, 
                      0.5, 
                      'langevin', 
                      scale=False, 
                      output_steps=100,
                      dump=dmp)

print(box.walkinfo)
time.sleep(2)

# box.mdsim.run(folder="serial_test") # Total wall time: 0:00:36
# box.mdsim.run(folder="parallel_test1", lammps_path="~/Research/lammps/src/lmp_mpi", mpi=3) # Total wall time: 0:00:16
# box.mdsim.run(folder="parallel_test2", lammps_path="~/Research/lammps/src/lmp_mpi", mpi=5) # Total wall time: 0:00:12
box.mdsim.run(folder="parallel_test3", lammps_path="~/Research/lammps/src/lmp_mpi", mpi=7) # Total wall time: 0:00:12
# box.mdsim.run(folder="parallel_test4", lammps_path="~/Research/lammps/src/lmp_mpi", mpi=10) # Total wall time: 0:00:10
# box.mdsim.run(folder="parallel_test5", lammps_path="~/Research/lammps/src/lmp_mpi", mpi=15) # Total wall time: 0:00:10
