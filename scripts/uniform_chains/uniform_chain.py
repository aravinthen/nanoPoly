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
box_size = 50.0
t0 = time.time()    
box = PolyLattice(box_size, cellnums=30)
t1 = time.time()
print(f"Box generated, with {len(box.Cells)} cells in total. Time taken: {t1 - t0}")

# Atom interactions
# TYPES
box.interactions.newType("a", 1.0,
                         (1.0, 0.2, 1.5))

box.interactions.newType("b", 0.5,
                         (1.0, 1.0, 1.5),
                         ('a,b', (1.0, 0.2, 1.5)))

size = 500 # size of the chain
rw_kval = 30.0
rw_cutoff = 1.5
rw_epsilon = 1.0
rw_sigma = 1.0


    

copolymer_frac = 10
copolymer = []
for i in range(copolymer_frac):
    copolymer.append('a')
for i in range(copolymer_frac):
    copolymer.append('b')

random_copolymer = []    

box.uniform_chain(50, 
                  rw_kval,
                  rw_cutoff,
                  rw_epsilon,
                  rw_sigma,
                  copolymer)


total_time = 0
t0 = time.time()    
# box.randomwalk(size,
#                 rw_kval,
#                 rw_cutoff,
#                 rw_epsilon,
#                 rw_sigma,
#                 bead_sequence = copolymer,
#                 termination="retract")
t1 = time.time()
total_time+= t1-t0
print(f"Walk completed in {t1-t0} seconds")

box.mdsim.structure("test_structure.in")
box.mdsim.settings("test_settings.in")

timestep = 0.01
desc1 = "testing"


view_path = "~/ovito-basic-3.5.4-x86_64/bin/ovito"
box.mdsim.view(view_path, "test_structure.in")

# add mpi=7 argument to run with mpi
# box.mdsim.run(folder="comp_test")
# box.mdsim.run(folder="long_biax_test")
