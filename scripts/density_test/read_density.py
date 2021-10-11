# Program Name: read_density.py
# Author: Aravinthen Rajkumar
# Description: Example of reading a density into nanopoly

import numpy as np
import time
import sys

sys.path.insert(0, '../../main')

from mdsim import MDSim
from poly import PolyLattice
from analysis import Check
from meanfield import MeanField

print("NANOPOLY SIMULATION")
box_size = 50.0
t0 = time.time()    
box = PolyLattice(box_size, cellnums=40)
t1 = time.time()

simname = "walks_example"
dmp = 100
print(f"Box generated, with {len(box.Cells)} cells in total. Time taken: {t1 - t0}")
#----------------------------------------------------------------------------------------
# INTERACTION ASSIGNMENTS
#----------------------------------------------------------------------------------------
# Order of properties: Sigma, energy, cutoff
box.interactions.newType("a", 1.0,
                         (0.01, 0.2, 1.5))

box.interactions.newType("b", 0.5,
                         (0.01, 1.0, 1.5),
                         ('a,b', (0.1, 0.2, 1.5)))

#----------------------------------------------------------------------------------------
# DENSITY ASSIGNMENT
#----------------------------------------------------------------------------------------
t0 = time.time()    
box.meanfield.density_file("rho_grid", [20,20,20])
t1 = time.time()
print(f"Density file read in. Time taken: {t1 - t0}")

#----------------------------------------------------------------------------------------
# RANDOM WALK ASSIGNMENT
#----------------------------------------------------------------------------------------
# following values determine the bonding of the random walks
num_walks = 100
size = 800
# size of the chain
rw_kval = 30
rw_cutoff = 1.5
rw_epsilon = 1.0
rw_sigma = 1.0

# block = 100

copolymer = []
for i in range(400):
    copolymer.append('a')
for i in range(400):
    copolymer.append('b')

total_time = 0
for i in range(num_walks):
    t0 = time.time()
    box.randomwalk(size,
                   rw_kval,
                   rw_cutoff,
                   rw_epsilon,
                   rw_sigma,
                   bead_sequence = copolymer,
                   meanfield = True,
                   initial_failures= 10000,
                   walk_failures = 10000,
                   soften=True,
                   termination="soften")
    t1 = time.time()
    total_time+= t1-t0
    print(f"Walk {i} completed in {t1-t0} seconds. Total time elapsed: {total_time}")

    
total_time+= t1-t0
print(f"Walks complete. Total time: {total_time} seconds")

t0 = time.time()
box.mdsim.structure("test_structure2.in")
t1 = time.time()
total_time = t1-t0
print(f"Structure file created. Total time: {total_time} seconds.")

box.mdsim.settings("test_settings.in", nskin=2.0, nlist=[10,10000]) 
desc1 = "testing"

timestep = 1e-3

dmp = 100
# -------------------------------------------------------------------------------------

view_path = "~/ovito/build/bin/ovito"
box.mdsim.view(view_path, "test_structure2.in")

# box.mdsim.run(folder="frozen_velocities", lammps_path="~/Research/lammps/src/lmp_mpi", mpi=18)
