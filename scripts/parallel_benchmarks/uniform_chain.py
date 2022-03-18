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

double = False

print("NANOPOLY SIMULATION")
box_size = 17.0
t0 = time.time()    
box = PolyLattice(box_size, cellnums=12)
t1 = time.time()
print(f"Box generated, with {len(box.Cells)} cells in total. Time taken: {t1 - t0}")

# Atom interactions
# TYPES
box.interactions.newType("a", 1.0,
                         (1.0, 0.2, 1.5))

box.interactions.newType("b", 0.5,
                         (1.0, 1.0, 1.5),
                         ('a,b', (1.0, 0.2, 1.5)))

size = 100 # size of the chain
rw_kval = 30.0
rw_cutoff = 1.5
rw_epsilon = 1.0
rw_sigma = 1.0    

copolymer_frac = 5
copolymer = []
for i in range(copolymer_frac):
    copolymer.append('a')
for i in range(copolymer_frac):
    copolymer.append('b')

random_copolymer = []    
num_walks = 15

for i in range(num_walks):
    box.uniform_chain(size,
                      rw_kval,
                      rw_cutoff,
                      rw_epsilon,
                      rw_sigma,
                      termination='soften',
                      bead_sequence=copolymer,
                      srate=0.99,
                      suppress=False,
                      danger=1.09)

    print(f"{i+1} random walked")

print(f"Walk completed in {t1-t0} seconds")

box.mdsim.structure("test_structure.in")
box.mdsim.settings("test_settings.in", prebuilt="test_structure.in")

timestep = 0.01
desc1 = "testing"

dmp=0

box.mdsim.minimize(1e-5, 1e-5, 10000, 100000)

box.mdsim.equilibrate(5000, 
                      timestep, 
                      0.5, 
                      'langevin', 
                      scale=False, 
                      output_steps=100,
                      dump=dmp)

# box.mdsim.run(folder="serial_test") # Total wall time: 0:00:36
box.mdsim.run(folder="parallel_test1", lammps_path="~/Research/lammps/src/lmp_mpi", mpi=3) # Total wall time: 0:00:16
# box.mdsim.run(folder="parallel_test2", lammps_path="~/Research/lammps/src/lmp_mpi", mpi=5) # Total wall time: 0:00:12
# box.mdsim.run(folder="parallel_test3", lammps_path="~/Research/lammps/src/lmp_mpi", mpi=7) # Total wall time: 0:00:12
# box.mdsim.run(folder="parallel_test4", lammps_path="~/Research/lammps/src/lmp_mpi", mpi=10) # Total wall time: 0:00:10
# box.mdsim.run(folder="parallel_test5", lammps_path="~/Research/lammps/src/lmp_mpi", mpi=15) # Total wall time: 0:00:10


if double == True:
    box.reset()

    # Atom interactions
    # TYPES
    box.interactions.newType("a", 1.0,
                             (1.0, 0.2, 1.5))

    box.interactions.newType("b", 0.5,
                             (1.0, 1.0, 1.5),
                             ('a,b', (1.0, 0.2, 1.5)))

    for i in range(3):
        box.uniform_chain(size,
                          rw_kval,
                          rw_cutoff,
                          rw_epsilon,
                          rw_sigma,
                          copolymer,
                          mini=1.1,
                          srate=0.4)
        print(f"{i+1} random walked")

    print(f"Walk completed in {t1-t0} seconds")

    box.mdsim.structure("test_structure.in")
    box.mdsim.settings("test_settings.in", prebuilt="test_structure.in")

    timestep = 0.01
    desc1 = "testing"

    box.mdsim.minimize(1e-5, 1e-5, 10000, 100000)

    box.mdsim.equilibrate(20000, 
                          timestep, 
                          0.5, 
                          'langevin', 
                          scale=False, 
                          output_steps=100,
                          dump=dmp)

    box.mdsim.structure("test_structure.in")
    box.mdsim.settings("test_settings.in", prebuilt="test_structure.in")

    timestep = 0.01
    desc1 = "testing"

    box.mdsim.minimize(1e-5, 1e-5, 10000, 100000)

    box.mdsim.equilibrate(10000, 
                          timestep, 
                          0.5, 
                          'langevin', 
                          scale=False, 
                          output_steps=100,
                          dump=dmp)

    box.mdsim.run(folder="parallel_test2", lammps_path="~/Research/lammps/src/lmp_mpi", mpi=3) # Total wall time: 0:00:16

    # add mpi=7 argument to run with mpi
    # box.mdsim.run(folder="comp_test")
    # box.mdsim.run(folder="long_biax_test")
