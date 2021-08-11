# Program Name: equilibration_test.py 
# Author: Aravinthen Rajkumar
# Description: How long do I need to equilibrate?

import numpy as np
import time
import sys

sys.path.insert(0, '../main')
from simulation import Simulation
from poly import PolyLattice
from analysis import Check

print("NANOPOLY SIMULATION")
box_size = 200.0
t0 = time.time()    
box = PolyLattice(box_size, cellnums=100)
t1 = time.time()

simname = "long_equilibrium_test"
dmp = 1000
print(f"Box generated, with {len(box.Cells)} cells in total. Time taken: {t1 - t0}")

# Order of properties: Sigma, energy, cutoff
box.interactions.newType("a", 1.0,
                         (1.0, 1.0, 1.5))

box.interactions.newType("b", 1.01,
                         (1.0, 0.5, 1.5),
                         ('a,b', (1.0, 0.2, 1.5)))

# following values determine the bonding of the random walks
num_walks = 600
size = 800
rw_kval = 30
rw_cutoff = 1.5
rw_epsilon = 1.0
rw_sigma = 1.0

# block = 100

copolymer = []
for i in range(40):
    copolymer.append('a')
for i in range(720):
    copolymer.append('b')
for i in range(40):
    copolymer.append('a')

total_time = 0
for i in range(num_walks):
    t0 = time.time()
    box.randomwalk(size,
                    rw_kval,
                    rw_cutoff,
                    rw_epsilon,
                    rw_sigma,
                    bead_sequence = copolymer,
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
box.simulation.structure("test_structure.in")
t1 = time.time()
total_time = t1-t0
print(f"Structure file created. Total time: {total_time} seconds.")

box.simulation.settings("test_settings.in", nskin=2.0) 
desc1 = "testing"

timestep = 1e-3
# equilibrate the system to iron out the minima
box.simulation.equilibrate(15000,
                           timestep,
                           1.0,
                           'langevin',
                           output_steps=100,
                           description=desc1,
                           dump=dmp)

# compress the system
strain1 = [-2e-2, -2e-2, -2e-2]
box.simulation.deform(9000, 
                      timestep,
                      strain1,
                      1.0,
                      reset=False,
                      dump=dmp,
                      datafile=False,
                      description=desc1)

# equilibrate the system at a higher temperature.
# this allows reptation to take place and further entangle the system
box.simulation.equilibrate(1000000,
                           timestep,
                           2.0,
                           'langevin',
                           output_steps=100,
                           description=desc1,
                           dump=dmp)

# cool the system to below the glass transition temperature
box.simulation.equilibrate(3000000,
                           timestep,
                           0.29,
                           'langevin',
                           output_steps=100,
                           description=desc1,
                           dump=dmp)

# # add mpi=8 argument to run with mpi
box.simulation.run(folder=simname, lammps_path="~/Research/lammps/src/lmp_mpi", mpi=21)
