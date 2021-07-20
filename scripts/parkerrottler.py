# Program Name: parker_rottler.py
# Author: Aravinthen Rajkumar
# Description: recreation of parker-rottler simulation

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

simname = "parker_rottler"
dmp = 1000
print(f"Box generated, with {len(box.Cells)} cells in total. Time taken: {t1 - t0}")

# Order of properties: Sigma, energy, cutoff
box.interactions.newType("a", 1.0,
                         (1.0, 1.0, 1.5))

box.interactions.newType("b", 0.5,
                         (1.0, 1.0, 1.5),
                         ('a,b', (1.0, 0.2, 1.5)))

# following values determine the bonding of the random walks
num_walks = 600
size = 800
# size of the chain
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
    box.random_walk(size,
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

box.simulation.settings("test_settings.in") 
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
box.simulation.equilibrate(100000,
                           timestep,
                           2.0,
                           'langevin',
                           output_steps=100,
                           description=desc1,
                           dump=dmp)

# cool system down to a rubbery state.
box.simulation.equilibrate(50000,
                           timestep,
                           2.0,
                           'langevin',
                           final_temp=1.0,
                           output_steps=100,
                           description=desc1,
                           dump=dmp)

# cool the system to below the glass transition temperature
box.simulation.equilibrate(25000,
                           timestep,
                           0.29,
                           'langevin',
                           output_steps=100,
                           description=desc1,
                           dump=dmp)

# equilibrate the system at glass transition temperature
box.simulation.equilibrate(50000,
                           timestep,
                           0.29,
                           'langevin',
                           output_steps=100,
                           description=desc1,
                           dump=dmp)

# stretch to 200% extension
# conserves box volume
# dy, dz = -1 + sqrt( 1/( 1 + dx))))
strain3 = [0.5e-3, -2.5e-4, -2.5e-4]
box.simulation.deform(640000, 
                      timestep,
                      strain3,
                      0.29,
                      reset=False,
                      dump=dmp,
                      description=desc1)

# # add mpi=8 argument to run with mpi
box.simulation.run(folder=simname, mpi=16)
