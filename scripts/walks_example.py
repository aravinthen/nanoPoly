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
box_size = 100.0
t0 = time.time()    
box = PolyLattice(box_size, cellnums=80)
t1 = time.time()

simname = "walks_example"
dmp = 100
print(f"Box generated, with {len(box.Cells)} cells in total. Time taken: {t1 - t0}")

# Order of properties: Sigma, energy, cutoff
box.interactions.newType("a", 1.0,
                         (1.0, 1.0, 1.5))

box.interactions.newType("b", 0.5,
                         (1.0, 1.0, 1.5),
                         ('a,b', (1.0, 0.2, 1.5)))

# following values determine the bonding of the random walks
num_walks = 1
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

# Total wall time: 0:00:56
# box.simulation.run(folder=simname)

# Total wall time: 0:00:15
# box.simulation.run(folder=simname, lammps_path="~/Research/lammps/src/lmp_mpi", mpi=8)

# Total wall time: 0:00:09
# box.simulation.run(folder=simname, lammps_path="~/Research/lammps/src/lmp_mpi", mpi=16)

# Total wall time: 0:00:08
# box.simulation.run(folder=simname, lammps_path="~/Research/lammps/src/lmp_mpi", mpi=20)

# Total wall time: 0:00:08
box.simulation.run(folder=simname, lammps_path="~/Research/lammps/src/lmp_mpi", mpi=21)

# Total wall time: 0:00:09
# box.simulation.run(folder=simname, lammps_path="~/Research/lammps/src/lmp_mpi", mpi=24)

# Total wall time: 0:00:12
# box.simulation.run(folder=simname, lammps_path="~/Research/lammps/src/lmp_mpi", mpi=32)
