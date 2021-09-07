# Program Name: parker_rottler.py
# Author: Aravinthen Rajkumar
# Description: recreation of parker-rottler simulation

import numpy as np
import time
import random
import sys

sys.path.insert(0, '../../main')
from simulation import Simulation
from poly import PolyLattice
from analysis import Check

print("NANOPOLY SIMULATION")
box_size = 100.0
t0 = time.time()    
box = PolyLattice(box_size, cellnums=80)
t1 = time.time()

simname = "walks_example"
dmp = 10
print(f"Box generated, with {len(box.Cells)} cells in total. Time taken: {t1 - t0}")

# Order of properties: Sigma, energy, cutoff
box.interactions.newType("a", 1.0,
                         (1.0, 1.0, 1.5))

box.interactions.newType("b", 0.5,
                         (1.0, 1.0, 1.5),
                         ('a,b', (1.0, 0.2, 1.5)))

# following values determine the bonding of the random walks
num_walks = 1
size = 200
# size of the chain
rw_kval = 30
rw_cutoff = 1.5
rw_epsilon = 1.0
rw_sigma = 1.0

# block = 100

copolymer = []
for i in range(200):
    copolymer.append('a')
for i in range(200):
    copolymer.append('b')
for i in range(200):
    copolymer.append('a')

total_time = 0
t0 = time.time()

starts = []
sixth = 1

for i in range(sixth):
    start = [0,0,0]
    for i in range(3):
        start[i] = random.randrange(2,98)        
    start[0] = 0
    starts.append(start)
    
for i in range(sixth):
    start = [0,0,0]
    for i in range(3):
        start[i] = random.randrange(2,98)        
    start[1] = 0
    starts.append(start)

for i in range(sixth):
    start = [0,0,0]
    for i in range(3):
        start[i] = random.randrange(2,98)        
    start[2] = 0
    starts.append(start)

for i in range(sixth):
    start = [0,0,0]
    for i in range(3):
        start[i] = random.randrange(2,98)        
    start[0] = 98    
    starts.append(start)
    
for i in range(sixth):
    start = [0,0,0]
    for i in range(3):
        start[i] = random.randrange(2,98)        
    start[1] = 98
    starts.append(start)

for i in range(sixth):
    start = [0,0,0]
    for i in range(3):
        start[i] = random.randrange(2,98)        
    start[2] = 98
    starts.append(start)


for i in starts:
    box.randomwalk(size,
                   rw_kval,
                   rw_cutoff,
                   rw_epsilon,
                   rw_sigma,
                   bead_sequence = copolymer,
                   initial_failures= 1000,
                   walk_failures = 1000,
                   starting_pos= i,
                   end_pos= [50.0,50.0,50.0],
                   soften=True,
                   termination="soften")

    t1 = time.time()
    total_time+= t1-t0
    print(f"Walk completed in {t1-t0} seconds. Total time elapsed: {total_time}")

    
total_time+= t1-t0
print(f"Walks complete. Total time: {total_time} seconds")

t0 = time.time()
box.simulation.structure("test_structure.in")
t1 = time.time()
total_time = t1-t0
print(f"Structure file created. Total time: {total_time} seconds.")

view_path = "~/ovito/build/bin/ovito"
box.simulation.view(view_path, "test_structure.in")