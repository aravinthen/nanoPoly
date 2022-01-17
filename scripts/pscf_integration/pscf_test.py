# Program Name: 
# Author: Aravinthen Rajkumar
# Description:

import numpy as np
import time
import sys

sys.path.insert(0, '../../main')

from mdsim import MDSim
from poly import PolyLattice
from analysis import Check
from meanfield import MeanField

print("NANOPOLY SIMULATION")
box_size = 10.0
t0 = time.time()    
box = PolyLattice(box_size, cellnums=30)
t1 = time.time()
    
box.interactions.newType("a", 0.5,
                         (0.1, 0.2, 1.5, (1, 1.0, 0.1)))

box.interactions.newType("b", 1.0,
                         (0.1, 0.5, 1.5, (1, 1.0, 0.1)),
                         ('a,b', (0.1, 0.2, 1.5, (1, 1.0, 0.1))))

box.meanfield.parameters("test",
                         [15,15,15],
                         30.0,
                        'cubic',
                         1.0e-2,
                         'I m -3 m',
                         [
                             [("a", 1.0, 0.40), ("b", 1.0, 0.60)]
                         ],
                         cell_param=1.923202,
                         error_max=1e-8,
                         max_itr=100,
                         ncut=95,)

box.meanfield.model_field("test_model", 0, [[0.0, 0.0, 0.0], 
                                            [0.5, 0.5, 0.5]])

path_pscf = "/home/u1980907/Documents/Academia/Research/Code/pscf/bin"
box.meanfield.run(path_pscf)

#----------------------------------------------------------------------------------------
# DENSITY ASSIGNMENT
#----------------------------------------------------------------------------------------
t0 = time.time()    
box.meanfield.density_file("rgrid", [15,15,15])
t1 = time.time()

print(f"Density file read in. Time taken: {t1 - t0}")

#----------------------------------------------------------------------------------------
# RANDOM WALK ASSIGNMENT
#----------------------------------------------------------------------------------------
# following values determine the bonding of the random walks
num_walks = 10
size = 1000
# size of the chain
rw_kval = 30
rw_cutoff = 1.5
rw_epsilon = 1.0
rw_sigma = 1.0

# block = 100

copolymer = []
for i in range(400):
    copolymer.append('a')
for i in range(600):
    copolymer.append('b')

total_time = 0
for i in range(num_walks):
    t0 = time.time()
    box.uniform_chain(size,
                      rw_kval,
                      rw_cutoff,
                      rw_epsilon,
                      rw_sigma,
                      bead_sequence = copolymer,
                      meanfield = True)
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
