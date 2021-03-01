# Program Name: walks_example.py
# Author: Aravinthen Rajkumar
# Description: nanopoly configuration that just runs the random walks

# Program Name: polylattice_testing.py
# Author: Aravinthen Rajkumar
# Description: The actual use of the polylattice program will be confined to this file.

import numpy as np
import time
import sys

sys.path.insert(0, '../main')
from simulation import Simulation
from poly import PolyLattice
from analysis import Analysis

print("NANOPOLY SIMULATION")
pair_cutoff = 1.5
pair_sigma = 0.3 # lennard jones potential length, pair potential information
pair_epsilon = 0.05
box_size = 1.0
t0 = time.time()    
box = PolyLattice(box_size, pair_cutoff, pair_sigma, pair_epsilon)
t1 = time.time()
print(f"Box generated, with {len(box.Cells)} cells in total. Time taken: {t1 - t0}")

# types of atom in box-----------------------------------------------------------------------
# key = types, entry = masses
types={1 : 1.0,
       2 : 1.0,       
       3 : 1.0}
# random walk information--------------------------------------------------------------------
nums = 2 # number of random walks
size = 10 # size of the chain

# following values determine the bonding 
rw_kval = 30.0
rw_cutoff = 3.5
rw_epsilon = 0.05
rw_sigma = 0.5

total_time = 0
for i in range(nums):
    t0 = time.time()    
    # you can use cell_num to control where walk starts
    box.random_walk(size, rw_kval, rw_cutoff, rw_epsilon, rw_sigma, bead_types = types, termination="retract")
    t1 = time.time()
    total_time+= t1-t0
    print(f"Random walks: attempt {i+1} successful. Time taken: {t1 - t0}")
print(f"Total time taken for random walk configuration: {total_time}")


timestep = 0.01
t0 = time.time()
box.simulation.structure("test_structure.in")
t1 = time.time()
print(f"Structure file created.Time taken: {t1 - t0}")
box.simulation.settings("test_lattice.in", comms=1.9)

box.analysis.error_check()

desc1 = "Initialization"                                                                               
desc2 = "Equilibration"                                                                                
desc3 = "Deformation procedure"    

box.simulation.equilibrate(20000,
                           timestep,
                           0.1,
                           'langevin',
                           final_temp=0.1,
                           bonding=False,
                           description=desc1,
                           reset=False,
                           dump=10)           

box.simulation.equilibrate(100000,
                           timestep,
                           0.1,
                           'langevin',
                           final_temp=0.3,
                           description=desc1,
                           reset=False)    

box.simulation.equilibrate(30000,
                           timestep,
                           0.3,
                           'nose_hoover',
                           description=desc2,
                           reset=False)

# vector that controls nature of deformation.
strain = [-1e-2, -1e-2, -1e-2]
#         xx    yy    zz   xy  yz  yz
box.simulation.deform(10000, timestep, strain, 0.3, reset=False, description=desc3) 


# box.simulation.view("test_structure.in")
# add mpi=7 argument to run with mpi
box.simulation.run(folder="comp_test")
