# Program Name:x crosslink_example.py
# Author: Aravinthen Rajkumar
# Description: nanopoly configuration that runs the 

import time
import sys

sys.path.insert(0, '../main')
from simulation import Simulation
from poly import PolyLattice
from analysis import Analysis


print("NANOPOLY SIMULATION")
pair_cutoff = 3.0
pair_sigma = 1.0 # lennard jones potential length, pair potential information
pair_epsilon = 1.0
box_size = 100.0
t0 = time.time()    
box = PolyLattice(box_size, pair_cutoff, pair_sigma, pair_epsilon)
t1 = time.time()
print(f"Box generated, with {len(box.Cells)} cells in total. Time taken: {t1 - t0}")

# types of atom in box-----------------------------------------------------------------------
# key = types, entry = masses
types={1 : 1.0,
       2 : 1.0,
       3 : 1.0}

# RANDOM WALKS ------------------------------------------------------------------------------
nums = 1  # number of random walks
size = 10000 # size of the chain
rw_cutoff = 5.0
rw_kval = 150.0
rw_epsilon = 1.0
rw_sigma = 1.1

total_time = 0
for i in range(nums):
    t0 = time.time()    
    # use cell_num to control where walk starts
    box.random_walk(size,
                    rw_kval,
                    rw_cutoff,
                    rw_epsilon,
                    rw_sigma,
                    bead_types = types,
                    termination = "retract",
                    allowed_failures=50000)
    
    t1 = time.time()
    total_time+= t1-t0
    print(f"Random walks: attempt {i+1} successful. Time taken: {t1 - t0}")    
print(f"Total time taken for random walk configuration: {total_time}")

#  -----------------------------------------------------------------------------------

num_links = 1000 # number of crosslinks
mass = 3.0 # mass of crosslinker bead
cl_kval = rw_kval
cl_epsilon = rw_epsilon
cl_sigma = rw_sigma
cl_cutoff = rw_cutoff
t0 = t1 = 0
t0 = time.time()

# crosslinks = box.bonded_crosslinks(num_links,
#                                    mass,
#                                    cl_kval,
#                                    cl_cutoff,
#                                    cl_epsilon,
#                                    cl_sigma,
#                                    forbidden=[2],
#                                    selflinking=5)

box.unbonded_crosslinks(num_links,
                        mass,
                        cl_kval,
                        cl_cutoff,
                        cl_epsilon,
                        cl_sigma,
                        allowed=None,
                        style='fene',
                        prob=0.8,
                        ibonds=2)

t1 = time.time()
print(f"Crosslinking concluded. Time taken: {t1 - t0}")

# SIMULATION --------------------------------------------------------------------------------|

timestep = 0.01
t0 = time.time()
box.simulation.structure("test_structure.in")
t1 = time.time()
print(f"Structure file created.Time taken: {t1 - t0}")
box.simulation.settings("test_lattice.in", comms=1.9)

desc1 = "Initialization"
desc2 = "Equilibration"
desc3 = "Deformation procedure"

box.simulation.equilibrate(15000,
                           timestep,
                           5.0,
                           'langevin',
                           final_temp= 30.0,
                           bonding=False,
                           description=desc1,
                           reset=False,
                           dump=100)

box.simulation.equilibrate(5000,
                           timestep,
                           30.0,
                           'langevin',
                           bonding=True,
                           description=desc1,
                           reset=False)    

# box.simulation.equilibrate(30000,
#                            timestep,
#                            0.8,
#                            'nose_hoover',
#                            final_temp=0.5,
#                            description=desc2,
#                            reset=False)

# strain = [1e-2, 0, 0]
# box.simulation.deform(50000, timestep, strain, 0.5, reset=False, description=desc3)

# box.analysis.error_check()

box.simulation.structure("test_structure.in")

# box.simulation.view("test_structure.in")
box.simulation.run(folder="hot_box_test")
