# Program Name: polylattice_testing.py
# Author: Aravinthen Rajkumar
# Description: The actual use of the polylattice program will be confined to this file.

from simulation import Simulation
from poly import PolyLattice
import multiprocessing
import time

print("NANOPOLY SIMULATION")
pair_cutoff = 1.5
pair_sigma = 0.3 # lennard jones potential length, pair potential information
pair_epsilon = 0.05
box_size = 1
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
nums = 1# number of random walks
size = 35# size of the chain

# following values determine the bonding 
rw_kval = 30.0
rw_cutoff = 3.5
rw_epsilon = 0.05
rw_sigma = 0.5

# SERIALIZED VERSION: 
total_time = 0
for i in range(nums):
    t0 = time.time()    
#   you can use cell_num to control where walk starts
    # box.random_walk(i, size, rw_kval, rw_cutoff, rw_epsilon, rw_sigma, bead_types = types)
    box.random_walk(size, rw_kval, rw_cutoff, rw_epsilon, rw_sigma, bead_types = types, termination="retract")
    t1 = time.time()
    total_time+= t1-t0
    print(f"Random walks: attempt {i+1} successful. Time taken: {t1 - t0}")    
print(f"Total time taken for random walk configuration: {total_time}")

# crosslinking information-------------------------------------------------------------------
num_links = 100 # number of crosslinks
mass = 3.0 # mass of crosslinker bead
cl_kval = rw_kval
cl_epsilon = rw_epsilon
cl_sigma = rw_sigma
cl_cutoff = rw_cutoff

t0 = t1 = 0
t0 = time.time()
# crosslinks = box.bonded_crosslinks(num_links, mass, cl_kval, cl_cutoff, cl_epsilon, cl_sigma, forbidden=[2], selflinking=30)
# box.unbonded_crosslinks(num_links, mass, cl_kval, cl_cutoff, cl_epsilon, cl_sigma, allowed=None, style='fene', prob=0.8, ibonds=2)
t1 = time.time()
print(f"Crosslinking concluded. Time taken: {t1 - t0}")

# box.file_dump("data.txt")

timestep = 0.01
desc1 = "Langevin dynamics at 2T*, NVE ensemble."
desc2 = "Nose-Hoover dynamics at 2T*, NPT ensemble."
desc3 = "Nose-Hoover dynamics from 2T* to 0.5T*, NPT ensemble."
desc4 = "Deformation procedure, 3e-2 engineering strain at temp"

t0 = time.time()
box.simulation.structure("test_structure.in")
t1 = time.time()
print(f"Structure file created.Time taken: {t1 - t0}")
t0 = time.time()
box.simulation.settings("test_lattice.in", comms=1.9)
box.simulation.equilibrate(20000, timestep, 0.05, 'langevin', final_temp=0.2, pdamp=1000000, bonding=False, description=desc1, reset=False, dump=100)
# box.simulation.equilibrate(10000, timestep, 0.8, 'langevin', description=desc1, reset=False, dump=100)
# box.simulation.equilibrate(10000, timestep, 0.8, 'nose_hoover', description=desc2, reset=False)
# box.simulation.equilibrate(30000, timestep, 0.8, 'nose_hoover', final_temp=0.5, description=desc3, reset=False)
# box.simulation.deform(100000, timestep, 3e-2, 0.5, reset=False, description=desc4)
t1 = time.time()
print(f"Simulation file created.Time taken: {t1 - t0}")

box.simulation.view("test_structure.in")
# box.simulation.run(folder="test", mpi=7)

