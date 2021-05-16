import time
import sys

sys.path.insert(0, '../../main')
from simulation import Simulation
from poly import PolyLattice
from analysis import Check, Percolation

print("NANOPOLY SIMULATION")
pair_cutoff = 1.5
pair_sigma = 0.3 # lennard jones potential length, pair potential information
pair_epsilon = 0.05
box_size = 4.0
t0 = time.time()    
box = PolyLattice(box_size, pair_cutoff, pair_sigma, pair_epsilon)
t1 = time.time()
print(f"Box generated, with {len(box.Cells)} cells in total. Time taken: {t1 - t0}")

# types of atom in box-----------------------------------------------------------------------
# key = types, entry = masses
types={"a" : 1.0,
       "b" : 1.0,
       "c" : 1.0}

# RANDOM WALKS ------------------------------------------------------------------------------
nums = 10  # number of random walks
size = 20 # size of the chain
rw_kval = 30.0
rw_cutoff = 1.0
rw_epsilon = 0.05
rw_sigma = 0.4

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

num_links = 5000 # number of crosslinks
mass = 3.0 # mass of crosslinker bead
cl_kval = rw_kval
cl_epsilon = rw_epsilon
cl_sigma = rw_sigma
cl_cutoff = rw_cutoff
t0 = t1 = 0
t0 = time.time()

box.unbonded_crosslinks(50,
                        mass,
                        cl_kval,
                        cl_cutoff,
                        cl_epsilon,
                        cl_sigma,
                        allowed=None,
                        style='fene',
                        prob=0.8,
                        ibonds=2)

crosslinks = box.bonded_crosslinks(num_links,
                                   mass,
                                   cl_kval,
                                   cl_cutoff,
                                   cl_epsilon,
                                   cl_sigma,
                                   forbidden=['a', 'c'],
                                   selflinking=5)

for i in range(3):
    box.random_walk(size,
                    rw_kval,
                    rw_cutoff,
                    rw_epsilon,
                    rw_sigma,
                    bead_types = {'e': 0.5, 'f': 0.3},
                    termination = "retract",
                    allowed_failures=50000)

crosslinks = box.bonded_crosslinks(num_links,
                                   mass,
                                   cl_kval,
                                   cl_cutoff,
                                   cl_epsilon,
                                   cl_sigma,
                                   forbidden=['a', 'c'],
                                   selflinking=5)

for i in range(2):
    box.random_walk(size,
                    rw_kval,
                    rw_cutoff,
                    rw_epsilon,
                    rw_sigma,
                    bead_types = {'f': 0.3},
                    termination = "retract",
                    allowed_failures=50000)

box.unbonded_crosslinks(50,
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

print(f"Crosslinking and entangling concluded. Time taken: {t1 - t0}")

for bead in box.walk_data():
    print(bead)
for bead in box.crosslink_data():
    print(bead)
    
print(box.walkinfo)
print(box.uclinfo)
print(box.bclinfo)

# SIMULATION --------------------------------------------------------------------------------|

timestep = 0.01
t0 = time.time()
box.simulation.structure("test_structure.in")
t1 = time.time()
print(f"Structure file created.Time taken: {t1 - t0}")
box.simulation.settings("test_lattice.in", comms=1.9)

view_path = "~/ovito/build/bin/ovito"
box.simulation.view(view_path, "test_structure.in")


