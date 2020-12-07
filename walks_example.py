# Program Name: walks_example.py
# Author: Aravinthen Rajkumar
# Description: nanopoly configuration that just runs the random walks

# Program Name: polylattice_testing.py
# Author: Aravinthen Rajkumar
# Description: The actual use of the polylattice program will be confined to this file.

from simulation import Simulation
from poly import PolyLattice
import numpy as np
import multiprocessing
import time

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
nums = 3 # number of random walks
size = 5 # size of the chain

# following values determine the bonding 
rw_kval = 30.0
rw_cutoff = 3.5
rw_epsilon = 0.05
rw_sigma = 0.5

errors = 0
total_time = 0
for i in range(nums):
    t0 = time.time()    
    # you can use cell_num to control where walk starts
    box.random_walk(size, rw_kval, rw_cutoff, rw_epsilon, rw_sigma, bead_types = types, termination="retract")
    t1 = time.time()
    total_time+= t1-t0
    print(f"Random walks: attempt {i+1} successful. Time taken: {t1 - t0}")    
print(f"Total time taken for random walk configuration: {total_time}")

# ERROR TESTING ---------------------------------------------------------------------------
shortest_distance = 100
errors = 0
zeros = 0
mean_occ = 0
print(f"\nNumber of cells per side: {box.cellnums}")
print(f"-------------------------------------------------------------------------------")
print(f"Distance Errors:")
print(f"G.num_b\tG.num_n\tWalk_b\tWalk_n\twnum_b\twnum_n\tDistance")
for bead in box.walk_data():
    error_count = 0
    nlist = box.check_surroundings(bead[-1])
    occ = len(nlist)
    mean_occ += occ
    for neighbour in box.check_surroundings(bead[-1]):
        distance = np.linalg.norm(bead[-1] - neighbour[-1])
        if distance < pair_sigma and distance != 0.0:
            print(f"{bead[-2]}\t{neighbour[-2]}\t{bead[0]}\t{neighbour[0]}\t{bead[1]}\t{neighbour[1]}\t{distance}")
            error_count+=1
            if bead[1] == 0 or neighbour[1] == 0:
                zeros +=1
    if error_count > 0:
        errors+=1
print(f"-------------------------------------------------------------------------------")
print(f"The number of errors in distance calculation: {errors}/{box.num_beads}")
print(f"The check_surroundings algorithm messes up {100*(errors)/box.num_beads}% of the time.")
print(f"Of these errors, {zeros} occur with the first bead of a new walk.")
print(f"The mean number of occupants in a surroundings check is {mean_occ/box.num_beads}")
# -----------------------------------------------------------------------------------------

timestep = 0.01
t0 = time.time()
box.simulation.structure("test_structure.in")
t1 = time.time()
print(f"Structure file created.Time taken: {t1 - t0}")
box.simulation.settings("test_lattice.in", comms=1.9)

desc1 = "Testing"
box.simulation.equilibrate(5000, timestep, 0.05, 'langevin', final_temp=0.05, bonding=False, description=desc1, reset=False, dump=0)
# box.simulation.equilibrate(10000, timestep, 0.8, 'langevin', description=desc1, reset=False, dump=100)
# box.simulation.equilibrate(10000, timestep, 0.8, 'nose_hoover', description=desc2, reset=False)
# box.simulation.equilibrate(30000, timestep, 0.8, 'nose_hoover', final_temp=0.5, description=desc3, reset=False)
# box.simulation.deform(100000, timestep, 3e-2, 0.5, reset=False, description=desc4)
# t1 = time.time()
# print(f"Simulation file created.Time taken: {t1 - t0}")

# box.simulation.view("test_structure.in")
# add mpi=7 argument to run with mpi
box.simulation.run(folder="small_test",)



