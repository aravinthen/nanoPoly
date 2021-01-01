# Program name: data_set.py
# Author: Aravinthen Rajkumar
# Description: Generates a data set using the nanoPoly code

from simulation import Simulation
from poly import PolyLattice
from datetime import date
import time
import os

import sys
sys.path.insert(0, '../main')

#############################################################################################
#                                     DATA SPECIFICATIONS                                   # 
#############################################################################################
# the system defined in this section will be generated repeatedly.

pair_cutoff = 1.5
pair_sigma = 0.5 # lennard jones potential length, pair potential information
pair_epsilon = 1.0

celllen = 2.0 # cell length for lattice box
cellnums = 10 # number of cells in each direction

# types of atom in box-----------------------------------------------------------------------
# key = types, entry = masses
types={1 : 1.0,
       2 : 1.0,
       3 : 1.0}

# random walk information  ------------------------------------------------------------------
num_walks = 5 # number of random walks
size = 1360 # size of the chain

# following values determine the bonding 
rw_kval = 30.0
rw_cutoff = 4.5
rw_epsilon = 1.0
rw_sigma = 1.0

# crosslinking information  -----------------------------------------------------------------
mass = 3.0 # mass of crosslinker bead
cl_kval = rw_kval
cl_epsilon = rw_epsilon
cl_sigma = rw_sigma
cl_cutoff = rw_cutoff

# simulation information  -------------------------------------------------------------------
timestep = 0.01
desc1 = "Langevin dynamics at 2T*, NVE ensemble."
desc2 = "Nose-Hoover dynamics at 2T*, NPT ensemble."
desc3 = "Nose-Hoover dynamics from 2T* to 0.8*, NPT ensemble."
desc4 = "Deformation procedure, 3e-2 engineering strain at temperture 0.5T*."
#############################################################################################

cl_values = [100]
number_of_samples = 10

for i in cl_values:
    for sample in range(number_of_samples):
        print("--------------------------------------------------------------------------------------")
        print(f"               MOLECULAR DYNAMICS SIMULATION WITH {i} CROSSLINKERS")
        print(f"                                SAMPLE NUMBER: {sample}")
        print("--------------------------------------------------------------------------------------")
        box = PolyLattice(celllen, cellnums, pair_cutoff, pair_sigma, pair_epsilon)
        t0 = time.time()    
        for j in range(num_walks):
            current_walks = box.num_walks
            while True:
                try:
                    box.random_walk(j, size, rw_kval, rw_cutoff, rw_epsilon, rw_sigma, bead_types = types)
                    if box.num_walks == current_walks+1:
                        print(f"Simulation {i}_{sample}: Random walk {j} complete")
                        break
                except:
                    print("Oooops! Random walk crashed!")
                    for cell in box.Cells:
                        cell.beads = [bead for bead in cell.beads if bead[0] != j]
                    
        t1 = time.time()
        print(f"Total time taken for random walk configuration: {t1 - t0}")
        
        # Crosslinking procedure
        t0 = t1 = 0
        t0 = time.time()
        crosslinks = box.crosslink(i, mass, cl_kval, cl_cutoff, cl_epsilon, cl_sigma, forbidden=[2], selflinking=30)
        t1 = time.time()
        print(f"Crosslinking concluded. Time taken: {t1 - t0}")
        

        box.simulation.structure("test_structure.in")
        box.simulation.settings("test_lattice.in")
        box.simulation.equilibration(10000, timestep, 2, 'langevin', output_steps=1000, description=desc1)
        box.simulation.equilibration(10000, timestep, 2, 'nose_hoover', description=desc2)
        box.simulation.equilibration(30000, timestep, 2, 'nose_hoover', final_temp=0.8, description=desc3)

        box.simulation.deform(25000, timestep, 3e-2, 0.8, description=desc4)

        box.simulation.files()
        box.simulation.run(folder=f"Experiment_{sample}_{i}", mpi=4)
    
    os.system(f"mkdir crosslink_{i}")
    os.system(f"mv *Experiment* crosslink_{i}")

today = date.today()
os.system(f"mkdir dataset_{today}")
os.system(f"mv *crosslink_* dataset_{today}")

