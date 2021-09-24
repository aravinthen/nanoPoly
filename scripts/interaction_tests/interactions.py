# Program Name: interactions.py
# Author: Aravinthen Rajkumar
# Description: Example of the interactions class

import numpy as np
import time
import sys
sys.path.insert(0, '../../main')
from mdsim import MDSim
from poly import PolyLattice

# polylattice object
box = PolyLattice(22.0, 15)
dmp = 100

def print_type():
    print(box.interactions.types)
    print(box.interactions.type_matrix)
    print(box.interactions.sigma_matrix)
    print(box.interactions.energy_matrix)
    print(box.interactions.cutoff_matrix)
    print(box.interactions.n_matrix)
    print(box.interactions.alpha_matrix)
    print(box.interactions.lmbda_matrix)
    
box.interactions.newType("mainbead", 0.5,
                         (0.1, 0.2, 1.5, (1, 1.0, 0.1)))

box.interactions.newType("subbead", 1.0,
                         (0.1, 0.5, 1.5, (1, 1.0, 0.1)),
                         ('mainbead,subbead', (0.1, 0.2, 1.5, (1, 1.0, 0.1))))


# print_type()
# -------------------------------------------------------------------------
# Running random walk
# -------------------------------------------------------------------------
num_walks = 10
rw_kval = 30
rw_cutoff = 1.5
rw_epsilon = 1.0
rw_sigma = 1.0

print("Starting walks.")
for i in range(num_walks):
    print(f"Walk {i}")
    box.randomwalk(800,
                   rw_kval,
                   rw_cutoff,
                   rw_epsilon,
                   rw_sigma,
                   bead_sequence = ['mainbead', 'subbead'],
                   initial_failures= 10000,
                   walk_failures = 10000,
                   soften=True,
                   termination="soften")

print("Walks complete.")
box.mdsim.structure("int_structure.in")
box.mdsim.settings("int_settings.in", nlist=[10,10000], nskin=4.0) 

# # -------------------------------------------------------------------------
# # Modifying interactions mid simulation
# # -------------------------------------------------------------------------

def chain_grow(type1, type2, lmbdaval, mod_time, mod_space, equib_time, temp, tstep):
    box.mdsim.modify_interaction(type1, type2, new_lambda= lmbdaval)
    box.mdsim.modify_interaction(type1, type1, new_lambda= lmbdaval)
    box.mdsim.modify_interaction(type2, type2, new_lambda= lmbdaval)
    box.mdsim.run_modifications(mod_time, mod_space, temp, tstep, damp=1, dump=dmp, scale=False)

    box.mdsim.equilibrate(equib_time, 
                               tstep, 
                               temp, 
                               'langevin', 
                               scale=False, 
                               output_steps=100,
                               damp=1,
                               dump=dmp)

box.mdsim.modify_interaction('mainbead', 'subbead', new_sigma=1.0, new_cutoff=1.5)
box.mdsim.modify_interaction('mainbead', 'mainbead', new_sigma=1.0, new_cutoff=1.5)
box.mdsim.modify_interaction('subbead', 'subbead', new_sigma=1.0, new_cutoff=1.5)
box.mdsim.run_modifications(2000, 100, 0, 0.1, dump=dmp, scale=False)


chain_grow('mainbead', 'subbead', 1.0, 20000, 200, 50000, 0.0, 0.01)

box.mdsim.run(folder="fail2", lammps_path="~/Research/lammps/src/lmp_mpi", mpi=18)
