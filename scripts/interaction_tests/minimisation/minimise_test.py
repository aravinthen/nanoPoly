# Program Name: interactions.py
# Author: Aravinthen Rajkumar
# Description: Example of the interactions class

import numpy as np
import time
import sys
sys.path.insert(0, '../../../main')
from mdsim import MDSim
from poly import PolyLattice

# polylattice object
box = PolyLattice(50, 30)
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
                         (0.1, 0.2, 1.5))

box.interactions.newType("subbead", 1.0,
                         (0.1, 0.5, 1.5),
                         ('mainbead,subbead', (0.1, 0.2, 1.5)))


# print_type()
# -------------------------------------------------------------------------
# Running random walk
# -------------------------------------------------------------------------
num_walks = 225
rw_kval = 30
rw_cutoff = 1.5
rw_epsilon = 1.0
rw_sigma = 1.0
a
print("Starting walks.")
for i in range(num_walks):
    print(f"Walk {i}")
    box.randomwalk(500,
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

# -------------------------------------------------------------------------
# Modifying interactions mid simulation
# -------------------------------------------------------------------------

box.interactions.modify_sigma('mainbead', 'mainbead', 1.0)
box.interactions.modify_sigma('subbead', 'subbead', 1.0)
box.interactions.modify_sigma('mainbead', 'subbead', 1.0)

# -------------------------------------------------------------------------
# Structure writing
# -------------------------------------------------------------------------

box.mdsim.structure("min_structure.in")
box.mdsim.settings("min_settings.in", nlist=[10,10000], nskin=4.0) 

# -------------------------------------------------------------------------
# Minimization and dynamics
# -------------------------------------------------------------------------
timestep = 0.005
desc1 = "Post minimization run."
box.mdsim.minimize(1e-5, 1e-5, 10000, 100000)
box.mdsim.equilibrate(15000,
                           timestep,
                           1.0,
                           'langevin',
                           output_steps=100,
                           description=desc1,
                           dump=dmp)

view_path = "~/ovito-basic-3.5.4-x86_64/bin/ovito"
box.mdsim.view(view_path, "test_structure.in")

box.mdsim.run(folder="min_test", lammps_path="~/Research/lammps/src/lmp_mpi", mpi=18)


