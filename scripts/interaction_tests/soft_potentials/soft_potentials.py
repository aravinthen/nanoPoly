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
box = PolyLattice(22.0, 40)
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
                         (0.1, 0.2, 1.5, (1, 1.0, 0.3)))

box.interactions.newType("subbead", 1.0,
                         (0.1, 0.5, 1.5, (1, 1.0, 0.3)),
                         ('mainbead,subbead', (0.1, 0.2, 1.5, (1, 1.0, 0.3))))


# print_type()
# -------------------------------------------------------------------------
# Running random walk
# -------------------------------------------------------------------------
num_walks = 10
rw_kval = 30
rw_cutoff = 1.5
rw_epsilon = 1.0
rw_sigma = 1.0

timestep = 0.005

print("Starting walks.")
for i in range(num_walks):
    print(f"Walk {i}")
    box.randomwalk(1000,
                   rw_kval,
                   rw_cutoff,
                   rw_epsilon,
                   rw_sigma,
                   bead_sequence = ['mainbead' for i in range(500)]+['subbead' for i in range(500)],
                   initial_failures= 10000,
                   walk_failures = 10000,
                   soften=True,
                   termination="soften")

print("Walks complete.")
# -------------------------------------------------------------------------
# Writing structure and settings
# -------------------------------------------------------------------------
box.mdsim.structure("int_structure.in")
box.mdsim.settings("int_settings.in", nlist=[10,10000], nskin=4.0) 

# -------------------------------------------------------------------------
# Modifying interactions
# -------------------------------------------------------------------------
box.mdsim.modify_interaction('mainbead', 'mainbead', new_sigma=1.0)
box.mdsim.modify_interaction('subbead', 'subbead', new_sigma=1.0)
box.mdsim.modify_interaction('mainbead', 'subbead', new_sigma=1.0)

box.mdsim.run_modifications(2000, 100, 0, 0.1, dump=dmp, scale=False)

# -------------------------------------------------------------------------
# Running soft-core potentials at 0 temperature
# -------------------------------------------------------------------------
# It is possible to run this step without damping, but for the assembly of 
# block copolymers it is preferable to have very high damping.
# box.mdsim.equilibrate(50000, 
#                       timestep, 
#                       0, 
#                       'langevin', 
#                       scale=False, 
#                       output_steps=100,
#                       dump=dmp)

# -------------------------------------------------------------------------
# Modifying soft-core potentials to hard-core Lennard-Jones potential
# -------------------------------------------------------------------------
box.interactions.modify_lambda('mainbead', 'mainbead', 1.0)
box.interactions.modify_lambda('subbead', 'subbead', 1.0)
box.interactions.modify_lambda('mainbead', 'subbead', 1.0)

box.mdsim.reapply_interactions() # simply writes any updated interactions
                                 # no smooth transitions included.

# -------------------------------------------------------------------------
# Minimize to iron out any weird spots
# -------------------------------------------------------------------------
box.mdsim.minimize(1e-5, 1e-5, 10000, 100000)

# -------------------------------------------------------------------------
# Final equilibration
# -------------------------------------------------------------------------
box.mdsim.equilibrate(20000, 
                      timestep, 
                      0, 
                      'langevin', 
                      scale=False, 
                      output_steps=100,
                      dump=dmp)

# -------------------------------------------------------------------------
# Heating
# -------------------------------------------------------------------------
box.mdsim.equilibrate(20000, 
                      timestep, 
                      1.0,
                      'langevin', 
                      scale=False, 
                      output_steps=100,
                      dump=dmp)

# -------------------------------------------------------------------------
# Quenching
# -------------------------------------------------------------------------
box.mdsim.equilibrate(20000, 
                      timestep, 
                      0.5,
                      'langevin', 
                      scale=False, 
                      output_steps=100,
                      dump=dmp)

# -------------------------------------------------------------------------
# Run
# -------------------------------------------------------------------------
box.mdsim.run(folder="fail2", lammps_path="~/Research/lammps/src/lmp_mpi", mpi=18)

view_path = "~/ovito-basic-3.5.4-x86_64/bin/ovito"
box.mdsim.view(view_path, "./fail2/dump.int_settings.in_0.cfg")
