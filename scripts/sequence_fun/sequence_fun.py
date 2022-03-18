# Program Name: 
# Author: Aravinthen Rajkumar
# Description:

import numpy as np
import time
import sys
import os

sys.path.insert(0, '../../main')

from mdsim import MDSim
from poly import PolyLattice
from analysis import Check
from meanfield import MeanField

print("NANOPOLY SIMULATION")
box_size = 50.0
t0 = time.time()    
box = PolyLattice(box_size, cellnums=15)
dmp = 100
scft_calc = False

t1 = time.time()


box.interactions.newType("a", 0.5,
                         (1.0, 1.0, 1.5))

box.interactions.newType("b", 0.55,
                         (1.0, 0.5, 1.5),
                         ('a,b', (0.1, 0.2, 1.5)))


if scft_calc == True:
    box.meanfield.parameters("test",
                             [15,15,15],
                             25.0,
                             'cubic',
                             1.0e-2,
                             'I m -3 m',
                             [
                             [("a", 1.0, 0.25), ("b", 1.0, 0.75)]
                             ],
                             cell_param=1.923202,
                             error_max=1e-5,
                             max_itr=100,
                             ncut=95,)

    box.meanfield.model_field("test_model", 0, [[0.0, 0.0, 0.0], 
                                            [0.5, 0.5, 0.5]])

    path_pscf = "/home/wmg/phrmzc/Research/research-code/pscf/bin"
    box.meanfield.run(path_pscf)


#----------------------------------------------------------------------------------------
# DATA-SET GENERATION:
#----------------------------------------------------------------------------------------
# set details:
data_set = 10 # number of samples
os.mkdir("DATASET")

# walk details
num_walks = 5 # 225 # number of walks in a box
size = 1000 # size of the chain
rw_kval = 30
rw_cutoff = 1.5
rw_epsilon = 1.0
rw_sigma = 1.0
timestep = 0.005
copolymer = []

# mechanics
# relaxation = 2000
# num_relax = 5
# deform_steps = 10000

relaxation = 20
num_relax = 1
deform_steps = 10

# copolymer specification
for i in range(250):
    copolymer.append('a')
for i in range(750):
    copolymer.append('b')

for iteration in range(data_set):
    sequence_range = 20*(iteration+1)
    #----------------------------------------------------------------------------------------
    # DENSITY ASSIGNMENT
    #----------------------------------------------------------------------------------------
    box.meanfield.density_file("rgrid", [15,15,15])

    print(f"Density file read in. Time taken: {t1 - t0}")

    #----------------------------------------------------------------------------------------
    # RANDOM WALK ASSIGNMENT
    #----------------------------------------------------------------------------------------
    total_time = 0
    for i in range(num_walks):
        t0 = time.time()
        box.uniform_chain(size,
                          rw_kval,
                          rw_cutoff,
                          rw_epsilon,
                          rw_sigma,
                          bead_sequence = copolymer,
                          meanfield = True,
                          srate=0.99, 
                          suppress=False,
                          initial_failures=1000,
                          sequence_range=sequence_range,
                          walk_failures=5000)
        t1 = time.time()
        print(f"Walk {i} complete. Time taken: {t1-t0} seconds.")
        total_time+= t1-t0

    total_time+= t1-t0
    print(f"Walks complete. Total time: {total_time} seconds")

    t0 = time.time()
    box.mdsim.structure(f"structure-{sequence_range}.in")
    box.mdsim.settings(f"settings-{sequence_range}.in", nskin=4.0, nlist=[10,10000]) 

    # # minimise to iron out issues
    box.mdsim.minimize(1e-8, 1e-8, 100000, 1000000)

    squash = [-1.2e-2, -1.2e-2, -1.2e-2]
    box.mdsim.deform(9000, 
                     timestep,
                     squash,
                     0.1,
                     reset=False,
                     dump=dmp,
                     description="deform")

    for i in range(num_relax):
        box.mdsim.minimize(1e-8, 1e-8, 100000, 1000000)
        box.mdsim.equilibrate(relaxation, 
                              timestep, 
                              0.3, 
                              'langevin', 
                              scale=False, 
                              output_steps=100,
                              dump=dmp)

    stretch = [0.5e-3, -2.5e-4, -2.5e-4]

    box.mdsim.deform(deform_steps, 
                     timestep,
                     stretch,
                     0.3,
                     reset=False,
                     dump=dmp,
                     description="deform")

    box.mdsim.run(folder=f"dataset-{sequence_range}", lammps_path="~/Research/lammps/src/lmp_mpi", mpi=10)

    os.rename(f"structure-{sequence_range}.in", f"dataset-{sequence_range}/structure-{sequence_range}.in")
    os.rename(f"settings-{sequence_range}.in", f"dataset-{sequence_range}/settings-{sequence_range}.in")
    os.rename(f"dataset-{sequence_range}", f"DATASET/dataset-{sequence_range}")

    box.reset(keep_types=True)


