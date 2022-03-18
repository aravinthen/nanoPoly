import numpy as np
import time
import sys
import os
import shutil

sys.path.insert(0, '../../../main')

from mdsim import MDSim
from poly import PolyLattice
from analysis import Check
from meanfield import MeanField

print("NANOPOLY SIMULATION")
box_size = 90.0
cellnums = 70
t0 = time.time()    
box = PolyLattice(box_size, cellnums)

dmp = 1000

scft_calc = False
t1 = time.time()
print(f"Box initialised. Time taken: {t1-t0} seconds.")

box.interactions.newType("a", 0.5,
                         (1.0, 1.0, 1.5))

box.interactions.newType("b", 0.51,
                         (1.0, 0.5, 1.5),
                         ('a,b', (1.0, 0.2, 1.5)))

t0 = time.time()
if scft_calc == True:
    box.meanfield.parameters("test",
                             [int(0.5*cellnums), int(0.5*cellnums), int(0.5*cellnums)],
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

t1 = time.time()
print(f"PSCF calculation concluded. Time taken: {t1-t0} seconds.")

#----------------------------------------------------------------------------------------
# DATA-SET GENERATION:
#----------------------------------------------------------------------------------------
# set details:
dir = "TEMPERATURE_DATASET"
if os.path.exists(dir):
    shutil.rmtree(dir)
os.makedirs(dir)

# walk details
num_walks = 125 # 100 # number of walks in a box
size = 1000 # size of the chain
rw_kval = 30
rw_cutoff = 1.5
rw_epsilon = 1.0
rw_sigma = 1.0
timestep = 0.005
copolymer = []

# 48.33904695510864 seconds

# mechanics
# relaxation = 2000
# num_relax = 5
# deform_steps = 10000

iterations = 15

base = [5*0.5e-3,
        5*(-2.5e-4),
        5*(-2.5e-4)]

relaxation = 2500
num_relax = 4
shrink_steps = 90000
equib = 80000

# copolymer specification
for i in range(int(size*0.25)):
    copolymer.append('a')
for i in range(int(size*0.75)):
    copolymer.append('b')

for it in range(iterations):
    stretch_val = it
    #----------------------------------------------------------------------------------------
    # DENSITY ASSIGNMENT
    #----------------------------------------------------------------------------------------
    box.meanfield.density_file("rgrid", [int(0.5*cellnums), int(0.5*cellnums), int(0.5*cellnums)])

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
                          srate = 0.99, 
                          suppress= False,
                          danger = 1.03,
                          initial_failures=15000,
                          walk_failures=15000,
                          sequence_range=int(0.125*size))
        t1 = time.time()
        print(f"Walk {i} complete. Time taken: {t1-t0} seconds.")
        total_time+= t1-t0

    print(f"Walks complete. Total time: {total_time} seconds")

    t0 = time.time()
    box.mdsim.structure(f"structure-{stretch_val}.in")
    box.mdsim.settings(f"settings-{stretch_val}.in", nskin=4.0, nlist=[10,10000]) 

    # # minimise to iron out issues
    box.mdsim.minimize(1e-10, 1e-10, 100000, 1000000)

    squash = [-1.2e-3, -1.2e-3, -1.2e-3]
    box.mdsim.deform(shrink_steps, 
                     timestep,
                     squash,
                     0.0,
                     reset=False,
                     dump=dmp,
                     description="deform")

    box.mdsim.minimize(1e-10, 1e-10, 100000, 1000000)
    
    box.mdsim.equilibrate(equib, 
                          timestep, 
                          0.29, 
                          'langevin', 
                          scale=False, 
                          output_steps=100,
                          dump=dmp)

    stretch = [(1+0.2*it)*b for b in base]
    deform_steps = abs(int(350/stretch[0]))
    print(f"Stretching for {deform_steps}.")    
    box.mdsim.deform(deform_steps, 
                     timestep,
                     stretch,
                     0.29,
                     reset=False,
                     dump=dmp,
                     description="deform")

    box.mdsim.run(folder=f"dataset-{stretch_val}", lammps_path="~/Research/lammps/src/lmp_mpi", mpi=10)

    os.rename(f"structure-{stretch_val}.in", f"dataset-{stretch_val}/structure-{stretch_val}.in")
    os.rename(f"settings-{stretch_val}.in", f"dataset-{stretch_val}/settings-{stretch_val}.in")
    os.rename(f"dataset-{stretch_val}", f"STRAIN_DATASET/dataset-{stretch_val}")

    box.reset(keep_types=True)
