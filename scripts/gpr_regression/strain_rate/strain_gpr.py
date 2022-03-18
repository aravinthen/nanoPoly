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
dir = "STRAIN_DATASET"
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

iterations = 10
relaxation = 2500
num_relax = 4
shrink_steps = 100000
equib = 150000

# copolymer specification
for i in range(int(size*0.25)):
    copolymer.append('a')
for i in range(int(size*0.75)):
    copolymer.append('b')

for it in range(1, iterations+1):
    deform_steps = 100000*it
    strain = np.round(250/deform_steps, 4)
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
    box.mdsim.structure(f"structure-{strain}.in")
    box.mdsim.settings(f"settings-{strain}.in", nskin=4.0, nlist=[10,10000]) 

    # # minimise to iron out issues
    box.mdsim.minimize(1e-10, 1e-10, 100000, 1000000)

    box.mdsim.deform(shrink_steps, 
                     timestep,
                     [['x', 50],
                      ['y', 50],
                      ['z', 50]],
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

    print(f"Stretching for {deform_steps}.")    
    box.mdsim.deform(deform_steps, 
                     timestep,
                     [['x', 300]],
                     0.29,
                     reset=False,
                     dump=dmp,
                     description="Deformation at strain rate {strain} time^(-1) ")

    box.mdsim.run(folder=f"dataset-{strain}", lammps_path="~/Research/lammps/src/lmp_mpi", mpi=10)

    os.rename(f"structure-{strain}.in", f"dataset-{strain}/structure-{strain}.in")
    os.rename(f"settings-{strain}.in", f"dataset-{strain}/settings-{strain}.in")
    os.rename(f"dataset-{strain}", f"STRAIN_DATASET/dataset-{strain}")

    box.reset(keep_types=True)
