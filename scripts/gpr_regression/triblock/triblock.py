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
box_size = 85.0
cellnums = 70
t0 = time.time()    
box = PolyLattice(box_size, cellnums)

dmp = 10000

scft_calc = True
t1 = time.time()
print(f"Box initialised. Time taken: {t1-t0} seconds.")

box.interactions.newType("a", 0.5,
                         (1.0, 1.0, 1.5))

box.interactions.newType("b", 0.51,
                         (1.0, 0.5, 1.5),
                         ('a,b', (1.0, 0.2, 1.5)))

#----------------------------------------------------------------------------------------
# DATA-SET GENERATION:
#----------------------------------------------------------------------------------------
# set details:
dir = "STRAIN_DATASET"
if os.path.exists(dir):
    shutil.rmtree(dir)
os.makedirs(dir)

# walk details
num_walks = 62 # 100 # number of walks in a box
size = 3000 # size of the chain
rw_kval = 30
rw_cutoff = 1.5
rw_epsilon = 1.0
rw_sigma = 1.0
timestep = 0.005

# 48.33904695510864 seconds

# mechanics
# relaxation = 2000
# num_relax = 5
# deform_steps = 10000

iterations = 10

relaxation = 2500
num_relax = 4
shrink_steps = 100000
equib = 100000
deform_steps = 100000

for it in range(iterations):
    bsize = np.round(0.8+it*0.02, 5)
    asize = np.round((1-bsize)/2, 5)
    
    print(f"Block copolymer fraction: {asize} {bsize} {asize}")
    
    t0 = time.time()
    if scft_calc == True:
        box.meanfield.parameters("test",
                                 [int(0.5*cellnums), int(0.5*cellnums), int(0.5*cellnums)],
                                 60.0,
                                 'cubic',
                                 1.0e-2,
                                 'I m -3 m',
                                 [
                                     [("a", 1.0, asize), ("b", 1.0, bsize), ("a", 1.0, asize)]
                                 ],
                                 cell_param=1.923202,
                                 error_max=1e-5,
                                 max_itr=400,
                                 ncut=95,)

        box.meanfield.model_field("test_model", 0, [[0.0, 0.0, 0.0], 
                                            [0.5, 0.5, 0.5]])

        path_pscf = "/home/wmg/phrmzc/Research/research-code/pscf/bin"
        box.meanfield.run(path_pscf)

        t1 = time.time()
        print(f"PSCF calculation concluded. Time taken: {t1-t0} seconds.")

    #----------------------------------------------------------------------------------------
    # DENSITY ASSIGNMENT
    #----------------------------------------------------------------------------------------
    box.meanfield.density_file("rgrid", [int(0.5*cellnums), int(0.5*cellnums), int(0.5*cellnums)])

    print(f"Density file read in. Time taken: {t1 - t0}")

    #----------------------------------------------------------------------------------------
    # RANDOM WALK ASSIGNMENT
    #----------------------------------------------------------------------------------------

    # copolymer specification
    copolymer = []
    for i in range(int(size*asize)):
        copolymer.append('a')
    for i in range(int(size*bsize)):
        copolymer.append('b')
    for i in range(int(size*asize)):
        copolymer.append('a')

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
                          retraction_limit = 5,
                          sequence_range=int(0.125*size))
        t1 = time.time()
        print(f"Walk {i} complete. Time taken: {t1-t0} seconds.")
        total_time+= t1-t0

    print("Dead walks: {box.dead_walks}")

    while box.dead_walks > 0:
        print("Attempting resurrection of walks.")
        reattempts = box.dead_walks
        box.dead_walks = 0 
        for i in range(reattempts):
            box.uniform_chain(size,
                              rw_kval,
                              rw_cutoff,
                              rw_epsilon,
                              rw_sigma,
                              termination='soften',
                              bead_sequence=copolymer,
                              srate=0.99,
                              suppress=False,
                              sequence_range = 5,
                              retraction_limit = 0,
                              danger=1.12)


    print(f"Walks complete. Total time: {total_time} seconds")

    t0 = time.time()
    box.mdsim.structure(f"structure-{bsize}.in")
    box.mdsim.settings(f"settings-{bsize}.in", nskin=4.0, nlist=[10,10000]) 

    # # minimise to iron out issues
    box.mdsim.minimize(1e-10, 1e-10, 100000, 1000000)

    squash = [-1.2e-3, -1.2e-3, -1.2e-3]
    box.mdsim.deform(shrink_steps,
                     timestep,
                     [['x', 52],
                      ['y', 52],
                      ['z', 52]],
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

    box.mdsim.deform(deform_steps, 
                     timestep,
                     [['x', 300]],
                     0.29,
                     reset=False,
                     dump=dmp,
                     description="deform")

    box.mdsim.run(folder=f"dataset-{bsize}", lammps_path="~/Research/lammps/src/lmp_mpi", mpi=10)

    os.rename(f"structure-{bsize}.in", f"dataset-{bsize}/structure-{bsize}.in")
    os.rename(f"settings-{bsize}.in", f"dataset-{bsize}/settings-{bsize}.in")
    os.rename(f"dataset-{bsize}", f"STRAIN_DATASET/dataset-{bsize}")

    box.reset(keep_types=True)
