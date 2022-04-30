import numpy as np
import time
import sys
import os
import shutil

sys.path.insert(0, '../../main')

from mdsim import MDSim
from poly import PolyLattice
from analysis import Check
from meanfield import MeanField

print("NANOPOLY SIMULATION")
box_size = 90.0
cellnums = 80
t0 = time.time()
box = PolyLattice(box_size, cellnums)
dmp = 1000

scft_calc = True
t1 = time.time()
print(f"Box initialised. Time taken: {t1-t0} seconds.")

box.interactions.newType("a", 0.5,
                         (1.0, 1.0, 1.5))

box.interactions.newType("b", 0.51,
                         (1.0, 0.5, 1.5),
                         ('a,b', (1.0, 0.2, 1.5)))

bsize = 0.8
asize = (1-bsize)/2

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
# walk details
num_walks = 50 # # number of walks in a box
size = 2500 # size of the chain
rw_kval = 30
rw_cutoff = 1.5
rw_epsilon = 1.0
rw_sigma = 1.0
timestep = 0.005

# equilibration details
relaxation = 2500
num_relax = 4
shrink_steps = 100000
equib = 50000

copolymer = []
# copolymer specification
for i in range(int(asize*size)):
    copolymer.append('a')
for i in range(int(bsize*size)):
    copolymer.append('b')
for i in range(int(asize*size)):
    copolymer.append('a')

total_rets = 10
strain_rate = 5e-3
start = 52
end = 300
ret = 200

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
                      termination='soften',
                      srate = 0.98, 
                      suppress= False,
                      danger = 1.03,
                      initial_failures=15000,
                      walk_failures=15000,
                      retraction_limit = 10,
                      sequence_range=int(0.9*asize*size))
    t1 = time.time()
    print(f"Walk {i} complete. Time taken: {t1-t0} seconds.")
    total_time+= t1-t0

print(f"Walks complete. Total time: {total_time} seconds")

t0 = time.time()
box.mdsim.structure(f"structure-{ret}.in")
box.mdsim.settings(f"settings-{ret}.in", nskin=4.0, nlist=[10,10000]) 

# # minimise to iron out issues
box.mdsim.minimize(1e-10, 1e-10, 100000, 1000000)

box.mdsim.deform(shrink_steps, 
                 timestep,
                 [['x', start],
                  ['y', start],
                  ['z', start]],
                 0.15,
                 reset=False,
                 dump=dmp,
                 description="Fix density.")

box.mdsim.minimize(1e-10, 1e-10, 100000, 1000000)

box.mdsim.equilibrate(equib, 
                      timestep, 
                      0.29, 
                      'npt',
                      pdamp=1,
                      scale=False, 
                      output_steps=100,
                      dump=dmp)


ret_steps = int((ret - start)/strain_rate)
box.mdsim.deform(ret_steps, 
                 timestep,
                 [['x', ret]],
                 0.29,
                 reset=False,
                 dump=dmp,
                 description="Deformation to {ret}")

box.mdsim.equilibrate(equib, 
                      timestep, 
                      0.29, 
                      'npt', 
                      scale=False, 
                      pdamp=0.1,
                      output_steps=100,
                      dump=dmp)

fsteps = int((end-start)/strain_rate)
box.mdsim.deform(fsteps, 
                 timestep,
                 [['x', end]],
                 0.29,
                 reset=False,
                 dump=dmp,
                 description="Final deformation to {end}")

box.mdsim.run(folder=f"dataset-{ret}", lammps_path="~/Research/lammps/src/lmp_mpi", mpi=10)

box.reset(keep_types=True)














