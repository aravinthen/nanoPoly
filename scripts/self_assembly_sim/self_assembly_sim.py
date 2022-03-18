import numpy as np
import time
import sys

sys.path.insert(0, '../../main')

from mdsim import MDSim
from poly import PolyLattice
from analysis import Check
from meanfield import MeanField

print("NANOPOLY SIMULATION")
box_size = 70.0
t0 = time.time()    
cellnums = 70
box = PolyLattice(box_size, cellnums)
dmp = 100
scft_calc = True
t1 = time.time()


box.interactions.newType("a", 0.5,
                         (0.0, 1.0, 1.5))

box.interactions.newType("b", 0.55,
                         (0.0, 0.5, 1.5),
                         ('a,b', (0.0, 0.2, 1.5)))
if scft_calc == True:
    box.meanfield.parameters(f"test",
                             [35,35,35],
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
                             ncut=95)

    box.meanfield.model_field(f"test_model", 0, [[0.0, 0.0, 0.0], 
                                            [0.5, 0.5, 0.5]])

    path_pscf = "/home/wmg/phrmzc/Research/research-code/pscf/bin"
    box.meanfield.run(path_pscf)

#----------------------------------------------------------------------------------------
# DENSITY ASSIGNMENT
#----------------------------------------------------------------------------------------
t0 = time.time()    
box.meanfield.density_file("rgrid", [35, 35, 35])
t1 = time.time()

print(f"Density file read in. Time taken: {t1 - t0}")

#----------------------------------------------------------------------------------------
# RANDOM WALK ASSIGNMENT
#----------------------------------------------------------------------------------------
# following values determine the bonding of the random walks
num_walks = 100
size = 2000
# size of the chain
rw_kval = 30
rw_cutoff = 1.5
rw_epsilon = 1.0
rw_sigma = 1.0
timestep = 0.005

# block = 100

copolymer = []
for i in range(int(0.25*size)):
    copolymer.append('a')
for i in range(int(0.75*size)):
    copolymer.append('b')

print(int(0.25*size))
print(int(0.75*size))

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
                      srate=0.85,
                      sequence_range = 75,
                      suppress=False)
    t1 = time.time()
    total_time+= t1-t0
    print(f"Walk {i} completed in {t1-t0} seconds. Total time elapsed: {total_time}")


total_time+= t1-t0
print(f"Walks complete. Total time: {total_time} seconds")

t0 = time.time()
box.mdsim.structure(f"test_structure{cellnums}.in")
box.mdsim.settings(f"test_settings{cellnums}.in", nskin=4.0, nlist=[10,10000]) 

# # minimise to iron out issues
# box.mdsim.minimize(1e-5, 1e-5, 10000, 100000)

# final equilibration
#box.mdsim.equilibrate(10000, 
#                      timestep, 
#                      0.4, 
#                      'langevin', 
#                      scale=False, 
#                      output_steps=100,
#                      dump=dmp)

# # heat to solid
# box.mdsim.equilibrate(20000, 
#                       timestep, 
#                       0.5,
#                       'langevin', 
#                       scale=False, 
#                       output_steps=100,
#                       dump=dmp)


# box.mdsim.run(folder="test")

view_path = "~/ovito-basic-3.5.4-x86_64/bin/ovito"
box.mdsim.view(view_path, "test_structure60.in")
 
