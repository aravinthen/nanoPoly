# Program Name: 
# Author: Aravinthen Rajkumar
# Description:

import numpy as np
import time
import sys

sys.path.insert(0, '../../main')

from mdsim import MDSim
from poly import PolyLattice
from analysis import Check
from meanfield import MeanField


scft = False
print("NANOPOLY SIMULATION")
box_size = 80.0
cellnums = 70

t0 = time.time()    
box = PolyLattice(box_size, cellnums)
t1 = time.time()
    
box.interactions.newType("a", 0.5,
                         (0.1, 1.0, 1.5, (1, 1.0, 0.1)))

box.interactions.newType("b", 1.0,
                         (0.1, 0.5, 1.5, (1, 1.0, 0.1)),
                         ('a,b', (0.1, 0.2, 1.5, (1, 1.0, 0.1))))


b_frac = 0.8
a_frac = (1-b_frac)/2

if scft == True:
    box.meanfield.parameters("test",
                             [int(0.5*cellnums), int(0.5*cellnums), int(0.5*cellnums)],
                             60.0,
                            'cubic',
                             1.0e-2,
                             'I m -3 m',
                             [
                                 [("a", 1.0, a_frac), ("b", 1.0, b_frac), ("a", 1.0, a_frac)]
                             ],
                             cell_param=1.923202,
                             error_max=1e-5,
                             max_itr=400,
                             ncut=95,)

    box.meanfield.model_field("test_model", 0, [[0.0, 0.0, 0.0], 
                                                [0.5, 0.5, 0.5]])

    path_pscf = "/home/wmg/phrmzc/Research/research-code/pscf/bin"
    box.meanfield.run(path_pscf)


walk_vars = [i*100 for i in range(10,11)]
for walkv in walk_vars:
    #----------------------------------------------------------------------------------------
    # DENSITY ASSIGNMENT
    #----------------------------------------------------------------------------------------
    t0 = time.time()    
    box.meanfield.density_file("rgrid", [int(0.5*cellnums), int(0.5*cellnums), int(0.5*cellnums)])
    t1 = time.time()

    print(f"Density file read in. Time taken: {t1 - t0}")
    #----------------------------------------------------------------------------------------
    # RANDOM WALK ASSIGNMENT
    #----------------------------------------------------------------------------------------
    # following values determine the bonding of the random walks    
    size = walkv
    num_walks = int(125000/walkv)

    print(f"{num_walks} walks of size {size} will be simulated.")
    # size of the chain
    rw_kval = 30
    rw_cutoff = 1.5
    rw_epsilon = 1.0
    rw_sigma = 1.0

    # block = 100

    a_bead = int(size*a_frac)
    b_bead = int(size*b_frac)

    copolymer = []
    for i in range(a_bead):
        copolymer.append('a')
    for i in range(b_bead):
        copolymer.append('b')
    for i in range(a_bead):
        copolymer.append('a')

    print(a_bead)
    print(b_bead)
    print(a_bead+b_bead)

    total_time = 0
    for i in range(num_walks):
        t0 = time.time()
        box.uniform_chain(size,
                          rw_kval,
                          rw_cutoff,
                          rw_epsilon,
                          rw_sigma,
                          bead_sequence = copolymer,
                          sequence_range=int(0.9*a_bead),
                          meanfield = True)
        t1 = time.time()
        total_time+= t1-t0
        print(f"Walk {i} completed in {t1-t0} seconds. Total time elapsed: {total_time}")


    total_time+= t1-t0
    print(f"Walks complete. Total time: {total_time} seconds")

    t0 = time.time()
    box.mdsim.structure(f"struc-{walkv}.in")
    t1 = time.time()
    total_time = t1-t0
    print(f"Structure file created. Total time: {total_time} seconds.")

    box.mdsim.settings("test_settings.in", nskin=2.0, nlist=[10,10000]) 

#    view_path = "~/ovito-basic-3.5.4-x86_64/bin/ovito"
#    box.mdsim.view(view_path, f"struc{walkv}.in")

    box.reset(keep_types=True)
