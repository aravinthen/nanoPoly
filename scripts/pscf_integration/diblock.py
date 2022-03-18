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
box_size = 90.0
t0 = time.time()    
box = PolyLattice(box_size, cellnums=30)
dmp = 100
scft_calc = True

t1 = time.time()


box.interactions.newType("a", 0.5,
                         (1.0, 1.0, 1.5))

box.interactions.newType("b", 0.55,
                         (1.0, 0.5, 1.5),
                         ('a,b', (0.1, 0.2, 1.5)))


if scft_calc == True:
    box.meanfield.parameters("test",
                             [30,30,30],
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

