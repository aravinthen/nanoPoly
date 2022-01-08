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

print("NANOPOLY SIMULATION")
box_size = 10.0
t0 = time.time()    
box = PolyLattice(box_size, cellnums=30)
t1 = time.time()
    
box.interactions.newType("a", 0.5,
                         (0.1, 0.2, 1.5, (1, 1.0, 0.1)))

box.interactions.newType("b", 1.0,
                         (0.1, 0.5, 1.5, (1, 1.0, 0.1)),
                         ('a,b', (0.1, 0.2, 1.5, (1, 1.0, 0.1))))

box.meanfield.parameters("test",
                         [5,5,5],
                         20.0,
                         'cubic',
                         1.0e-2,
                         'I m -3 m',
                         [("a", 0.25), ("b", 0.75)],
                         cell_param=1.9,
                         error_max=1e-5,
                         max_itr=100,
                         ncut=80,)

box.meanfield.model_field("test_model", 0, [[0.0, 0.0, 0.0], 
                                            [0.5, 0.5, 0.5]])

path_pscf = "/home/u1980907/Documents/Academia/Research/Code/pscf/bin"
box.meanfield.run(path_pscf)
