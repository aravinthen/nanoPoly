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
box = PolyLattice(box_size, cellnums=20)
t1 = time.time()
    
box.interactions.newType("a", 0.5,
                         (0.1, 0.2, 1.5, (1, 1.0, 0.1)))

box.interactions.newType("b", 1.0,
                         (0.1, 0.5, 1.5, (1, 1.0, 0.1)),
                         ('a,b', (0.1, 0.2, 1.5, (1, 1.0, 0.1))))

box.meanfield.parameters("test",
                         "test_guess",
                         [5,5,5],
                         2.21,
                         'cubic',
                         0.0001,
                         '2 3 5 2',
                         [("a", 0.1), ("b", 0.9), ("a", 0.1)])

box.meanfield.model_field("test_model", 1, [[0.0, 0.0, 0.0], 
                                            [0.5, 0.5, 0.5]])