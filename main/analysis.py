# Program Name: analysis.py
# Author: Aravinthen Rajkumar
# Description: Analysis routines for data produced by poly.
#              This module is meant for pre-simulation processing, acting on the polylattice
#              box object itself.



import numpy as np

class Analysis:    
    def __init__(self, polylattice):
        self.polylattice = polylattice

    def error_check(self,):
        "Checks to see if any of the beads are too close inside a simulation."
        shortest_distance = self.polylattice.cellside*self.polylattice.cellnums
        errors = 0
        zeros = 0
        mean_occ = 0
        print(f"\nNumber of cells per side: {self.polylattice.cellnums}")
        print(f"-------------------------------------------------------------------------------")
        print(f"Distance Errors:")
        print(f"G.num_b\tG.num_n\tWalk_b\tWalk_n\twnum_b\twnum_n\tDistance")
        
        for bead in self.polylattice.walk_data():
            error_count = 0
            nlist = self.polylattice.check_surroundings(bead[-1])
            occ = len(nlist)
            mean_occ += occ
            for neighbour in nlist:
                distance = np.linalg.norm(bead[-1] - neighbour[-1])
                if distance < self.polylattice.lj_sigma and distance != 0.0:
                    print(f"{bead[-2]}\t{neighbour[-2]}\t{bead[0]}\t{neighbour[0]}\t{bead[1]}\t{neighbour[1]}\t{distance}")
                    error_count+=1

            if error_count > 0:
                errors+=1
        print(f"-------------------------------------------------------------------------------")
        print(f"The number of errors in distance calculation: {errors}/{self.polylattice.num_beads}")
        print(f"The check_surroundings algorithm messes up {100*(errors)/self.polylattice.num_beads}% of the time.")
        
        print(f"The mean number of occupants in a surroundings check is {mean_occ/self.polylattice.num_beads}")

