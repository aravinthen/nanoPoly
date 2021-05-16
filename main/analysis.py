# Program Name: analysis.py
# Author: Aravinthen Rajkumar
# Description: Analysis routines for data produced by poly.
#              This module is meant for pre-simulation processing, acting on the polylattice
#              box object itself.

import numpy as np
import time

class Check:
    """
    Routines in Check are used to 
    """
    def __init__(self, polylattice):
        self.polylattice = polylattice


    # ERROR CHECKING ROUTINES
    # allows the user to 
    def distance_check(self,):
        "Checks to see if any of the beads are too close inside a simulation."
        shortest_distance = self.polylattice.cellside*self.polylattice.cellnums
        errors = 0
        zeros = 0
        mean_occ = 0
        print(f"\nNumber of cells per side: {self.polylattice.cellnums}")
        print(f"-------------------------------------------------------------------------------")
        print(f"Distance Errors:")
        print(f"G.num_b\tG.num_n\tWalk_b\tWalk_n\twnum_b\twnum_n\tDistance")

        all_data = self.polylattice.walk_data() + self.polylattice.cl_data()
        for bead in all_data:
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


class Percolation:
    """
    Converts a polymer network into a directed graph.
    """
    def __init__(self, polylattice):
        self.polylattice = polylattice
        
    def blaze(self, start):
        # SO BEAUTIFUL!
        # Algorithm description:
        # INPUT: the STRUCTURE COORDINATES of bead: [a, b], where a: random walk number
        #                                                         b: bead number
        # OUTPUT: all the possible directed paths that can be generated from this bead.        
        # new paths are added to [paths] as the algorithm progresses


        # The crosslinkers play an important part in this algorithm
        # They introduce *new* branching points.
        paths = [[[start[0], start[1], 0]]] # affixes a burning number to the starting bead
        
        path_index = 0
        num_paths = 1

        crosslinks = self.polylattice.crosslinks_loc        
        while path_index < num_paths:
            blazing = True
            burn_num = 0
            while blazing:
                path = paths[path_index]            
                # the next beads will all be stored in the below list
                burn_num+=1
                next_beads = []
                current = path[-1]
                current_walk = self.polylattice.walk_data(current[0])
                bead = current_walk[current[1]]

                # crosslinkers: a slightly more complex treatment than below.
                # check if the  bead is connected to a crosslinker.
                # if so, add the crosslinker as well as the connecting bead as new entries to the path.

                for link in crosslinks:
                    if bead == link[0]:
                        link_path = path.copy()
                        link_path.append([link[1][0], link[1][1], burn_num])
                        link_path.append([link[2][0], link[2][1], burn_num+1])
                        paths.append(link_path)
                        crosslinks.remove(link)
                        num_paths+=1
                        
                    if bead == link[2]:
                        link_path = path.copy()
                        link_path.append([link[1][0], link[1][1], burn_num])
                        link_path.append([link[0][0], link[0][1], burn_num+1])
                        paths.append(link_path)
                        crosslinks.remove(link)
                        num_paths+=1

                # beads along the same random walk will be stored here.
                walk_beads = []
                if current[1]+1 > len(current_walk)-1:
                    walk_beads.append([current[0], current[1]-1, burn_num])
                elif current[1]-1 < 0:
                    walk_beads.append([current[0], current[1]+1, burn_num])
                else:
                    walk_beads.append([current[0], current[1]+1, burn_num])
                    walk_beads.append([current[0], current[1]-1, burn_num])

                # The bead must be checked against every bead already in the path.
                # This is vital, as it avoids circles.
                for candidate in walk_beads:
                    all_beads_so_far = [(i[0], i[1]) for i in path for path in paths]
                    if (candidate[0], candidate[1]) not in all_beads_so_far:
                        distance = np.linalg.norm(current_walk[candidate[1]][-1] - bead[-1])
                        if distance < 0.5*self.polylattice.boxsize:
                            next_beads.append(candidate)
                                                            
                # should have the next bead candidates in here.

                # terminate
                print(num_paths, path_index, burn_num)
                if len(next_beads) == 0:
                    path_index+=1
                    if path_index == num_paths:
                        blazing = False
                    else:
                        burn_num = paths[path_index][-1][2]                                        
                else:
                    path.append(next_beads[0])
                    # add the multiple candidates as new paths in the PATH list
                    if len(next_beads) > 1:
                        for coord in next_beads[1::]:
                            new_path = path[0:-1] # you don't have to copy this
                            new_path.append(coord)
                            paths.append(new_path)
                            num_paths+=1

                            
            return paths
