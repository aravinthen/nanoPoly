# Program Name: polylattice.py
# Author: Aravinthen Rajkumar
# Description: Contains all the code necessary for the following tasks:
#              1. Generating empty lattices for random-walks
#              2. Populating lattices with polymeric random walks.
#              3. Populating lattice cells with cross-linkers.
#              4. Scanning each lattice cell for cross-linking pairs.
#              5. Making validity checks.
#                 a) Are there enough cells for the lattices?
#                 b) Are the atoms too close to eachother?

import numpy as np
import math as m
import time
import random

import sys
sys.path.append(".")

from mdsim import MDSim
from analysis import Check, Percolation
from meanfield import MeanField
from structure import Structure

class PolyLattice:
    """
    The PolyLattice class initialises an array composed of cells.
    Each cell is identified by it's position and a number of attributes.
    Attributes for a PolyLattice: 1. cellside = sidelength of induvidual cell
                                  2. cellnum = the number of cells in each dimensionpp
                                  3. celltotal = total number of cells.
                                  4. Cells = list of cells in lattice. each list element is Cell class

    Attributes for a Cell:        1. i, j, k, ijk: indices (and tuple of indices).
                                  2. position: the position of cell origin relative to lattice origin
                                  3. cellsides: inherited from PolyLattice. Same as cellside
                                  4. forbidden: BOOLEAN -  Cannot host atoms if this is valued True
    """
    
    md_attr = MDSim
    che_attr = Check
    per_attr = Percolation
    scf_attr = MeanField
    str_attr = Structure
    
    class Cell:
        def __init__(self, index1, index2, index3, position, cellsides):            
            """
        index1, index2, index3: Index of the cell
            position: position of the origin of the cell, relative to the lattice origin (0,0,0)
            cellsides: inherited (but not in the OOP way) from the lattice.
            """
            self.i, self.j, self.k = index1, index2, index3
            self.ijk = (index1, index2, index3) # used for index identification function
            self.position = position
            self.cellsides = cellsides # same as that specified for lattice.
            # sets a unique ID for the crosslinker bead
            
            # if the cell, for any reason, is not allowed to have beads or crosslinkers inside.
            self.forbidden = False

            # a list of densities per bead.
            # These must be calculated via self-consistent field theory.
            self.densities = []
            
            # only one crosslinker allowed in every cell.        
            self.cl_pos = None
            # multiple beads allowed in a cell.
            self.beads = []

            self.energy = 0.0 # used to populate cell

            
    def __init__(self, boxsize, cellnums=1.0):
        """        
        boxsize: length of a box in one dimension
        cellnums: Number of cells in one-dimension.
        density: the rough density of the box
        """
        self.boxsize = boxsize 
        self.cellnums = cellnums
        self.cellside = self.boxsize/self.cellnums
        self.celltotal = self.cellnums**3
        
        self.crossings = np.zeros((3, 1)) # number of crossings in each direction, [x, y, z]
                                          # encapsulates boundary conditions

        self.retractions = 0 # the number of times a chain has had to retract in the course of a random walk
        self.dead_walks = 0

        # attached libraries
        self.mdsim = MDSim(self) # the simulation class
        self.check = Check(self) # the Check subclass
        self.percolation = Percolation(self) # the Percolation class
        self.meanfield = MeanField(self)
        
        # initializing the interactions class 
        # this is initialised as it's own class due to legacy code
        self.interactions = Structure.Interactions(self.cellside)

        # bond details
        self.bonds = None # 

        # crosslink details
        self.crosslinks_loc = []
        # unbonded crosslinks
        self.cl_unbonded = False # Will be set to true once unbonded crosslinks are added
        self.cl_bonding = None # saves the bond configuration for use in the simulation file
        
        self.random_walked = False
        # note: For info dictionaries the starting values all begin at zero.
        #       This is so that the beads are given the correct values for atom number in the
        #       simulation directory.
        
        self.walkinfo = {0:0}  # this is supposed to store the walk as well as the number of
                               # beads in that walk.
                               # used to control the bond writing process in the simulation class
        self.uclinfo = {0:0}   # the same as above, but for unbonded crosslink structures
        self.bclinfo = {0:0}   # the same as above, but for unbonded crosslink structures


        self.nanostructure = None
        
        # counts
        self.num_uclstructs = 0 # the number of unbonded crosslink structures
        self.num_uclbeads = 0
        
        self.num_bclstructs = 0 # the number of bonded crosslink structures present
        self.num_bclbeads = 0 

        self.num_walks = 0
        self.num_walk_beads = 0
        
        self.num_grafts = 0

        self.graft_ids = []
        self.graft_beads = 0
        self.graft_coords = [] # Contains the coordinates of the bead the graft is attached to,
                               # as well as the first bead on that grafted chain. This is ONLY
                               # used to build the relevant bond!

        self.graft_fail = False # this is to 

        # global counts
        self.num_bonds = 0        
        self.num_beads = 0

        # this bit of code builds the cells within the lattice.
        self.Cells = []

        # when the simulation library is called, the structure is considered complete.
        self.structure_ready = False
        
        # for i in range(self.cellnums):
        #     for j in range(self.cellnums):
        #         for k in range(self.cellnums):
        #             # fastest index
        #             cell_positions = np.array([round(self.cellside*i, 16),
        #                                        round(self.cellside*j, 16),
        #                                        round(self.cellside*k, 16)])
        #             self.Cells.append(self.Cell(i, j, k, cell_positions, self.cellside))

        index_list = ((i,j,k)
                      for i in range(self.cellnums)
                      for j in range(self.cellnums)
                      for k in range(self.cellnums))
        
        for index in index_list:
            cell_positions = np.array([round(self.cellside*index[0], 16),
                                       round(self.cellside*index[1], 16),
                                       round(self.cellside*index[2], 16)])
            self.Cells.append(self.Cell(index[0], index[1], index[2], cell_positions, self.cellside))
            
                
    # -------------------------------- METHODS --------------------------------
    # The following methods are included:
    # 0a. Retrieve a cell based on index.
    # 0b. Retrieve a cell based on position.
    # 1.  Populate the lattice with a random walk. Should be repeatable.
    # 2.  Populate the lattice with crosslinkers.
    # 3.  Checks cross-linkers for bonds.
    # ... a bunch more. Oh boy, we need to update this...
    
    # LATTICE METHODS ---------------------------------------------------------
    
    def index(self, cell_list):
        """
        USE THIS TO ACCESS THE CELLS IN THE LATTICE!
        """
        if any(cell_list) < 0 or any(cell_list) >= self.cellnums:
            raise IndexError("Cell index out of range.")
        
        cell=self.Cells[self.cellnums*self.cellnums*cell_list[0]+self.cellnums*cell_list[1]+cell_list[2]]
        
        return cell
        
    def which_cell(self, position):
        """
        Given a position, returns the index it'd be in as a list.
        """
        index = []
        for i in position:
            index.append(m.floor(i/self.cellside))
        return np.array(index)

    def check_surroundings(self, position):
        """
        gets a list of all the surrounding beads
        CHECKS SURROUNDINGS OF A POSITION, _NOT_ A CELL INDEX!
        This works by 
        1. Getting the index of the cell we're in (position)
        2. Getting all surrounding indices.
        3. Removing all the indices that aren't valid (less than 0, greater than cellnums)
        4. Accessing each cell marked by that index
        5. Appending the bead list to a main list and outputting the mainlist.
           Parameter: position
        """
        
        cell_index = self.which_cell(position)

        surround_ind = ([(cell_index[0]+i)%self.cellnums,
                         (cell_index[1]+j)%self.cellnums,
                         (cell_index[2]+k)%self.cellnums] for i in range(-1,2) for j in range(-1,2) for k in range(-1,2))

        return [bead for cell in surround_ind for bead in self.index(cell).beads]

    def reset(self, keep_types=False):
        """Completely resets a box into it's initial state"""
        
        # save the important arguments
        cbox = self.boxsize
        ccellnum = self.cellnums
        ccellnum = self.cellnums

        if keep_types == True:
            old_interactions = self.interactions

        # full reset of parameters
        self.boxsize = cbox
        self.cellnums = ccellnum
        self.cellside = self.boxsize/self.cellnums
        self.celltotal = self.cellnums**3        
        self.crossings = np.zeros((3, 1)) # number of crossings in each direction, [x, y, z]
                                          # encapsulates boundary conditions

        # attached libraries
        self.mdsim = MDSim(self) # the simulation class
        self.check = Check(self) # the Check subclass
        self.percolation = Percolation(self) # the Percolation class
        self.meanfield = MeanField(self)
        
        # initializing the interactions class 
        # this is initialised as it's own class due to legacy code
        if keep_types == False:
            self.interactions = Structure.Interactions(self.cellside)
        else:
            self.interactions = old_interactions
        # bond details
        self.bonds = None # 

        # crosslink details
        self.crosslinks_loc = []
        # unbonded crosslinks
        self.cl_unbonded = False # Will be set to true once unbonded crosslinks are added
        self.cl_bonding = None # saves the bond configuration for use in the simulation file
        
        self.random_walked = False
        # note: For info dictionaries the starting values all begin at zero.
        #       This is so that the beads are given the correct values for atom number in the
        #       simulation directory.
        
        self.walkinfo = {0:0}  # this is supposed to store the walk as well as the number of
                               # beads in that walk.
                               # used to control the bond writing process in the simulation class
        self.uclinfo = {0:0}   # the same as above, but for unbonded crosslink structures
        self.bclinfo = {0:0}   # the same as above, but for unbonded crosslink structures


        self.nanostructure = None
        
        # counts
        self.num_uclstructs = 0 # the number of unbonded crosslink structures
        self.num_uclbeads = 0
        
        self.num_bclstructs = 0 # the number of bonded crosslink structures present
        self.num_bclbeads = 0 

        self.num_walks = 0
        self.num_walk_beads = 0
        
        self.num_grafts = 0
        self.graft_beads = 0
        self.graft_coords = [] # Contains the coordinates of the bead the graft is attached to,
                               # as well as the first bead on that grafted chain. This is ONLY
                               # used to build the relevant bond!

        # global counts
        self.num_bonds = 0        
        self.num_beads = 0
    
        # this bit of code builds the cells within the lattice.
        self.Cells = []

        # when the simulation library is called, the structure is considered complete.
        self.structure_ready = False
        
        # for i in range(self.cellnums):
        #     for j in range(self.cellnums):
        #         for k in range(self.cellnums):
        #             # fastest index
        #             cell_positions = np.array([round(self.cellside*i, 16),
        #                                        round(self.cellside*j, 16),
        #                                        round(self.cellside*k, 16)])
        #             self.Cells.append(self.Cell(i, j, k, cell_positions, self.cellside))

        index_list = ((i,j,k)
                      for i in range(self.cellnums)
                      for j in range(self.cellnums)
                      for k in range(self.cellnums))
        
        for index in index_list:
            cell_positions = np.array([round(self.cellside*index[0], 16),
                                       round(self.cellside*index[1], 16),
                                       round(self.cellside*index[2], 16)])
            self.Cells.append(self.Cell(index[0], index[1], index[2], cell_positions, self.cellside))

        print("Box reset successfully.")


    #-----------------------------------------------------------------------------------------------
    # # # # # # # # # # # # # # # # # # # # # KEY ALGORITHM # # # # # # # # # # # # # # # # # # # # 
    #-----------------------------------------------------------------------------------------------
    def randomwalk(self, numbeads, Kval, cutoff, energy, sigma, bead_sequence, mini=1.12234,        
                   style='fene', phi=None, theta=None, 
                   cell_num=None, cell_list=None, 
                   starting_pos=None,
                   meanfield=False,
                   end_pos=None,
                   soften=True, srate=0.99, suppress=False, danger = 1.0,
                   widths=None, 
                   depths=None,
                   retraction_limit = 10,
                   termination='soften', initial_failures=10000, walk_failures=10000,
                   history=False):
        """index_c
        Produces a random walk.
        If cell_num argument is left as None, walk sprouts from a random cell.
        Otherwise, walk begins at specified cell.
        PARAMETER - beads: number of beads 
        PARAMETER - Kval: FENE potential parameter
        PARAMETER - cutoff: the extension limiter for the chain
        PARAMETER - sigma:  The distance value of the LJ potential.
                            This is *not* the distance between the atoms.
        PARAMETER - energy: the energy of the bond in LJ units.
        PARAMETER - phi: the torsion angle of the system.
        PARAMETER - theta: the bond angle of the system.
        PARAMETER - cell_num: The number of the cell that the random walk begins in.
                              Defaulted as None, but this can be set to an index.        
                              MUST be a list.
        PARAMETER - cell_list: a list of good positions to start the random walk.
        PARAMETER - starting_pos
        PARAMETER - bead_types: defines and identifies the beads and their types.
                                MUST BE A DICTIONARY.
        OPTION -    termination: THREE OPTIONS - a) None, b) Retract and c) Break
        OPTION -    grafting: used if the generated random walk will be grafted onto something.
        OPTION -    srate: the rate at which the minimum distance requirement softens
        OPTION -    sequence_range: the range at which a bead's type is selected when running mc-biased random walks
        OPTION -    history: prints the density difference history of a into a file. 
                             used for visualisation purposes.
                    
        Notes: Nested functions prevent the helper functions from being used outside of scope.       
        """        
        
        if self.structure_ready == True:
            raise EnvironmentError("Structures cannot be built or modified when simulation procedures are in place.")
        
        ID = self.num_walks + 1
        if termination == "None":
            print("Warning: if this random walk is unsucessful, the program will terminate.")

        if self.meanfield.density != True and meanfield != False:
            raise EnvironmentError("Density file has not been assigned.")

        if self.meanfield.density == True and cell_list==None:
            print("\nWARNING: It is highly recommended to use density_search() to find good positions to start")
            print("the random walk. This will typically provide better results.\n")

        if srate == 1.0:
            print("Setting srate to unity means that the softening rate of the potential will stay the same.")
            print("This will probably lead to the program getting stuck. Rerun the program with an s-rate")
            print("that is lower than unity.")
            print("(This particular mistake once cost the author of this program almost seventeen hours of")
            print("compute time.)")
            raise EnvironmentError("Program terminated.")

        def rand_distance(length, dimension):
            """
            Used to generate freely jointed chain models. This is the default option.
            produces a vector with fixed length and random direction. 
            length: fixed length of the vector.
            dimension: self-explanatory
            """
            x = np.random.randn(dimension, 3)
            magnitude = np.linalg.norm(x)

            return length*(x/magnitude)
        
        def new_position(position, length, dimension, maxdis, phi=None, theta=None):
            """
            NOTE! This function can be used to encode periodic boundary conditions.
                  Should this perhaps be an optional thing?
            """
            # need to flip between configurations of the angles
            flipper = [-1, 1]
            flip1 = random.sample(flipper,1)[0] # for theta
            flip2 = random.sample(flipper,1)[0] # for phi
            
            if (phi == None and theta == None):            
                new_pos = np.array(position) + rand_distance(length, dimension)
                new_pos = new_pos[0]
            elif (phi != None and theta != None):
                theta = flip1*theta
                phi = flip2*phi
                x = length*m.cos(theta)*m.cos(phi) 
                y = length*m.sin(theta)*m.sin(phi)
                z = length*m.cos(theta)
                new_pos = np.array(position) + np.array([x, y, z])
            else:
                if phi==None:
                    phi = np.random.uniform(-m.pi, m.pi)
                else:
                    theta = np.random.uniform(-m.pi, m.pi)

                theta = flip1*theta
                phi = flip2*phi
                                    
                x = length*m.cos(theta)*m.cos(phi) 
                y = length*m.sin(theta)*m.sin(phi)
                z = length*m.cos(theta)
                
                new_pos = np.array(position) + np.array([x, y, z])

            # Boundary conditions
            for i in range(len(new_pos)):
                if new_pos[i] < 0:
                    new_pos[i] = maxdis + new_pos[i]
                if new_pos[i] > maxdis:
                    new_pos[i] = new_pos[i] - maxdis
                
            return new_pos

        def forced_new_position(end_pos, bead_number, total_beads,
                                position, length, dimension, maxdis, phi=None, theta=None):
            """ used to direct the random walk to the end position."""
            
            # calculate initial new position vector:            
            new_pos = new_position(position, length, dimension, maxdis, phi, theta)

            # calculate forcing direction
            forcing_dir = end_pos - new_pos
            full_distance = np.linalg.norm(forcing_dir)
            forcing_vec = forcing_dir/full_distance
            
            # forcing = []
            # if length*(total_beads-bead_number) <= full_distance:
            #     forcing_mag = np.random.uniform(0.9, 1.0)
            #     print(bead_number, forcing_mag)
            #     forcing.append(forcing_mag)
            # else:
            measure = full_distance/((total_beads-bead_number)*length)
            mean = (2**7)*(measure - 0.5)**7            
            sigma = 1.0
            
            forcing_mag = np.random.normal(mean)
            if (measure > 0.2) and (measure < 0.8):
                forcing_mag += 0.5
                print("bing!")
                
            print(measure, mean, forcing_mag)
                
            # print(bead_number, forcing_mag)
            # forcing.append(forcing_mag)

            # print(mean, bead_number, forcing_mag[0])

            # add and rescale the forced direction
            new_pos = new_pos + forcing_mag*forcing_vec
            new_pos = length*(new_pos-position)/np.linalg.norm(new_pos - position) + position

            for i in range(len(new_pos)):
                if new_pos[i] < 0:
                    new_pos[i] = maxdis + new_pos[i]
                if new_pos[i] > maxdis:
                    new_pos[i] = new_pos[i] - maxdis
            
            return new_pos
            
        #----------------------------------------------------------------------------------------------
        def bcp_analyze(bcp):
            # returns the full details of the block
            # index1: the bead types.
            # index2: the indices of the block

            beadtypes = list[set(bcp)]

            # stores the information about the current block of the system
            block_indices = []

            number_of_blocks = 0

            # changes along the system as new blocks are found.
            first_index = 0

            initial_type = bcp[0]
            final_type = None

            for index in range(len(bcp)):
                if bcp[index] != initial_type:
                    new_type = bcp[index]
                    block_desc = [(initial_type, new_type), (first_index, index-1)]
                    block_indices.append(block_desc)
                    initial_type = new_type
                    first_index = index
                    number_of_blocks +=1

                if index == len(bcp)-1:
                    block_desc = [(initial_type, initial_type), (first_index, len(bcp)-1)]
                    block_indices.append(block_desc)

                    number_of_blocks += 1

            return block_indices, number_of_blocks 

        def allowed_range(block, max_range, initial_range, frac):
            # converts a bcp into a range of density differences.
            # block must have the following format:
            # a) [(type1, type2), (ind1, ind2)]
            # 
            # frac determines the point where the range is at the maximum. 
            # Within frac, beads should be able to travel wherever they want.
            # terminology: 
            #   the push region: the chain is pushed away into a region where the bead type is 
            #   the pull region: the chain is pulled towards the bead of the next type

            range_diff = max_range - initial_range

            ind1, ind2 = block[1][0], block[1][1]
            num_beads = ind2-ind1+1 # the number of beads in the block

            firstq = int(ind1+frac[0]*num_beads)
            secondq = int(ind1+frac[1]*num_beads)

            step1 = range_diff/(firstq - ind1) # the ascending step
            step2 = range_diff/(ind2 - secondq) # the descending step

            if block[0][0] == block[0][1]:
                rise = [0.0 for i in range(0, firstq-ind1)]
                mid = [max_range for i in range(0, secondq-firstq)]
                fall = [-1 for i in range(0, ind2-secondq+1)]
            else:
                rise = [round(initial_range + i*step1, 5) for i in range(0, firstq-ind1)]
                mid = [max_range for i in range(0, secondq-firstq)]
                fall = [round(max_range - i*step2,5) for i in range(0, ind2-secondq+1)]

            return rise+mid+fall, (firstq, secondq)
        
        #-------------------------------------------------------------------------------------------    

        # copolymer information: find the ranges 
        bcp_specs, num_blocks = bcp_analyze(bead_sequence)

        
        if history==True:
            history_file = open(f"history_{self.num_walks}", "w")

        if widths==None:
            widths =[[0.4,0.6] for i in range(num_blocks)]

        if depths==None:
            depths =[0.0 for i in range(num_blocks)]

        if len(widths) != num_blocks:
            print(f"Not enough widths entered for polymer chain [{len(widths)} instead of numblocks]")
            print("Reverting to default.")
            widths =[[0.4,0.6] for i in range(num_blocks)]

        weights = []
        boundaries = [] # the full list of boundaries based on the widths
        for count, block in enumerate(bcp_specs):
            type1 = block[0][0]
            type2 = block[0][1]
            alrange = allowed_range(block, 
                                    self.meanfield.max_dranges[(type1, type2)],
                                    depths[count],
                                    frac=widths[count])

            weights += alrange[0]
            boundaries.append(alrange[1])

        # bond type dictionary
        if self.bonds == None:
            self.bonds = {}
            self.bonds[0] = (Kval, cutoff, energy, sigma, style)
        else:
            repeat = 0
            for i in range(len(self.bonds)):
                if self.bonds[i] != (Kval, cutoff, energy, sigma, style):
                    repeat += 1
                else:
                    store = self.bonds[i]

            if repeat >= len(self.bonds):
                self.bonds[len(self.bonds)] = (Kval, cutoff, energy, sigma, style)
                
        bond_type = [i for i in range(len(self.bonds)) if self.bonds[i]==(Kval, cutoff, energy, sigma, style)][0]+1
        
        # put the types of being used in the random walk into the used types set
        for i in bead_sequence:
            if i not in self.interactions.used_types:
                self.interactions.used_types.append(i)

        if starting_pos != None:            
            invalid_start = True
            total_failure = False
            failure = 0
            while invalid_start:
                maxdis = self.cellside*self.cellnums
                for i in range(len(starting_pos)):
                    if starting_pos[i] < 0:
                        starting_pos[i] = maxdis + starting_pos[i]
                    if starting_pos[i] > maxdis:
                        starting_pos[i] = starting_pos[i] - maxdis

                    # nudge this out the way slightly
                    if starting_pos[i] == maxdis:
                        starting_pos[i] = 0.9999*maxdis

                starting_pos = np.array(starting_pos)            
                index_c = self.which_cell(starting_pos)                    
                neighbours = self.check_surroundings(starting_pos)
                issues = 0
                mc_issues = 0
                for j in neighbours:
                    index_n = self.which_cell(j[-1])
                    if self.cellnums-1 in np.abs(index_c - index_n):
                        period = np.array([i if abs(i) == (self.cellnums-1) else 0 for i in (index_n - index_c)])
                        real_j = -period*(self.cellnums/(self.cellnums - 1))*self.cellside + j[-1]
                    else:
                        real_j = j[-1]

                    # check the sigma distance
                    # the lhs is the distance between the bead and the neighbour
                    distance = np.round(np.linalg.norm(starting_pos - real_j), 5)

                    check = mini*self.interactions.return_sigma(bead_sequence[0], j[2])
                    if distance < check:
                        issues+=1
                        break # stop checking neighbours

                # monte carlo check NOT required, but will be a necessary component of all randomly generated positions.

                # issues should only occur the provided position is too close to another bead. 
                # softening can be employed here to make position selection easier.
                if issues>0:
                    invalid_start = True
                    failure+=1

                    if failure > initial_failures:
                        if soften == True:
                            total_failure = False
                            mini = srate*mini
                            failure = 0
                            issues = 0
                            if suppress == False:
                                print(f"Failure tolerance reached at random walk {self.num_walks}.")
                                print(f"Softening minimum requirement. Current minima: {mini}")

                        else:
                            total_failure = True
                            invalid_start = False

                else:
                    invalid_start = False
                    failure = 0

            if total_failure == True:
                print("Provided starting position is too close to another bead.")
                print("Please choose a more appropriate position.")
                raise Exception("Program terminated.")
                 
            bead_data = [ID,
                         0,
                         bead_sequence[0],
                         bond_type,
                         self.num_beads,
                         starting_pos]

            self.num_beads+=1
            self.num_walk_beads += 1 # the individual count for the number of beads specific to a walk
            self.index(self.which_cell(starting_pos)).beads.append(bead_data)
                                
        else:
            # trial for the initial position
            invalid_start = True
            total_failure = False
            failure = 0
            while invalid_start:   
                if cell_list != None:
                    cell_num = random.choice(cell_list)
                
                if cell_num != None:
                    current_cell = np.array(cell_num)
                else:
                    current_cell = np.array([random.randrange(0, self.cellnums),
                                             random.randrange(0, self.cellnums),
                                             random.randrange(0, self.cellnums)])

                cell_pos = self.index(current_cell).position
                cell_bound = cell_pos + self.cellside


                starting_pos = np.array([random.uniform(cell_pos[0], cell_bound[0]),
                                        random.uniform(cell_pos[1], cell_bound[1]),
                                        random.uniform(cell_pos[2], cell_bound[2])])                    
                index_c = self.which_cell(starting_pos)                    
                neighbours = self.check_surroundings(starting_pos)
            
                issues = 0
                
                for j in neighbours:
                    index_n = self.which_cell(j[-1])
                    if self.cellnums-1 in np.abs(index_c - index_n):
                        period = np.array([i if abs(i) == (self.cellnums-1) else 0 for i in (index_n - index_c)])
                        real_j = -period*(self.cellnums/(self.cellnums - 1))*self.cellside + j[-1]
                    else:
                        real_j = j[-1]

                    # check the sigma distance
                    # the lhs is the distance between the bead and the neighbour
                    distance = np.linalg.norm(starting_pos - real_j)
                    
                    if distance < mini*self.interactions.return_sigma(bead_sequence[0], j[2]):
                        issues+=1
                        break # stop checking neighbours
                    
                   

                # monte carlo check
                if meanfield == True:
                    density = self.index(index_c).densities[self.interactions.typekeys[bead_sequence[0]]-1]
                    
                    # limit detection
                    limit = self.interactions.limits[bead_sequence[0]]
                    if density < limit:
                        issues+=1 
                    
                    # mc acceptance
                    random_num = np.random.uniform(0,1)
                    if random_num > density:
                        issues+=1

                if issues > 0:                    
                    invalid_start = True
                    failure+=1
                    
                    if failure > initial_failures:                        
                        if soften == True:
                            total_failure = False
                            mini = srate*mini
                            failure = 0
                            issues = 0
                            if suppress == False:
                                print(f"Failure tolerance reached at random walk {self.num_walks}.")
                                print(f"Softening minimum requirement. Current minima: {mini}")

                        if mini < danger and cell_list != None:
                            mini = 1.12234
                            cell_num = random.choice(cell_list)
                            print("Finding new starting cell in provided cell list.")
                            

                        else:
                            total_failure = True
                            invalid_start = False

                else:
                    invalid_start = False
                    failure = 0        

            if total_failure == True:
                print("\nThe number of permitted failures for the starting position have")
                print("exceeded the set value. The box is too dense for a valid position to")
                print("be found.")
                print("It is recommended to run the algorithm in a less packed box.")
                raise Exception("Program terminated.")

            # The full data entry for keeping track of the bead.
            bead_data = [ID,
                         0,
                         bead_sequence[0],
                         bond_type,
                         self.num_beads,
                         starting_pos]

            self.num_beads += 1
            self.num_walk_beads += 1 # the individual count beads that make up a walk
            self.index(current_cell).beads.append(bead_data)

        mini = 1.12234 # resetting minimum
        
        # Begin loop here.
        bond = mini*sigma # the minimum of the LJ potential
        i = 1
        current_pos = starting_pos

        danger_mode = False
        retraction_count = 0
        current_block = 0 # this is the index of bcp_specs
        region = 1
        
        while i < numbeads:            
            too_close = True # used to check if the cell is too close
            generation_count = 0 # this counts the number of random vectors generated.
                                 # used to raise error messages


            # -------------------------------------------------------------------------------------------------
            # Who am I? Where am I going?
            # -------------------------------------------------------------------------------------------------
            bead_type = bead_sequence[i]                    
            next_type = bcp_specs[current_block][0][1] # this will be the next region

            if meanfield == True:
                # the allowed range calcuated using bcp_analyze and the allowed_range function
                allowed_range = weights[i]            

                # store the current densities in here: all densities are required.
                current_densities = self.index(current_cell).densities            

                # the density of the *c*urrent bead and the density of the *n*ext type of bead
                cdensity = current_densities[self.interactions.typekeys[bead_type]-1]
                ndensity = current_densities[self.interactions.typekeys[next_type]-1]

                # the current range CAN be negative.
                current_range = cdensity - ndensity

                history_file.write(f"{i}\t{np.round(allowed_range, 5)}\t{np.round(current_range,5)}\n")
            # -------------------------------------------------------------------------------------------------

            while too_close:
                # calculate the bead subsequence            
                # MAIN LOOP -----------------------------------------------------------------------------------------------
                # loop works like this:
                #    0. generates trial position
                #    1. generates a list of beads in neighbouring cell for this trial position
                #    2. checks the distance between neighours and trial_position
                #    3. too close is defaultly assumed and maintained until end of loop

                not_valid = True
                while not_valid:
                    if end_pos == None:
                        trial_pos = new_position(current_pos, bond, 1, self.cellside*self.cellnums, phi, theta) # new posn
                    else:
                        trial_pos = forced_new_position(end_pos,
                                                        i,
                                                        numbeads, 
                                                        current_pos,
                                                        bond,
                                                        1,
                                                        self.cellside*self.cellnums,
                                                        phi,
                                                        theta) # new posn
                        
                    not_valid = self.index(self.which_cell(trial_pos)).forbidden
                    
                previous = current_pos
                current_pos = trial_pos
                

                neighbours = self.check_surroundings(current_pos)
                issues = 0
                index_c = self.which_cell(current_pos)
                
                # neighbor checking takes place here.
                for j in neighbours:            
                    index_n = self.which_cell(j[-1])
                    
                    # deals with instances of periodicity, no different from the code above.
                    # to obtain the constant, calculate the maximum norm of two adjacent cell indices
                    if self.cellnums-1 in np.abs(index_c - index_n):
                        period = np.array([i if abs(i) == (self.cellnums-1) else 0 for i in (index_n - index_c)])
                        real_j = -period*(self.cellnums/(self.cellnums - 1))*self.cellside + j[-1]
                    else:
                        real_j = j[-1]                                            

                    distance = round(np.linalg.norm(current_pos - real_j), 5)
                    sigma = mini*self.interactions.return_sigma(bead_type, j[2])
                    if distance < sigma:                        
                        issues += 1
                        break
                    
                # monte carlo check
                if meanfield == True:
                    # checks if the bead is limited to any given region.
                    # this is experimental. However, it could be pretty useful!
                    limit = self.interactions.limits[bead_type]                
                    if density < limit:
                        issues+=1 

                    # ----------------------------------------------------------------------------------------
                    # MC meanfield check
                    # ----------------------------------------------------------------------------------------
                    # Descripton of the algorithm:
                    # There are three regions of the random walk in which different biases are applied.
                    #                    
                    # REGION 1: The push region. Here, we want to push the random walk into a place with higher 
                    #           density of the same type.
                    # REGION 0: The NORMAL region. Here, we should only push our region to a higher density of
                    #           the same type if the density difference is less than 0 (which indicates that we
                    #           are in a region that is more blue than red.)
                    # REGION -1: The push region. Here, we want to push the random walk into a place with higher 
                    #           density of the next occuring type in the chain. However, if the difference in
                    #           density is less than zero, we don't do any pushing at all.
                                  
                    if i <= boundaries[current_block][0]:
                        # push to a region with higher density of SAME type
                        region = 1
                    if i >= boundaries[current_block][1]:
                        # push to a region with higher density of DIFFERENT type
                        region = -1
                    if i > boundaries[current_block][0] and i < boundaries[current_block][1]:
                        # the anarchy region: no pushing or pulling
                        region = 0


                    # These regions are used to control the strength of the MC move applied.
                    # Strong move: If the density difference of the trial region is favorable, accept it straight
                    #              away. Essentially, SKIP the MC for density.
                    #              If the density difference of the trial region is unfavorable, apply the weak 
                    #              move.
                    # Weak move:   Simply apply the standard density random walk test. 
                    # ------------------------------------------------------------------------------------------
                    # Conditions for strong and weak moves 
                    # ------------------------------------------------------------------------------------------
                    # When in Region 1, apply strong move and push towards the SAME type of bead ONLY when:
                    #  * the density difference is LESS than the allowed range.
                    #  * the density difference is negative.
                    # Otherwise, apply weak move.
                    if region == 1:
                        # region == 1 -> push to a denser SAME TYPE region.
                        if current_range > 0:
                            if current_range > allowed_range:
                                nudge = "No nudge"
                                # apply basic MC scheme
                                random_num = np.random.uniform(0,1)                
                                trial_density = self.index(index_c).densities[self.interactions.typekeys[bead_type]-1]
                                if random_num > trial_density:
                                    issues+=1        
                        
                                else:
                                    # apply strong MC scheme: does not activate if trial_density < cdensity.
                                    nudge = "NUDGE"
                                    trial_density = self.index(index_c).densities[self.interactions.typekeys[bead_type]-1]
                                    if trial_density < cdensity:
                                        # the proposed cell is less dense than current cell.
                                        # apply the basic scheme.
                                        random_num = np.random.uniform(0,1)                
                                        if random_num > trial_density:
                                            issues+=1
                                        
                        else:
                            nudge = "NUDGE"
                            trial_density = self.index(index_c).densities[self.interactions.typekeys[bead_type]-1]
                            if trial_density < cdensity:
                                # the proposed cell is less dense in the current bead than current cell.
                                # apply the basic scheme.
                                random_num = np.random.uniform(0,1)
                                if random_num > trial_density:
                                    issues+=1


                    # When in Region -1, apply strong move and push towards the NEXT type of bead ONLY when:
                    #  * the density difference is MORE than the allowed range.
                    # Otherwise, apply weak move
                    if region == -1:
                        if current_range < allowed_range:
                            nudge = "No nudge"
                            # apply basic MC scheme
                            trial_density = self.index(index_c).densities[self.interactions.typekeys[bead_type]-1]
                            random_num = np.random.uniform(0,1)                
                            if random_num > trial_density:
                                issues+=1
                        else:
                            nudge = "NUDGE" # to next bead
                            trial_density = self.index(index_c).densities[self.interactions.typekeys[next_type]-1]
                            if trial_density < ndensity:
                                # the proposed cell is less dense than current cell.
                                # apply the basic scheme.
                                trial_density = self.index(index_c).densities[self.interactions.typekeys[bead_type]-1]
                                random_num = np.random.uniform(0,1)
                                if random_num > trial_density:
                                    issues+=1


                    # When in Region 0, apply strong move and push towards the SAME type of bead ONLY when:
                    #  * the density difference is negative.                    
                    if region == 0:
                        # use the normal monte-carlo rule.
                        if current_range > 0:
                            nudge = "No nudge"
                            trial_density = self.index(index_c).densities[self.interactions.typekeys[bead_type]-1]
                            # apply basic MC scheme
                            random_num = np.random.uniform(0,1)                
                            if random_num > trial_density:
                                issues+=1        
                                        
                        else:
                            nudge = "NUDGE"
                            trial_density = self.index(index_c).densities[self.interactions.typekeys[bead_type]-1]
                            if trial_density < cdensity:
                                # the proposed cell is less dense in the current bead than current cell.
                                # apply the basic scheme.
                                random_num = np.random.uniform(0,1)
                                if random_num > trial_density:
                                    issues+=1

                # print(f"{i}\t{bead_type}\t{region}\t{np.round(current_range, 5)}\t\t{np.round(allowed_range, 5)}\t\t{nudge}")

                if issues > 0:      
                    too_close = True
                    current_pos = previous
                else:
                    too_close = False

                # This has to be here: the failure condition is False when generation_count = 0
                generation_count += 1

                # FAILURE CONDITIONS -------------------------------------------------------------
                if generation_count % walk_failures == 0:
                    if termination == "break":
                        new_numbeads = numbeads - i
                        self.num_beads += i
                        self.num_bonds += i - 1
                        self.num_walks += 1
                        print(f"Breaking random walk! There are now {self.num_walks} chains in the system.")
                        print(f"New chain length: {new_numbeads}")
                        self.random_walk(new_numbeads,
                                         Kval,
                                         cutoff,
                                         energy,
                                         sigma,
                                         mini=mini,
                                         style=style,
                                         bead_sequence=bead_sequence,
                                         restart=restart,
                                         sequence_range = sequence_range,
                                         termination=termination)                    
                        return 0
                    
                    elif termination == "retract":
                        # pick a number of beads that will be deleted

                        self.retractions += 1
                        # adjust walk positions
                        
                        if i == 1:
                            # conditions for when the random walk returns to the initial bead
                            # here, retraction is not possible: two cases must be considered
                            # 1. CASE 1: Walk is standalone.
                            #            Here, the walk should be cancelled and resumed elsewhere.
                            #            One way to do this is via recursion.
                            # 2. CASE 2: Walk is part of a graft.
                            #            In this situation a new position must be found for the graft, but constrained to 
                            #            the region in which the grafting target is found.



                            # delete starter bead
                            progress = self.walk_data(ID)
                            bad_bead = progress[-1]
                            current_cell = self.which_cell(bad_bead[-1])
                            self.index(current_cell).beads.remove(bad_bead)

                            self.num_beads -= 1
                            self.num_walk_beads -= 1 # the individual count beads that make up a walk

                            # redo random walk
                            print("Walk has retracted to first bead - restarting random walk.")

                            retraction_count += 1
                            if retraction_count > retraction_limit:
                                print("----------------------------------------------------------")
                                print("Walk occuring in a highly unfavourable area.")
                                print("Terminating algorithm to avoid recursion problems.")
                                print("The walk may be rerun using the dead_walks attribute.")
                                print("----------------------------------------------------------")
                                self.dead_walks +=1
                                return 0

                            self.randomwalk(numbeads,
                                            Kval,
                                            cutoff,
                                            energy,
                                            sigma,
                                            bead_sequence,
                                            mini=1.12234,
                                            style='fene',
                                            phi=phi,
                                            theta=theta,
                                            cell_num=cell_num, 
                                            starting_pos=list(starting_pos),
                                            meanfield=meanfield,
                                            end_pos=end_pos,
                                            soften=soften, 
                                            srate=srate,
                                            suppress=suppress,
                                            danger = danger,
                                            termination=termination,
                                            initial_failures=initial_failures,
                                            walk_failures=walk_failures)
                            return 0
                        else:
                            
                            deletion = random.randint(1, int(0.5*i))
                            initial = i

                            for bead in range(deletion):
                                progress = self.walk_data(ID)
                                bad_bead = progress[-1]
                                current_cell = self.which_cell(bad_bead[-1])

                                self.index(current_cell).beads.remove(bad_bead)                        

                                new_progress = self.walk_data(ID)          
                                current_pos = new_progress[-1][-1]
                                current_cell = self.which_cell(current_pos)

                                i -= 1
                                self.num_beads -= 1
                                self.num_walk_beads -= 1

                            print(f"Retracting at bead {initial} of random walk {ID}. Current bead: {i}")

                            if danger_mode == True:
                                termination = 'soften'
                                mini = 1.12234
                                danger_mode = False

                    elif termination == "soften":
                        if suppress == False:
                            print(f"Failure tolerance reached at bead walk {self.num_beads}. Softening minimum requirement. Current minima: {mini}")
                            mini = srate*mini

                            if mini < danger:
                                danger_mode = True
                                termination = 'retract'

                    else:                        
                        print("Warning for Random Walk ID: " + str(ID))
                        print("The system has now generated %d unsuccessful positions for this walk." % (generation_count))
                        print("It's very likely that the random walk has either become trapped or")
                        print("that the lattice has become too dense to generate a valid position.")
                        print("It is highly recommended to restart the program with:")
                        print("a) a fewer number of atoms,")
                        print("b) a larger amount of lattice space,")
                        print("c) a smaller step distance.")
                        raise Exception("Program terminated.")
                        
                # -----------------------------------------------------------------------------------


            mini = 1.12234
            current_pos = trial_pos
            current_cell = self.which_cell(current_pos)

            # 0: random walk this belongs to
            # 1: number of the bead (on the random walk)
            # 2: bead type
            # 3: bead mass
            # 4: bond type (number)
            # 5: number of beads within structure
            # 6: grafting
            # -1: current position

            # Note 6: grafting. if -1, no graft. Otherwise? graft is present.
            #         When there IS a graft, 6 is replaced with the global number of the first grafted
            #         bead.
            # Note -1: the last index of the bead data, used as convention throughout the program.

            bead_data = [ID,
                         i,
                         bead_sequence[(i % len(bead_sequence))],
                         bond_type,
                         self.num_beads,
                         current_pos]

            self.num_walk_beads += 1
            self.num_beads += 1
            self.index(current_cell).beads.append(bead_data)

            self.index(current_cell).energy += energy

            i+=1
            if i > bcp_specs[current_block][1][1]:
                current_block+=1

        self.num_walks += 1
        self.num_bonds += numbeads - 1
        self.random_walked = True

        self.walkinfo[ID] = self.num_walk_beads

        if history==True:
            history_file.close()

        return 1    
    
    def graft_chain(self, starting_bead, numbeads, Kval, cutoff, energy, sigma, bead_sequence, mini=1.12234, 
                    starting_distance = None,
                    style='fene', phi=None, theta=None, cell_num=None,
                    meanfield=False,
                    soften=True, srate=0.99, suppress=False,
                    sequence_range = 1,
                    termination='soften', danger=0.6,
                    retraction_limit = 10,
                    initial_failures=10000, walk_failures=10000):
        """
        This method is used to grow extra beads at a particular point in a given chain.
        Intended to study the effects of different chain architectures on macroscopic properties.
        
        num_beads: the number of beads on the grafted chain
        bead_types: the bead type sequence on the grafted chain
        starting_bead: the bead that the new chain will be grafted to.
        starting_distance: the initial distance from the bead at which the grafted random walk will begin.

        The algorithm works by *storing* the connection of a chain in terms of coordinates.
        (Note: the coordinates of a bead are (A, B), where A is the walk number and B is the
        number of the bead on that walk.)
        Then, with a position randomly chosen about the bead that was specified for fixing,
        a new random walk is generated.
        """
        def rand_distance(length, dimension):
            """
            Used to generate freely jointed chain models. This is the default option.
            produces a vector with fixed length and random direction. 
            length: fixed length of the vector.
            dimension: self-explanatory
            """

            x = np.random.randn(dimension, 3)
            magnitude = np.linalg.norm(x)

            return length*(x/magnitude)
        
        def new_position(position, length, dimension, maxdis, phi=None, theta=None):
            """
            Copy from the previous. I should probably structure this better...
            """
            # need to flip between configurations of the angles
            flipper = [-1, 1]
            flip1 = random.sample(flipper,1)[0] # for theta
            flip2 = random.sample(flipper,1)[0] # for phi
            
            if (phi == None and theta == None):            
                new_pos = np.array(position) + rand_distance(length, dimension)
                new_pos = new_pos[0]
            elif (phi != None and theta != None):
                theta = flip1*theta
                phi = flip2*phi
                x = length*m.cos(theta)*m.cos(phi) 
                y = length*m.sin(theta)*m.sin(phi)
                z = length*m.cos(theta)
                new_pos = np.array(position) + np.array([x, y, z])
            else:
                if phi==None:
                    phi = np.random.uniform(-m.pi, m.pi)
                else:
                    theta = np.random.uniform(-m.pi, m.pi)

                theta = flip1*theta
                phi = flip2*phi
                                    
                x = length*m.cos(theta)*m.cos(phi) 
                y = length*m.sin(theta)*m.sin(phi)
                z = length*m.cos(theta)
                
                new_pos = np.array(position) + np.array([x, y, z])

            # Boundary conditions
            for i in range(len(new_pos)):
                if new_pos[i] < 0:
                    new_pos[i] = maxdis + new_pos[i]
                if new_pos[i] > maxdis:
                    new_pos[i] = new_pos[i] - maxdis
                
            return new_pos

        
        # -------------------------------------------------------------------------------------
        # bond type dictionary (same structure as for all files)
        if self.bonds == None:
            self.bonds = {}
            self.bonds[0] = (Kval, cutoff, energy, sigma, style)
        else:
            repeat = 0
            for i in range(len(self.bonds)):
                if self.bonds[i] != (Kval, cutoff, energy, sigma, style):
                    repeat += 1
                else:
                    store = self.bonds[i]

            if repeat >= len(self.bonds):
                self.bonds[len(self.bonds)] = (Kval, cutoff, energy, sigma, style)
                
        bond_type = [i for i in range(len(self.bonds)) if self.bonds[i]==(Kval, cutoff, energy, sigma, style)][0]+1

        # assign the beads in the bead sequence into the used_types set
        for i in bead_sequence:
            if i not in self.interactions.used_types:
                self.interactions.used_types.append(i)
        
        # get the position of the starting bead
        position = self.walk_data(starting_bead[0])[starting_bead[1]][-1]

        if starting_distance == None:
            starting_distance = mini*sigma

        # employ the same mechanism as in randomwalk to find a suitable position.
        too_close = True
        generation_count = 0
        while too_close:
            trial = new_position(position, starting_distance, 1, self.cellside*self.cellnums, phi, theta)
            
            previous = position
            current_pos = trial

            neighbours = self.check_surroundings(current_pos)
            issues = 0
            index_c = self.which_cell(current_pos)
            # neighbor checking takes place here.
            for j in neighbours:            
                index_n = self.which_cell(j[-1])
                    
                # deals with instances of periodicity, no different from the code above.
                # to obtain the constant, calculate the maximum norm of two adjacent cell indices
                if self.cellnums-1 in np.abs(index_c - index_n):
                    period = np.array([i if abs(i) == (self.cellnums-1) else 0 for i in (index_n - index_c)])
                    real_j = -period*(self.cellnums/(self.cellnums - 1))*self.cellside + j[-1]
                else:
                    real_j = j[-1]                                            

                distance = round(np.linalg.norm(current_pos - real_j), 5)
                check = mini*self.interactions.return_sigma(bead_sequence[0], j[2])
                if distance < check:                        
                    issues += 1
                    break

            # monte carlo check
            if meanfield == True:
                random_num = np.random.uniform(0,1)
                density = self.index(index_c).densities[self.interactions.typekeys[bead_sequence[0]]-1]
                if random_num > density:
                    issues+=1

            if issues > 0:      
                too_close = True
                current_pos = previous
                generation_count+=1

                if generation_count % initial_failures == 0:
                    if soften == True:
                        mini = srate*mini
                        if suppress == False:
                            print(f"Failure tolerance reached when grafting at {self.num_beads}.")
                            print(f"Softening minimum requirement. Current minima: {mini}")

                    else:
                        print("Position for graft bead not found. Consider reattempting with a sparser box.")
                        print("Graft unsuccessful.")
                        raise Exception("Program terminated.")            
            else:
                too_close = False

        # The full data entry for keeping track of the bead.
        ID = self.num_walks + 1
        bead_data = [ID,
                     0,
                     bead_sequence[0],
                     bond_type,
                     self.num_beads,
                     current_pos]

        self.num_beads += 1
        self.num_walk_beads += 1 # the individual count beads that make up a walk
        self.index(self.which_cell(current_pos)).beads.append(bead_data)

        # the algorithm should by now have returned a valid position for the random walk. (or failed)
        # now, all that's left is to run a random walk from this position.
        mini = 1.12234 # resetting minimum
        
        # Begin loop here.
        bond = mini*sigma # the minimum of the LJ potential
        i = 1
        
        retraction_count = 0 # used when retraction occurs at the first bead        
        danger_mode = False
        while i < numbeads:            
            too_close = True # used to check if the cell is too close
            generation_count = 0 # this counts the number of random vectors generated.
                                 # used to raise error messages
                
            # manually slice the list and build a subsequence from which the first bead is that of the current index
            #            bead_subseq = [bead_sequence[(i+j)%len(bead_sequence)] for j in range(sequence_range) 
            #                           if (i+j) < last_element else bead_sequence[(last_element+j)%len(bead_sequence)]

            last_element = len(bead_sequence) - sequence_range-1 # the last element before the subsequence terminates
            bead_subseq = []
            for j in range(sequence_range):
                if i+j > last_element:
                    bead_subseq.append(bead_sequence[(last_element+j)])
                else:
                    bead_subseq.append(bead_sequence[(i+j)])
                    
            while too_close:
                # loop works like this:
                #    0. generates trial position
                #    1. generates a list of beads in neighbouring cell for this trial position
                #    2. checks the distance between neighours and trial_position
                #    3. too close is defaultly assumed and maintained until end of loop

                not_valid = True
                while not_valid:
                    trial_pos = new_position(current_pos, bond, 1, self.cellside*self.cellnums, phi, theta) # new posn
                    not_valid = self.index(self.which_cell(trial_pos)).forbidden
                    
                previous = current_pos
                current_pos = trial_pos
                
                bead_type = bead_subseq[0]

                neighbours = self.check_surroundings(current_pos)
                issues = 0
                index_c = self.which_cell(current_pos)
                
                # neighbor checking takes place here.
                for j in neighbours:            
                    index_n = self.which_cell(j[-1])
                    
                    # deals with instances of periodicity, no different from the code above.
                    # to obtain the constant, calculate the maximum norm of two adjacent cell indices
                    if self.cellnums-1 in np.abs(index_c - index_n):
                        period = np.array([i if abs(i) == (self.cellnums-1) else 0 for i in (index_n - index_c)])
                        real_j = -period*(self.cellnums/(self.cellnums - 1))*self.cellside + j[-1]
                    else:
                        real_j = j[-1]                                            

                    distance = round(np.linalg.norm(current_pos - real_j), 5)
                    sigma = mini*self.interactions.return_sigma(bead_type, j[2])
                    if distance < sigma:                        
                        issues += 1
                        break
                    
                # monte carlo check
                if meanfield == True:
                    random_num = np.random.uniform(0,1)
                    avg_density = sum([self.index(index_c).densities[self.interactions.typekeys[i]-1] for i in bead_subseq])/len(bead_subseq)
                    if random_num > avg_density:
                        issues+=1
                    
                if issues > 0:      
                    too_close = True
                    current_pos = previous
                else:
                    too_close = False

                # This has to be here: the failure condition is False when generation_count = 0
                generation_count += 1

                # FAILURE CONDITIONS -------------------------------------------------------------
                if generation_count % walk_failures == 0:
                    if termination == "retract":
                        # pick a number of beads that will be deleted

                        self.retractions += 1
                        # adjust walk positions
                        
                        if i == 1:
                            # conditions for when the random walk returns to the initial bead
                            # here, retraction is not possible: two cases must be considered
                            # 1. CASE 1: Walk is standalone.
                            #            Here, the walk should be cancelled and resumed elsewhere.
                            #            One way to do this is via recursion.
                            # 2. CASE 2: Walk is part of a graft.
                            #            In this situation a new position must be found for the graft, but constrained to 
                            #            the region in which the grafting target is found.


                            # REDO RANDOM WALK ----------------------------------------------------------------------------
                            print("Graft has returned to initial bead. Finding a new starting position.")                                                           
                            # employ the same mechanism as in randomwalk to find a suitable position.
                            

                            i = 0 # reset index to 0, NOT 1.
                            self.num_beads -= 1
                            self.num_walk_beads -= 1 # the individual count beads that make up a walk

                            # delete the initially placed bead
                            progress = self.walk_data(ID)
                            bad_bead = progress[-1]
                            current_cell = self.which_cell(bad_bead[-1])
                            self.index(current_cell).beads.remove(bad_bead)                        

                            # this is to ensure that the graft bead is not truly stuck
                            retraction_count += 1
                            if retraction_count > retraction_limit:
                                print("----------------------------------------------------------")
                                print("Graft bead placed in highly unfavourable area.")
                                print("Deleting previous random walk. This walk should be re-run.")
                                print("----------------------------------------------------------")
                                self.delete_walk(self.num_walks)                                
                                self.dead_walks +=1
                                return 0


                            too_close = True
                            generation_count = 0
                            position = self.walk_data(starting_bead[0])[starting_bead[1]][-1]                            
                            
                            while too_close:
                                trial = new_position(position, starting_distance, 1, self.cellside*self.cellnums, phi, theta)

                                previous = position
                                current_pos = trial

                                neighbours = self.check_surroundings(current_pos)
                                issues = 0
                                index_c = self.which_cell(current_pos)
                                # neighbor checking takes place here.
                                for j in neighbours:            
                                    index_n = self.which_cell(j[-1])

                                    # deals with instances of periodicity, no different from the code above.
                                    # to obtain the constant, calculate the maximum norm of two adjacent cell indices
                                    if self.cellnums-1 in np.abs(index_c - index_n):
                                        period = np.array([i if abs(i) == (self.cellnums-1) else 0 for i in (index_n - index_c)])
                                        real_j = -period*(self.cellnums/(self.cellnums - 1))*self.cellside + j[-1]
                                    else:
                                        real_j = j[-1]                                            

                                    distance = round(np.linalg.norm(current_pos - real_j), 5)
                                    check = mini*self.interactions.return_sigma(bead_sequence[0], j[2])
                                    if distance < check:                        
                                        issues += 1
                                        break

                                # monte carlo check
                                if meanfield == True:
                                    random_num = np.random.uniform(0,1)
                                    density = self.index(index_c).densities[self.interactions.typekeys[bead_sequence[0]]-1]
                                    if random_num > density:
                                        issues+=1

                                if issues > 0:      
                                    too_close = True
                                    current_pos = previous
                                    generation_count+=1

                                    if generation_count % initial_failures == 0:
                                        if soften == True:
                                            mini = srate*mini
                                            if suppress == False:
                                                print(f"Failure tolerance reached when grafting at {self.num_beads}.")
                                                print(f"Softening minimum requirement. Current minima: {mini}")

                                        else:
                                            print("Position for graft bead not found. Consider reattempting with a sparser box.")
                                            print("Graft unsuccessful.")
                                            raise Exception("Program terminated.")            
                                else:
                                    too_close = False     
                                
                        else:
                            
                            deletion = random.randint(1, int(0.5*i))
                            initial = i

                            for bead in range(deletion):
                                progress = self.walk_data(ID)
                                bad_bead = progress[-1]
                                current_cell = self.which_cell(bad_bead[-1])

                                self.index(current_cell).beads.remove(bad_bead)                        

                                new_progress = self.walk_data(ID)          
                                current_pos = new_progress[-1][-1]
                                current_cell = self.which_cell(current_pos)

                                i -= 1
                                self.num_beads -= 1
                                self.num_walk_beads -= 1

                            print(f"Retracting at bead {initial} of random walk {ID}. Current bead: {i}")

                            if danger_mode == True:
                                termination = 'soften'
                                mini = 1.12234
                                danger_mode = False

                    elif termination == "soften":
                        if suppress == False:
                            print(f"Failure tolerance reached on graft walk {self.num_beads}. Softening minimum requirement. Current minima: {mini}")
                            mini = srate*mini

                            if mini < danger:
                                danger_mode = True
                                termination = 'retract'

                    else:                        
                        print("Warning for Random Walk ID: " + str(ID))
                        print("The system has now generated %d unsuccessful positions for this walk." % (generation_count))
                        print("It's very likely that the random walk has either become trapped or")
                        print("that the lattice has become too dense to generate a valid position.")
                        print("It is highly recommended to restart the program with:")
                        print("a) a fewer number of atoms,")
                        print("b) a larger amount of lattice space,")
                        print("c) a smaller step distance.")
                        raise Exception("Program terminated.")
                        
                # -----------------------------------------------------------------------------------


            mini = 1.12234
            current_pos = trial_pos
            current_cell = self.which_cell(current_pos)

            # 0: random walk this belongs to
            # 1: number of the bead (on the random walk)
            # 2: bead type
            # 3: bead mass
            # 4: bond type (number)
            # 5: number of beads within structure
            # 6: grafting
            # -1: current position

            # Note 6: grafting. if -1, no graft. Otherwise? graft is present.
            #         When there IS a graft, 6 is replaced with the global number of the first grafted
            #         bead.
            # Note -1: the last index of the bead data, used as convention throughout the program.

            bead_data = [ID,
                         i,
                         bead_sequence[(i % len(bead_sequence))],
                         bond_type,
                         self.num_beads,
                         current_pos]

            self.num_walk_beads += 1
            self.num_beads += 1
            self.index(current_cell).beads.append(bead_data)

            i+=1

        self.num_walks += 1
        self.num_bonds += numbeads - 1
        self.random_walked = True

        self.walkinfo[ID] = self.num_walk_beads        
        self.graft_ids.append(ID)
        
        gbead_0 = [self.num_walks, 0]
        self.graft_coords.append([starting_bead, gbead_0])
        self.num_grafts += 1

        # add that one extra bond
        self.num_bonds += 1    

        return 1    

    def unbonded_crosslinks(self, bead_type, crosslinks, Kval, cutoff, energy, sigma, allowed=None,bdist=None, prob=1.0, style='fene', allowed_failures = 50, ibonds=2, jbonds=1):
        """ 
        This method initializes crosslinks that will move around the box before bonding within the simulation.
        This makes use of the "self.unbonded_links" paramater, which stores the properties of the bonds that will be created.
        Arguments- bead_types:     Type of the bead. Has to be declared manually beforehand.
                                   (See interactions class)
                   crosslinks:     The number of crosslinks in the system.
                   sigma:           The main range at which molecules can be linked.
                   mass:           Mass of the crosslinker cell
                   allowed:        A list of bead types that are allowed to crosslink with the unbonded beads.
                   bdist:          the minimum bonding distance
                   prob:           the probability of linking
                   numlinks:       the number of links the crosslink atom is allowed to make.
        """
        if self.structure_ready == True:
            raise EnvironmentError("Structures cannot be built or modified when simulation procedures are in place.")
        
        if bdist == None:
            bdist=sigma

        if allowed == None:
            allowed = [i for i in self.interactions.types]
        # adding the bond detail of the crosslink to the system        
        if self.bonds == None:
            self.bonds = {}
            self.bonds[0] = (Kval, cutoff, energy, sigma, style)
        else:
            repeat = 0
            for i in range(len(self.bonds)):
                if self.bonds[i] != (Kval, cutoff, energy, sigma, style):
                    repeat += 1
                else:
                    store = self.bonds[i]
            if repeat >= len(self.bonds):            
                self.bonds[len(self.bonds)] = (Kval, cutoff, energy, sigma, style)
            
        bond_type = [i for i in range(len(self.bonds)) if self.bonds[i]==(Kval, cutoff, energy, sigma, style)][0]+1
        
        bond = sigma
        
        # Flagging up the presence of unbonded crosslinks within the system
        if self.cl_unbonded == True:
            s1 = "Unbonded crosslinks can only be defined once.\n"
            s2 = "Multitype unbonded crosslinking is not yet supported in nanoPoly functionality.\n"
            s3 = "Please contact the developer if you would like this functionality to be implemented.\n"
            raise Exception(f"{s1}{s2}{s3}Program terminated")
        else:
            self.cl_unbonded = True
        
        
        # arguments: 
        # 0: the crosslink bead
        # 1: the type of bond that the crosslink bead is allowed to have
        # 2: the beads that the cl bead is allowed to link to 
        # 3: the bonding distance
        # 4: number of bonds allowed on the crosslink bead
        # 5: number of bonds allowed on the attaching bead
        # 6: probability of linking

        self.cl_bonding = [bead_type, bond_type, allowed, bdist, ibonds, jbonds, prob]

        # add bead type into the used_types list
        if i not in self.interactions.used_types:
            self.interactions.used_types.append(bead_type)

        # increment the unbonded crosslinker structure count
        self.num_uclstructs += 1
        for i in range(crosslinks):                    
            # trial for the initial position
            invalid_pos = True
            total_failure = False
            failure = 0
            while invalid_pos:
                # select a random cell
                current_cell = np.array([random.randrange(0, self.cellnums),
                                         random.randrange(0, self.cellnums),
                                         random.randrange(0, self.cellnums)])

                # obtain the bounds of previously selected cell
                cell_pos = self.index(current_cell).position
                cell_bound = cell_pos + self.cellside

                # find a position within the cell
                trial_pos = np.array([random.uniform(cell_pos[0], cell_bound[0]),
                                      random.uniform(cell_pos[1], cell_bound[1]),
                                      random.uniform(cell_pos[2], cell_bound[2])])
                
                index_c = self.which_cell(trial_pos)                    
                neighbours = self.check_surroundings(trial_pos)
                
                issues = 0
                for j in neighbours:
                    index_n = self.which_cell(j[-1])
                    if self.cellnums-1 in np.abs(index_c - index_n):
                        period = np.array([i if abs(i) == (self.cellnums-1) else 0 for i in (index_n - index_c)])
                        real_j = -period*(self.cellnums/(self.cellnums - 1))*self.cellside + j[-1]
                    else:
                        real_j = j[-1]

                    sigma = self.interactions.return_sigma(bead_type, j[2])
                    if round(np.linalg.norm(trial_pos - real_j), 5) < sigma:
                        issues+=1
                        break # breaks out of the neighborlist search
                    
                if issues > 0:
                    invalid_pos = True
                    failure += 1                    
                    if failure > allowed_failures:
                        total_failure = True
                        invalid_pos = False
                else:
                    invalid_pos = False
                    failure = 0
                    
            if total_failure == True:
                print("---------------------------- CROSSLINK FAILURE ----------------------------")
                print("The number of failures have surpassed the fail count.")
                print("It is highly unlikely that the box is too dense to accomodate any more links.")
                print("The simulation will continue with the number of links that have already")
                print("been made. ")
                print(f"Number of crosslinks currently in systems: {i}")
                print("---------------------------------------------------------------------------")
                self.cl_bonding.append(i)                
                return 0
            
            bead = [f"uc{self.num_uclstructs}", # ubc: unbonded crosslinks
                    self.num_uclbeads,
                    bead_type,
                    bond_type,
                    self.num_beads,
                    trial_pos]                        
                                                                        
            self.index(current_cell).beads.append(bead)
            self.num_beads += 1
            self.num_uclbeads +=1
        
        self.cl_bonding.append(crosslinks)
        self.uclinfo[self.num_uclstructs] = self.num_uclbeads

                
    def bonded_crosslinks(self, bead_type, crosslinks, Kval, cutoff, energy, sigma, style='fene', forbidden=[], reaction=True, fail_count=50, selflinking=10, multilink=False):
        """ 
        This method initializes crosslinks that are already bonded within the system.
        Carries out the crosslinking procedure in the system.
        Rejects any system with less than one chain.

        Arguments- crosslink:     the number of crosslinks in the system
                   bond:          the main range at which molecules can be linked.
                   
                   mass:          mass of the crosslinker cell
                   forbidden:     a list of bead types that are forbidden to crosslink
                   reaction:      whether the links were formed via a crosslinking reaction or not
                                  all this does is add an atom to the middle of the cross-linkers 
                   fail_count:    the number of attempts the function will make to find a valid crosslink.
                                  When 10 different crosslink sites are found to fail, the algorithm will 
                                  cease to function.
                   selflinking:   allows the chain to link with itself after a series of links have pass
ed.
                   multilink:     allows a bead to link multiple times.                   
        """
        if self.structure_ready == True:
            raise EnvironmentError("Structures cannot be built or modified when simulation procedures are in place.")
        
        def rotation(axis, theta):
            """
            Implementation of a rotation matrix generator.
            Uses 
            """
            axis = axis/np.linalg.norm(axis)
            ux, uy, uz = axis[0], axis[1], axis[2]
            theta = theta % 2*np.pi
            cos_part = 1 - np.cos(theta)
            sin = np.sin(theta)
            
            rotation_mat = np.array([[np.cos(theta) + (ux**2)*cos_part,
                                      ux*uy*cos_part - uz*sin,
                                      ux*uz*cos_part + uy*sin],
                                     [uy*ux*cos_part + uz*sin,
                                      np.cos(theta) + (uy**2)*cos_part,
                                      uy*uz*cos_part - ux*sin],
                                     [uz*ux*cos_part - uy*sin,
                                      uz*uy*cos_part + ux*sin,
                                      np.cos(theta) + (uz**2)*cos_part]])
            
            return rotation_mat

        def periodize(new_pos, maxdis):
            for i in range(len(new_pos)):                
                if new_pos[i] < 0:
                    new_pos[i] = maxdis + new_pos[i]
                if new_pos[i] > maxdis:
                    new_pos[i] = new_pos[i] - maxdis
            return new_pos

        def random_pick(potential_pairs, accepted_links, multilink):
            """
            This function carries out a number of tasks:
            a) Checks a list of potential crosslink pairs and picks a pair randomly
            b) Makes sure that the bonds are not the same
            c) Incorporates an optional condition that disallows crosslinking between the same beads
            """

            all_beads = [bead[-2] for beads in accepted_links for bead in beads]
            
            count = 0
            if multilink==False:
                invalid = True
                
                while invalid:
                    if count>len(potential_pairs):
                        return 0
                        break
                    
                    link = random.choice(potential_pairs)                    
                    # makes sure the links haven't already been accepted    
                    if (link[0][-2] in all_beads) or (link[1][-2] in all_beads):
                        count += 1
                    else:
                        invalid = False
            else:
                raise EnvironmentError("Multilinking not yet supported!")
            
            return link
            
        
        if self.random_walked == False:
            raise EnvironmentError("Box has not been populated with any beads. Please use the random walk method and try again.")

        # adding the bond detail of the crosslink to the system
        repeat = 0
        for i in range(len(self.bonds)):
            if self.bonds[i] != (Kval, cutoff, energy, sigma, style):
                repeat += 1
        if repeat >= len(self.bonds):
            self.bonds[len(self.bonds)] = (Kval, cutoff, energy, sigma, style)

        bond_type = [i for i in range(len(self.bonds)) if self.bonds[i]==(Kval, cutoff, energy, sigma, style)][0]+1
        bond = sigma


        # update the bonded crosslink count
        self.num_bclstructs += 1

        # ---------------- compiling the full list of potential pairs. ------------------
        potential_pairs = []        
        for i in range(self.num_walks):
            for bead in self.walk_data(i):
                if (bead[2] not in forbidden):
                    neighbours = self.check_surroundings(bead[-1])
                    for neighbour in neighbours:
                        # considering neighbours that are higher in the index, neighbours are further along the chain
                        # not of the same random walk.
                        cond1 = ((neighbour[0] != bead[0]) or ((neighbour[0] == bead[0] and (neighbour[1] - bead[1] > selflinking))))
                        # within linking distance
                        cond2 = (np.linalg.norm(neighbour[-1] - bead[-1]) < 2*bond)
                        # allowed to link                    
                        cond3 = (neighbour[2] not in forbidden)                        
                        # from a random walk, not a crosslinker bead
                        cond4 = isinstance(neighbour[0], int)
                        
                        
                        if cond1 and cond2 and cond3 and cond4:
                            potential_pairs.append((neighbour, bead))
        # -------------------------------------------------------------------------------

        if len(potential_pairs) == 0:
            # terminating condition. Extremely useful!
            print("No sites for crosslinking. Simulation will run as planned.")
            return 1
        else:
            # add bead type into the used_types list
            if i not in self.interactions.used_types:
                self.interactions.used_types.append(bead_type)
                
            crosslink_data = []
            accepted_links = []
            if reaction == True:
                # this option is taken if chains are linked via an atom (such as sulfur)
                #-------------------------------------------------------------------------------
                # The following algorithm
                #   1. Calculates a normal vector between the midpoint of the two lines
                #   2. Normalises it and multiplies by required length
                #   3. Sticks it onto the midpoint and checks for any close-by atoms
                #   4. If close-by atoms exist, rotates by a random angle and checks again
                #-------------------------------------------------------------------------------
                                
                bad_links = 0 # number of links that failed to crosslink
                # used to evaluate situations where crosslinking clearly isn't working
                              
                complete_failure = 0 # flag that pops up when crosslinking is no longer possible
                
                while (len(accepted_links) < crosslinks):
                    failure = 0 # a flag used to denote a link as having "failed"
                                # defaulted to be 0: we're assuming that the link picked in the next line will work
                                
                    # picks a link
                    link = random_pick(potential_pairs, accepted_links, multilink)
                    if link == 0:
                        print("---------------------------- CROSSLINK FAILURE ----------------------------")
                        print("No more valid crosslinking sites available.")
                        print("The simulation will continue with the number of links that have already")
                        print("been made. ")                         
                        print(f"Number of crosslinks currently in systems: {len(accepted_links)}")
                        print("---------------------------------------------------------------------------")
                        complete_failure = 0
                        break
                    
                    a = link[0][-1] 
                    b = link[1][-1]                    
                    midpoint = 0.5*(a + b) # midpoint
                    mvec = midpoint-a # vector inbetween midpoint and a

                    # build vector normal to mvec
                    normal = np.array([1.0,
                                       1.0,
                                       -(mvec[0]+mvec[1])/mvec[2]])
                    
                    # convert normal to unit                    
                    normal = normal*(np.linalg.norm(normal)**(-1))
                    # scale to appropriate length
                    normal = (np.sqrt(bond**2 - np.linalg.norm(mvec)**2))*normal
                    
                    # add to midpoint.                    .                    
                    pot_cross_loc = periodize(midpoint + normal, self.cellside*self.cellnums)

                    # keep rotating until a valid position is found
                    invalid = True
                    rot_failed = 0 # number of times the rotation fails
                    # all the following loop is supposed to do is calculate a valid position

                    while invalid:
                        # provide an initial rotation, just to be fair                        
                        rot_mat = rotation(mvec, np.random.uniform(0, 2*np.pi))
                        pot_cross_loc = periodize(midpoint + np.dot(rot_mat, normal), self.cellside*self.cellnums)
                        index_c = self.which_cell(pot_cross_loc)

                        # check the surroundings of the potential link
                        neighbours = self.check_surroundings(pot_cross_loc)
                        issue = 0 # flag to bring up issues
                        
                        for neighbour in neighbours:
                            index_n = self.which_cell(neighbour[-1])

                            if self.cellnums-1 in np.abs(index_c - index_n):
                                period = np.array([i if abs(i) == (self.cellnums-1) else 0 for i in (index_n - index_c)])
                                real_neighbour = -period*(self.cellnums/(self.cellnums - 1))*self.cellside + neighbour[-1]
                            else:
                                real_neighbour = neighbour[-1]                        

                            distance = np.linalg.norm(real_neighbour - pot_cross_loc)
                            sigma = self.interactions.return_sigma(bead_type, neighbour[2])
                            if distance < sigma:
                                issue = 1
                                break

                        if issue == 1:
                            invalid = True # not necessary, but readability is key
                            rot_failed += 1
                        else:
                            invalid = False
                            rot_failed = 0 # resets failure counter

                        if rot_failed > fail_count:
                            # flags a message demonstrating that the link itself is not capable of crosslinking
                            bad_links += 1
                            failure = 1
                            invalid = False
                            

                    if failure == 1:
                        if bad_links > fail_count:
                            print("---------------------------- CROSSLINK FAILURE ----------------------------")
                            print("The number of failures have surpassed the fail count.")
                            print("It is highly unlikely that there any valid cross-linking sites available.")
                            print("The simulation will continue with the number of links that have already")
                            print("been made. ")
                            print(f"Number of crosslinks currently in systems: {len(accepted_links)}")
                            print("---------------------------------------------------------------------------")
                            complete_failure = 1
                            break
                        continue
                    else:
                        accepted_links.append(link)

                        bead = [f"bc{self.num_bclstructs}",
                                self.num_bclbeads,
                                bead_type,
                                bond_type,
                                self.num_beads,
                                pot_cross_loc]
                        
                        self.num_beads += 1
                        self.num_bonds += 2
                        self.num_bclbeads += 1
                        
                        # adds bead to the lattice cell that it's in
                        current_cell = self.which_cell(bead[-1])
                        self.index(current_cell).beads.append(bead)
                        crosslink_data.append([link[0], bead, link[1]])

                # adds crosslinker atom to bead-type list
                self.crosslinks_loc += crosslink_data
                self.bclinfo[self.num_bclstructs] = self.num_bclbeads

                # termination condition
                # can be used to allow control flow operations
                if complete_failure == 1:
                    return len(accepted_links)/crosslinks # fuzzy condition, allows an evaluation of how many links were generated
                else:
                    return 0
                
            else:
                # generic network crosslinking
                self.crosslinks_loc = crosslink_data

    #---------------------------------------------------------------------------------------------
    # COMPOSITE FUNCTIONS
    # The functions in this section do not introduce any new functionality into the system. Rather,
    # they use existing functionality to perform more complex tasks.

    def uniform_chain(self, size,
                      rw_kval, rw_cutoff, rw_epsilon, rw_sigma,
                      bead_sequence, mini=1.12234,
                      termination="soften", srate=0.99, suppress="False", danger=0.6,
                      initial_failures=10000, walk_failures=10000,
                      sequence_range = 1,
                      retraction_limit = 10,
                      halfpoint=False,
                      dlist=None,
                      meanfield=False, soften=True,):

        # arguments:
        # dlist = a list containing ideal places to start the first part of the random walk

        if halfpoint==True:
            index = size//2
        else:
            # pick the index of a random bead in walk
            index = random.randrange(sequence_range,size-sequence_range)

        # generate the full list, and then split it apart at index i
        full_list = [bead_sequence[i%len(bead_sequence)] for i in range(size)]
        list1 = [full_list[i] for i in range(index)]
        list2 = [full_list[i] for i in range(index, size)]

        if dlist == None:            
            cellnum = None
        else:
            if len(dlist) == 0:
                print("Empty list provided for density list. Reverting to random cell picking.")
                cellnum = None
            else:
                cellnum = random.choice(dlist) 

        if sequence_range > len(list1):
            l1seq = len(list1)
        else:
            l1seq = sequence_range

        if sequence_range > len(list2):
            l2seq = len(list2)
        else:
            l2seq = sequence_range

        # generate the SECOND list into a random walk
        self.randomwalk(len(list2),
                        rw_kval,
                        rw_cutoff,
                        rw_epsilon,
                        rw_sigma,
                        list2,
                        cell_num = cellnum,
                        termination=termination,
                        meanfield=meanfield,
                        soften=True,
                        srate=srate,
                        sequence_range = l2seq,
                        initial_failures=initial_failures,
                        danger = danger,
                        walk_failures=walk_failures,
                        suppress=suppress)

        # reverse the first list, then GRAFT it onto the first bead of the last generated chain
        starting_bead = [self.num_walks, 0]
        self.graft_chain(starting_bead,
                         len(list1),
                         rw_kval,
                         rw_cutoff,
                         rw_epsilon,
                         rw_sigma,
                         list1[::-1],
                         meanfield=meanfield,
                         termination=termination,
                         soften=True, 
                         srate=srate,
                         initial_failures=initial_failures,
                         retraction_limit = retraction_limit,
                         danger = danger,
                         sequence_range = l1seq,
                         walk_failures=walk_failures,
                         suppress=suppress)

    

    #---------------------------------------------------------------------------------------------
    # Data gathering and plotting.

    def walk_data(self, which_ID = None):
        all_data = []
        
        for cell in self.Cells:
            for bead in cell.beads:
                if isinstance(bead[0], int):
                    all_data.append(bead)

                
        all_data.sort(key = lambda x: (x[0], x[1]))
        
        if (which_ID == None):
            return all_data
        else:
            return [i for i in all_data if (i[0] == which_ID)]
    
    def delete_walk(self, walk_id):
        """
        Deletes the walk with the given ID.
        Limitations: should only be used when the walk_id is that of the latest generated walk.
                     This can be fixed by modifying every bead in the box, but this might be fiddly.
        """
        
        dead_walk = self.walk_data(walk_id)
        num_dead = len(dead_walk) # this is the number of beads that have to be deleted

        for i in range(num_dead):
            self.num_beads -= 1
            self.num_walk_beads -= 1 # the individual count beads that make up a walk

            # same method used for retraction
            progress = self.walk_data(walk_id)
            bad_bead = progress[-1]
            current_cell = self.which_cell(bad_bead[-1])
            self.index(current_cell).beads.remove(bad_bead)
        
        self.num_walks -= 1
        self.num_bonds -= num_dead - 1

        if walk_id in self.graft_ids:
            delbond = [[walk_id-1, 0], [walk_id, 0]]
            self.graft_coords.remove(delbond)
            self.graft_ids.remove(walk_id)
            self.num_bonds -= 1

        del self.walkinfo[walk_id]
        

    def crosslink_data(self, which_ID = None):
        all_data = []
        
        for cell in self.Cells:            
            for bead in cell.beads:
                if isinstance(bead[0], str):
                    all_data.append(bead)
                
        all_data.sort(key = lambda x: (x[0], x[1]))
        
        if (which_ID == None):
            return all_data
        else:
            return [i for i in all_data if (i[0] == which_ID)]

    def file_dump(self, filename):
        """
        Dumps all the data encompassed within an object into a file. Very generic. 
        Will eventually evolve into LAMMPS file writer.
        argument: name of the file you're writing to.        
        """
        if self.random_walked == False:
            print("No random walk within system.")
            return 1
        else:
            with open(filename, 'w') as file:
                file.write("SIMULATION OF ENTANGLED POLYMER SYSTEM\n")
                # gathers data using function above.
                all_data = self.walk_data() + self.crosslink_data()
                    
                for data in all_data:
                    data = [str(data[0]),
                            str(data[1]),
                            str(data[2]),
                            str(data[3]),
                            str(data[4]),
                            str(data[5]),
                            str(round(data[-1][0], 8)),
                            str(round(data[-1][1], 8)),
                            str(round(data[-1][2], 8))]
                    data_string = "\t".join(data) + "\n"
                    
                    file.write(data_string.format(1))
                            
            print("Data dump successful.")
            return 0

    def find_low_density(self, region=0, quals=1):
        """
        Used to find the regions of the box with the lowest densities.
        Only to be employed when the box is VERY dense!
        Region: the region over which partial densities of the box are calculated.
        quals: the number of cells allowed to qualify
        """
        low_density = 1
        low_dense_cells = []
        t0 = time.time()
        for i in range(self.cellnums):
            for j in range(self.cellnums):
                for k in range(self.cellnums):
                    cell = self.index([i,j,k])
                    density = len(cell.beads)/self.num_beads
                    if density < low_density:
                        low_density = density
                        low_dense_cells = [[i,j,k]]
                    if density == low_density:
                        low_dense_cells.append([i,j,k])

        t1 = time.time()
        print(f"{t1-t0}")
        
        if region == 0:
            return low_dense_cells
        else:
            t0 = time.time()
            densities = []
            for cell_ind in low_dense_cells:
                # find all cells adjacent by the region unit
                bead_count = 0
                for i in range(-region,region+1):
                    for j in range(-region,region+1):
                        for k in range(-region,region+1):
                            region_index = [(cell_ind[0]+i)%self.cellnums,
                                              (cell_ind[1]+j)%self.cellnums,
                                              (cell_ind[2]+k)%self.cellnums]

                            region_cell = self.index(region_index)                            
                            bead_count+=len(region_cell.beads)
                            
                densities.append((bead_count/self.num_beads, region_index))

            # qualifiers
            densities.sort(key=lambda x: x[0])
            t1 = time.time()
            print(f"{t1-t0}")
            return [i[1] for i in densities][0:quals]

    def file_read(self, filename):
        """ Used to read external data into the box."""
        if self.random_walked == True:
            print("\n#############################################################################")
            print("                                  WARNING!!")
            print("#############################################################################")
            print("There is at least one existing random walk in this box.")
            print("It is likely that self-avoiding constraints will not be effective against the \ncontents of the data file.")
            
        with open(filename) as f:
            read_data = f.readlines()

        read_data = [i.strip() for i in read_data]

        self_data = []
        for i in read_data:
            datum = i.split("\t")
            # length check.
            # gets rid of titles            
            if len(datum) != 6:
                continue
            else:
                self_data.append([int(datum[0]),
                                  int(datum[1]),
                                  datum[2],
                                  np.array([float(datum[3]),
                                            float(datum[4]),
                                            float(datum[5])])])

        for datum in self_data:
            current_cell = self.which_cell(datum[-1])
            self.index(current_cell).beads.append(datum)

    def see_distribution(self,):
        """allows user to view the distribution of beads within the system"""
        from mpl_toolkits import mplot3d
        import matplotlib.pyplot as plt
        
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        for i in range(self.cellnums):
            for j in range(self.cellnums):
                for k in range(self.cellnums):
                    cell = self.index([i,j,k])
                    pos = cell.position
                    ax.scatter3D(pos[0]+0.5*self.cellside,
                                 pos[1]+0.5*self.cellside,
                                 pos[2]+0.5*self.cellside,
                                 s=len(cell.beads)**2)
        plt.show()
    

    # ----------------------------- END OF CLASS ----------------------------



#-------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------
# 
# The code below was the initially employed technique to make sure that the beads were travelling to an
# appropriate region. (this was below the "mean_field == True" condition)
# It didn't work. However, it could be pretty useful?
# 
# bead_subseq = []                 
# manually slice the list and build a subsequence from which the first bead is that of the current index
#            bead_subseq = [bead_sequence[(i+j)%len(bead_sequence)] for j in range(sequence_range) 
#                           if (i+j) < last_element else bead_sequence[(last_element+j)%len(bead_sequence)]
# for j in range(sequence_range):

# last_element = len(bead_sequence) - sequence_range-1 # the last element before
#                                                      # the subsequence terminates
#     if i+j > last_element:
#         bead_subseq.append(bead_sequence[(last_element+j)])
#     else:
#         bead_subseq.append(bead_sequence[(i+j)])
# assumed_type = random.choice(bead_subseq)
# sub_densities = [self.index(index_c).densities[self.interactions.typekeys[i]-1] for i in bead_subseq]
# avg_density = sum(sub_densities)/len(bead_subseq)
# ------------------------------------------------------------------------------
# desnity of the cell in which the trial position resides
