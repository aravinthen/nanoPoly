# Program Name: polylattice.py
# Author: inthen Rajkumar
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
import sys
import multiprocessing
import random

sys.path.append(".")
from simulation import Simulation

class PolyLattice:
    """
    The PolyLattice class initialises an array composed of cells.
    Each cell is identified by it's position and a number of attributes.
    Attributes for a PolyLattice: 1. cellside = sidelength of induvidual cell
                                  2. cellnum = the number of cells in each dimensionpp
                                  3. celltotal = total number of cells.
                                  4. Cells = list of cells in lattice. each list element is Cell class

    Attributes for a Cell:        1. i, j, j, ijk: indices (and tuple of indices).
                                  2. position: the position of cell origin relative to lattice origin
                                  3. cellsides: inherited from PolyLattice. Same as cellside
                                  4. forbidden: BOOLEAN -  Cannot host atoms if this is valued True
    """
    
    sim_attr = Simulation
    
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
            # only one crosslinker allowed in every cell.        
            self.cl_pos = None
            # multiple beads allowed in a cell.
            self.beads = []
            
    def __init__(self, boxsize, cutoff, sigma, epsilon):
        """        
        cellside: Length of the side of a cells
        cellnums: Number of cells in one-dimension.
        sigma: the pairwise  for each particle in the box
        epsilon: lennard jones energy
        celltotal: total number of cells in lattice.
        """
        self.boxsize = boxsize # NOT GUARANTEED!!!!
        
        self.lj_param = sigma # the lj distance between two atoms
        self.lj_cut = cutoff
        self.lj_energy = epsilon
        self.crossings = np.zeros((3, 1)) # number of crossings in each direction, [x, y, z]
                                          # encapsulates boundary conditions

        self.cellside = 1.01*self.lj_param
        self.cellnums = m.ceil(self.boxsize/self.cellside)
        self.celltotal = self.cellnums**3
        
        # bond details
        self.bonds = None # 
        
        # crosslink details
        self.crosslink_id = None # this is the "CHAIN" for the crosslinks
        self.crosslinks_loc = []
        # unbonded crosslinks
        self.cl_unbonded = False # Will be set to true once unbonded crosslinks are added
        self.cl_bonding = None # saves the bond configuration for use in the simulation file

        # bead types and masses
        self.types = None # this is meant to be a dictionary
                          # format: "type": mass
                          # "type" is numerical
        
        self.random_walked = False
        self.walkinfo = [] # this is supposed to store the walk as well as the number of
                            # beads in that walk.
                            # used to control the bond writing process in the simulation class
        
        self.nanostructure = None
        self.simulation = Simulation(self) # attaches to the simulation class

        # counts
        self.num_walks = 0
        self.num_bonds = 0
        self.num_beads = 0

        # this bit of code builds the cells within the lattice.
        self.Cells = []
        
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

        surround_ind = []
        for i in range(-1,2):
            for j in range(-1,2):
                for k in range(-1,2):
                    surround_ind.append([(cell_index[0]+i)%self.cellnums,
                                         (cell_index[1]+j)%self.cellnums,
                                         (cell_index[2]+k)%self.cellnums])
                    
        bead_lists = []
        for cell in surround_ind:
            bead_lists.append(self.index(cell).beads)

        surrounding_beads = []
        for beadlist in bead_lists:
            for bead in beadlist:
                surrounding_beads.append(bead)
        
        return surrounding_beads

    def random_walk(self, numbeads, Kval, cutoff, energy, sigma, mini=1.12234, style='fene', phi=None, theta=None, bead_types=None, cell_num=None, termination=None, allowed_failures=10000):
        """
        Produces a random walk.
        If cell_num argument is left as None, walk sprouts from a random cell.
        Otherwise, walk begins at specified cell.
        PARAMETER - beads: number of beads 
        PARAMETER -controls the controls the  Kval: repulsion constant
        PARAMETER - cutoff: the extension limiter for the chain
        PARAMETER - sigma:  The distance value of the LJ potential.
                            This is *not* the distance between the atoms.
        PARAMETER - energy: the energy of the bond in LJ units.
        PARAMETER - phi: the torsion angle of the system.
        PARAMETER - theta: the bond angle of the system.
        PARAMETER - ID: identifies the walk.
        PARAMETER - cell_num: defaulted as None, but this can be set to an index.        
                    MUST be a list.
        PARAMETER - bead_types: defines and identifies the beads and their types.
                    MUST BE A DICTIONARY.
        PARAMETER - termination: THREE OPTIONS - a) None, b) Retract and c) Break
                    
        Notes: Nested functions prevent the helper functions from being used outside of scope.       
        """

        ID = self.num_walks + 1
        if termination == "None":
            print("Warning: if this random walk is unsucessful, the program will terminate.")
        if sigma < self.lj_param:
            print("You cannot set the bond length to be less than the pair potential")
            print("length. Either modify the bond length or increase the pair")
            print("potential length in the PolyLattice box.")
        
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


        # bead type dictionary
        if bead_types == None:            
            bead_types[1] = 1.0
        else:
            for i in bead_types:
                if not str(i).isdigit():
                    raise TypeError("Bead types must be integers.")
            bead_types = bead_types

        self.types = bead_types

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
        
        if cell_num == None:            
            current_cell = np.array([random.randrange(0, self.cellnums),
                                      random.randrange(0, self.cellnums),
                                      random.randrange(0, self.cellnums)])

        else:
            current_cell = np.array(cell_num)


        # Initial values of the random walk
        cell_pos = self.index(current_cell).position
        cell_bound = cell_pos + self.cellside

        bead_number = 0

        # trial for the initial position
        current_pos = np.array([random.uniform(cell_pos[0], cell_bound[0]),
                                random.uniform(cell_pos[1], cell_bound[1]),
                                random.uniform(cell_pos[2], cell_bound[2])])
                    
        invalid_start = True
        while invalid_start:
            index_c = self.which_cell(current_pos)                    
            neighbours = self.check_surroundings(current_pos)
            if len(neighbours)==0:
                invalid_start = False
            else:
                for j in neighbours:
                    index_n = self.which_cell(j[-1])
                    if self.cellnums-1 in np.abs(index_c - index_n):
                        period = np.array([i if abs(i) == (self.cellnums-1) else 0 for i in (index_n - index_c)])
                        real_j = -period*(self.cellnums/(self.cellnums - 1))*self.cellside + j[-1]
                    else:
                        real_j = j[-1]
                
                    if (np.linalg.norm(np.array(current_pos) - np.array(real_j)) < self.lj_param):
                        current_pos = np.array([random.uniform(cell_pos[0], cell_bound[0]),
                                                random.uniform(cell_pos[1], cell_bound[1]),
                                                random.uniform(cell_pos[2], cell_bound[2])])
                    else:
                        invalid_start = False

        # The full data entry for keeping track of the bead.
        bead_data = [ID, bead_number, 1, bead_types[1], bond_type, current_pos]
        self.index(current_cell).beads.append(bead_data)
        
        # Begin loop here.
        bond = mini*sigma # the minimum of the LJ potential
        i = 1
        while i < numbeads:            
            too_close = True # used to check if the cell is too close
            generation_count = 0 # this counts the number of random vectors generated.
                                 # used to raise error messages.
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
                neighbours = self.check_surroundings(current_pos)
                
                issues = 0
                index_c = self.which_cell(current_pos)
                for j in neighbours:            
                    if len(j) == 0:
                        continue

                    index_n = self.which_cell(j[-1])
                    # deals with instances of periodicity, no different from the code above.
                    # to obtain the constant, calculate the maximum norm of two adjacent cell indices
                    
                    if self.cellnums-1 in np.abs(index_c - index_n):
                        period = np.array([i if abs(i) == (self.cellnums-1) else 0 for i in (index_n - index_c)])
                        real_j = -period*(self.cellnums/(self.cellnums - 1))*self.cellside + j[-1]
                    else:
                        real_j = j[-1]                        
                
                    
                    if (np.linalg.norm(np.array(current_pos) - np.array(real_j)) < self.lj_param):
                        issues += 1
                        break
                        
                if issues > 0:
                    too_close = True
                    current_pos = previous
                else:
                    too_close = False

                generation_count += 1
                if generation_count % allowed_failures == 0:
                    if termination == "break":
                        new_numbeads = numbeads - i
                        self.num_beads += i
                        self.num_bonds += i - 1
                        self.num_walks += 1
                        print(f"Breaking random walk! There are now {self.num_walks} chains in the system.")
                        print(f"New chain length: {new_numbeads}")
                        self.random_walk(new_numbeads, Kval, cutoff, energy, sigma, mini=mini, style=style, bead_types=bead_types, termination=True)                    
                        return 0
                    
                    elif termination == "retract":
                        print(f"Retracting at bead {i} of random walk {ID}")                        
                        # adjust walk positions                        
                        i = i - 1
                        
                        progress = self.walk_data(ID)
                        bad_bead = progress[-1]
                        current_cell = self.which_cell(bad_bead[-1])

                        self.index(current_cell).beads.remove(bad_bead)

                        progress = self.walk_data(ID)
                        trial_pos = progress[-1][-1]

                    else:                        
                        raise Exception("Program terminated.")
                        print("")
                        print("Warning for Random Walk ID: " + str(ID))
                        print("The system has now generated %d unsuccessful positions for this walk." % (generation_count))
                        print("It's very likely that the random walk has either become trapped or")
                        print("that the lattice has become too dense to generate a valid position.")
                        print("It is highly recommended to restart the program with:")
                        print("a) a fewer number of atoms,")
                        print("b) a larger amount of lattice space,")
                        print("c) a smaller step distance.")
                        
            current_pos = trial_pos

            current_cell = self.which_cell(current_pos)
            bead_data = [ID,
                         i,
                         (i % len(bead_types))+1,
                         bead_types[(i % len(bead_types))+1],
                         bond_type,
                         current_pos]
            
            self.index(current_cell).beads.append(bead_data)
            i+=1

        self.num_walks += 1
        self.num_beads += numbeads
        self.num_bonds += numbeads - 1
        self.random_walked = True

        self.walkinfo.append([ID, numbeads])
        return 1


    def unbonded_crosslinks(self, crosslinks, mass, Kval, cutoff, energy, sigma, bdist=None, prob=None, style='fene', allowed=None, ibonds=2, jbonds=1):
        """ 
        This method initializes crosslinks that will move around the box before bonding within the simulation.
        This makes use of the "self.unbonded_links" paramater, which stores the properties of the bonds that will be created.
        Arguments- crosslinks:     The number of crosslinks in the system.
                   sigma:           The main range at which molecules can be linked.
                   mass:           Mass of the crosslinker cell
                   allowed:        A list of bead types that are allowed to crosslink with the unbonded beads.
                   bdist:          the minimum bonding distance
                   prob:           the probability of linking
                   numlinks:       the number of links the crosslink atom is allowed to make.
        """

        if bdist == None:
            bdist=sigma

        if allowed == None:
            allowed = [i for i in self.types]
            
        # adding the bond detail of the crosslink to the system        
        repeat = 0
        for i in range(len(self.bonds)):
            if self.bonds[i] != (Kval, cutoff, energy, sigma, style):
                repeat += 1

        if repeat >= len(self.bonds):
            self.bonds[len(self.bonds)] = (Kval, cutoff, energy, sigma, style)
            
        bond_type = [i for i in range(len(self.bonds)) if self.bonds[i]==(Kval, cutoff, energy, sigma, style)][0]+1
        bead_type = len(self.types)+1

        bond = sigma
        self.types[bead_type] = mass

        # Flagging up the presence of unbonded crosslinks within the system
        self.cl_unbonded = True
        self.cl_bonding = [bead_type, bond_type, allowed, bdist, ibonds, jbonds,prob]

        for i in range(crosslinks):
            # pick a random cell
            current_cell = np.array([random.randrange(0, self.cellnums),
                                     random.randrange(0, self.cellnums),
                                     random.randrange(0, self.cellnums)])
        
            cell_pos = self.index(current_cell).position
            cell_bound = cell_pos + self.cellside

            # trial for the initial position
            trial_pos = np.array([random.uniform(cell_pos[0], cell_bound[0]),
                                  random.uniform(cell_pos[1], cell_bound[1]),
                                  random.uniform(cell_pos[2], cell_bound[2])])

            invalid_pos = True
            while invalid_pos:
                index_c = self.which_cell(trial_pos)                    
                neighbours = self.check_surroundings(trial_pos)
                if len(neighbours)==0:
                    invalid_pos = False
                else:
                    for j in neighbours:
                        index_n = self.which_cell(j[-1])
                        if self.cellnums-1 in np.abs(index_c - index_n):
                            period = np.array([i if abs(i) == (self.cellnums-1) else 0 for i in (index_n - index_c)])
                            real_j = -period*(self.cellnums/(self.cellnums - 1))*self.cellside + j[-1]
                        else:
                            real_j = j[-1]
                
                        if (np.linalg.norm(np.array(trial_pos) - np.array(real_j)) < self.lj_param):
                            trial_pos = np.array([random.uniform(cell_pos[0], cell_bound[0]),
                                                  random.uniform(cell_pos[1], cell_bound[1]),
                                                  random.uniform(cell_pos[2], cell_bound[2])])
                        else:
                            invalid_pos = False

                bead_data = [self.num_walks,
                             0,
                             len(self.types),
                             mass,
                             0,
                             trial_pos]
        
                self.index(current_cell).beads.append(bead_data)
                self.num_beads += 1

        
                

    def bonded_crosslinks(self, crosslinks, mass, Kval, cutoff, energy, sigma, style='fene', forbidden=None, reaction=True, fail_count=30, selflinking=10, multilink=False):
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
                   selflinking:   allows the chain to link with itself after a series of links have passed.
                   multilink:     allows a bead to link multiple times.                   
        """
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
            valid = False

            all_beads = [bead for beads in accepted_links for bead in beads]

            count = 0
            if multilink==False:
                while not valid:
                    count+=1
                    link = random.choice(potential_pairs)
                    
                    # makes sure the links haven't already been accepted                
                    cond1 = (link[0] in all_beads) or (link[1] in all_beads)                 
                    if not cond1:
                        valid = True

                    if count>len(potential_pairs):
                        return 0
                        break
                
            else:
                raise EnvironmentError("Multilinking not yet supported!")
            return link
            
        
        if self.random_walked == False:
            raise EnvironmentError("Box has not been populated with a random walk. Please use the random walk method and try again.")

        # adding the bond detail of the crosslink to the system
        repeat = 0
        for i in range(len(self.bonds)):
            if self.bonds[i] != (Kval, cutoff, energy, sigma, style):
                repeat += 1
        if repeat >= len(self.bonds):
            self.bonds[len(self.bonds)] = (Kval, cutoff, energy, sigma, style)
        bond_type = [i for i in range(len(self.bonds)) if self.bonds[i]==(Kval, cutoff, energy, sigma, style)][0]+1
        bond = sigma

        # ---------------- compiling the full list of potential pairs. ------------------
        potential_pairs = []        
        for i in range(self.num_walks):
            for bead in self.walk_data(i):
                if (bead[2] not in forbidden):
                    neighbours = self.check_surroundings(bead[-1])
                    for neighbour in neighbours:
                        # considering neighbours that are higher in the index, neighbours are further along the chain
                        # not of the same random walk.
                        cond1 = ((neighbour[0] > bead[0]) or (neighbour[1] - bead[1] > selflinking)) 
                        cond2 = (np.linalg.norm(neighbour[-1] - bead[-1]) < 2*bond) # within linking distance
                        cond3 = (neighbour[2] not in forbidden) # allowed to link                    
                        
                        if cond1 and cond2 and cond3:
                            potential_pairs.append((neighbour, bead))
        # -------------------------------------------------------------------------------

        if len(potential_pairs) == 0:
            # terminating condition. Extremely useful!
            print("No sites for crosslinking. Simulation will run as planned.")
            return 1
        else:
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

                            if np.linalg.norm(real_neighbour - pot_cross_loc) < self.lj_param:
                                issue = 1
                                break

                        if issue == 1:
                            invalid = True # not necessary, but readability is key
                            rot_failed += 1
                        else:
                            invalid = False
                            rot_failed = 0 # resets failure counter

                        if rot_failed > 50:
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
                        if len(accepted_links) == 1:
                            # we have crosslinkers - they can be added to the types dictionary
                            self.types[len(self.types)+1] = mass
                            
                        bead = [self.num_walks,
                                0,
                                len(self.types),
                                mass,
                                bond_type,
                                pot_cross_loc]
                        
                        self.num_beads += 1
                        self.num_bonds += 2
                        
                        # adds bead to the lattice cell that it's in
                        current_cell = self.which_cell(bead[-1])
                        self.index(current_cell).beads.append(bead)
                        crosslink_data.append([link[0], bead, link[1]])

                # adds crosslinker atom to bead-type list
                self.crosslinks_loc += crosslink_data
                

                # termination condition
                # can be used to allow control flow operations
                if complete_failure == 1:
                    return len(accepted_links)/crosslinks # fuzzy condition, allows an evaluation of how many links were generated
                else:
                    return 0
                
            else:
                # generic network crosslinking
                self.crosslinks_loc = crosslink_data
        

                
    def verify(self,):
        """
        A verification function. How safe is my molecular dynamics simulation?
        Returns True if the atoms are all far enough to be safely simulated.
        """
        print("Verification of system:")
        all_data = self.walk_data()
        closest = np.linalg.norm(all_data[0][-1] - all_data[1][-1])
        catom1 = 0
        catom2 = 1
        
        datom1 = 0 
        datom2 = 1
        
        # used to calculate statistics
        average = 0
        num_distances = 0
        
        for i in range(len(all_data)):
            for j in range(i+1, len(all_data)):
                distance = np.linalg.norm(all_data[i][-1] - all_data[j][-1])
                if distance < closest:
                    closest = distance
                    catom1 = i                    
                    catom2 = j
                    
                    datom1 = all_data[i][-1]
                    datom2 = all_data[j][-1]
                    
                average += distance
                num_distances += 1

        average = average/num_distances
        print(f"Closest distance: {round(closest,5)} [between {catom1} and {catom2}]")

        if closest < self.lj_param:
            print("The closest distance is much less than the specified Lennard-Jones parameter.")
            print("This is a dangerous simulation to run. Rerunning nanoPoly with different")
            print("initial values is highly encouraged.")
            return False
        else:
            return True

    #---------------------------------------------------------------------------------------------
    # Data gathering and plotting.

    def walk_data(self, which_ID = None):
        all_data = []
        
        for cell in self.Cells:
            for bead in cell.beads:
                all_data.append(bead)
                
        all_data.sort(key = lambda x: (x[0], x[1]))
        
        if (which_ID == None):
            return all_data
        else:
            return [i for i in all_data if (i[0] == which_ID)]


    def plot_walk(self, ID, ax, col, bonds = False, size=50):
        if ID == 'S':
            ID = sys.maxsize
        my_walk = [i[-1] for i in self.walk_data(ID)]
        
        xvals = []
        yvals = []
        zvals = []

        for i in my_walk:
            xvals.append(i[0])
            yvals.append(i[1])
            zvals.append(i[2])
        
        col= col
        ax.scatter(xvals, yvals, zvals, c=col, s=size)
        if (bonds == True):
            ax.plot(xvals, yvals, zvals, color=col)

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
                all_data = self.walk_data()
                    
                for data in all_data:
                    if data[0] == self.crosslink_id:
                        print("Needs to be types+1")
                        bead_ID = 'C'
                    else:
                        bead_ID = str(data[0])
                        
                    data = [str(data[0]),
                            str(data[1]),
                            str(data[2]),
                            str(data[3]),
                            str(data[4]),
                            str(round(data[-1][0], 8)),
                            str(round(data[-1][1], 8)),
                            str(round(data[-1][2], 8))]
                    data_string = "\t".join(data) + "\n"
                    
                    file.write(data_string.format(1))
                            
            print("Data dump successful.")
            return 0

    def file_read(self, filename):
        """ Used to read external data into the box."""
        if self.random_walked == True:
            print("\n#############################################################################")
            print("                                  WARNING!!")
            print("#############################################################################")
            print("There is at least one existing random walk in this box.")
            print("It is likely that self-avoiding constraints will not be effective against the \ncontents of the data file.")
            print("Uploading the data file before running a random walk is highly suggested.")
            
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


    # ----------------------------- END OF CLASS ----------------------------
