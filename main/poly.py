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

from simulation import Simulation
from analysis import Check, Percolation

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
    
    sim_attr = Simulation
    che_attr = Check
    per_attr = Percolation
    
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

    class Interactions:
        """All the interactions between beads are stored here."""
        def __init__(self, max_dist):
            self.types = {}
            self.typekeys = {}
            self.type_matrix = None
            self.sigma_matrix = None   # distances between types
            self.energy_matrix = None  # interaction energies between types
            self.cutoff_matrix = None  # cutoff between types
            self.used_types = []        # the types that are actually used in the simulation
            self.max_dist = max_dist
            self.num_types = 0 # the number of types present in the box

        def newType(self, type_id, mass, potential, *args):
            """
            type_name: the id of the new type
            potential: TUPLE, three numbers.
            """
            if not type_id.isalnum():
                raise TypeError("Names must be composed of only alphanumeric characters.")

            if potential[0]*1.12234 > self.max_dist:
                error1 = "Error! The LJ-distance minimum is greater than the box grid spacing."
                error2 = "This means that random walk algorithm may not detect beads of this type."
                error3 = "Please respecify the number of cells within the box."
                raise SystemError(f"\n{error1}\n{error2}\n{error3}")
            
            if self.num_types == 0:
                if len(args) > 0:
                    raise TypeError("No other types have been defined in the system.")
                self.types[type_id] = mass
                self.typekeys[type_id] = self.num_types+1
                self.type_matrix = np.array([f"{type_id},{type_id}"])
                
                self.sigma_matrix = np.array([potential[0]])
                self.energy_matrix = np.array([potential[1]])
                self.cutoff_matrix = np.array([potential[2]])
                self.num_types+=1                
            else:
                self.types[type_id] = mass
                self.typekeys[type_id] = self.num_types+1
                self.num_types+=1
        
                # modify the type matrix
                type_matrix = np.empty((self.num_types,self.num_types), dtype=np.dtype('U50'))

                # list of types
                types_list = [i for i in self.types]
                # type connections are built in                
                for i in range(self.num_types):
                    for j in range(self.num_types):
                        type_matrix[i,j] = f'{types_list[i]},{types_list[j]}'

                self.type_matrix = type_matrix
                            
                # modify the sigma matrix
                sigma_matrix = self.sigma_matrix
                ##  build new columns and rows onto the energy matrix
                sigma_matrix = np.column_stack((sigma_matrix,
                                                np.array([0 for i in range(self.num_types-1)])))
                sigma_matrix = np.vstack((sigma_matrix,
                                          np.array([0 for i in range(self.num_types)])))
                # assign sigma
                sigma_matrix[self.num_types-1,self.num_types-1] = potential[0]
                
                self.sigma_matrix = sigma_matrix
                
                # modify the energy matrix
                energy_matrix = self.energy_matrix                
                ##  build new columns and rows onto the energy matrix
                energy_matrix = np.column_stack((energy_matrix,
                                                np.array([0 for i in range(self.num_types-1)])))
                energy_matrix = np.vstack((energy_matrix,
                                          np.array([0 for i in range(self.num_types)])))                
                # assign energy
                energy_matrix[self.num_types-1,self.num_types-1] = potential[1]
                
                self.energy_matrix = energy_matrix
                
                # modify the cutoff matrix
                cutoff_matrix = self.cutoff_matrix
                ##  build new columns and rows onto the energy matrix
                cutoff_matrix = np.column_stack((cutoff_matrix,
                                                np.array([0 for i in range(self.num_types-1)])))
                cutoff_matrix = np.vstack((cutoff_matrix,
                                          np.array([0 for i in range(self.num_types)])))
                # assign cutoff 
                cutoff_matrix[self.num_types-1,self.num_types-1] = potential[2]                
                self.cutoff_matrix = cutoff_matrix
                
                #----------------------------------------------------------------------------------
                # now deal with the interactions for different kinds of beads
                for i in args:
                    name = i[0].replace(" ","")

                    # reverse name has to be broke part, reversed, then joined.
                    revname = i[0].replace(" ","")
                    revname = revname.split(",")
                    revname = ",".join(revname[::-1])
                    properties = i[1]
                    
                    if name not in self.type_matrix or revname not in self.type_matrix:
                        raise TypeError("Type combination does not exist in system.")

                    if type_id not in name:
                        raise TypeError("Unrelated type interaction defined. Interactions must always contain the new type..")
                    
                    ni, nj = np.where(self.type_matrix == name)
                    ri, rj = np.where(self.type_matrix == revname)

                    # now with the array indices, simply slot everything into place.
                    self.sigma_matrix[ri[0], rj[0]] = properties[0]
                    self.sigma_matrix[ni[0], nj[0]] = properties[0]
                    
                    self.energy_matrix[ri[0], rj[0]] = properties[1]
                    self.energy_matrix[ni[0], nj[0]] = properties[1]

                    self.cutoff_matrix[ri[0], rj[0]] = properties[2]
                    self.cutoff_matrix[ni[0], nj[0]] = properties[2]

        def return_sigma(self, type1, type2):
            # returns the interaction between two beads
            namestring = f"{type1},{type2}"
            ni, nj = np.where(self.type_matrix == namestring)
            return self.sigma_matrix[ni,nj][0]

        def return_energy(self, type1, type2):
            # returns the interaction between two beads
            namestring = f"{type1},{type2}"
            ni, nj = np.where(self.type_matrix == namestring)
            return self.energy_matrix[ni,nj][0]
        
        def return_cutoff(self, type1, type2):
            # returns the interaction between two beads
            namestring = f"{type1},{type2}"
            ni, nj = np.where(self.type_matrix == namestring)
            return self.cutoff_matrix[ni,nj][0]

    def __init__(self, boxsize, cellnums=1.0):
        """        
        cellside: Length of the side of a cells
        cellnums: Number of cells in one-dimension.
        sigma: the pairwise  for each particle in the box
        epsilon: lennard jones energy
        celltotal: total number of cells in lattice.
        """
        self.boxsize = boxsize # NOT GUARANTEED!!!!        
        self.cellnums = cellnums # the lj distance between two 
        self.cellside = self.boxsize/self.cellnums
        self.celltotal = self.cellnums**3
        
        self.crossings = np.zeros((3, 1)) # number of crossings in each direction, [x, y, z]
                                          # encapsulates boundary conditions


        # initializing the interactions class
        self.interactions = self.Interactions(self.cellside)

        # attached libraries
        self.simulation = Simulation(self) # the simulation class
        self.check = Check(self) # the Check subclass
        self.percolation = Percolation(self) # the Percolation class        
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
    
    def random_walk(self, numbeads, Kval, cutoff, energy, sigma, bead_sequence, mini=1.12234, style='fene', phi=None, theta=None, cell_num=None, starting_pos=None, restart=False, termination=None, allowed_failures=10000):
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

        if self.structure_ready == True:
            raise EnvironmentError("Structures cannot be built or modified when simulation procedures are in place.")
        
        ID = self.num_walks + 1
        if termination == "None":
            print("Warning: if this random walk is unsucessful, the program will terminate.")

        
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
            starting_pos = np.array(starting_pos)
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
                
            if issues>0:
                print("Provided starting position is too close to another bead.")
                print("Please choose a more appropriate position.")
                raise Exception("Program terminated.")

            else:
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
            if cell_num != None:
                current_cell = np.array(cell_num)
            else:
                current_cell = np.array([random.randrange(0, self.cellnums),
                                         random.randrange(0, self.cellnums),
                                         random.randrange(0, self.cellnums)])
                
            

            cell_pos = self.index(current_cell).position
            cell_bound = cell_pos + self.cellside
            
            # trial for the initial position
            invalid_start = True
            total_failure = False
            failure = 0
            while invalid_start:            
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

                if issues > 0:                    
                    invalid_start = True
                    failure+=1
                    
                    if failure > allowed_failures:
                        if restart == True:
                            print(f"Failure tolerance reached at random walk {self.num_walks}.")
                            print("Restarting the walk is recommended.")
                            total_failure = False
                            failure = 0
                            issues = 0
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

            self.num_beads+=1
            self.num_walk_beads += 1 # the individual count beads that make up a walk
            self.index(current_cell).beads.append(bead_data)
        
        # Begin loop here.
        bead_number = 1
        bond = mini*sigma # the minimum of the LJ potential
        i = 1
        current_pos = starting_pos
        
        while i < numbeads:            
            too_close = True # used to check if the cell is too close
            generation_count = 0 # this counts the number of random vectors generated.
                                 # used to raise error messages.
            bead_type = bead_sequence[(i % len(bead_sequence))]
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

                if issues > 0:      
                    too_close = True
                    current_pos = previous
                else:
                    too_close = False

                # This has to be here: the failure condition is False when generation_count = 0
                generation_count += 1

                # FAILURE CONDITIONS -------------------------------------------------------------
                if generation_count % allowed_failures == 0:
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
                                         termination=termination)                    
                        return 0
                    
                    elif termination == "retract":
                        print(f"Retracting at bead {i} of random walk {ID}")                        
                        # adjust walk positions                        
                        i = i - 1
                        self.num_beads -= 1
                        
                        progress = self.walk_data(ID)
                        bad_bead = progress[-1]
                        current_cell = self.which_cell(bad_bead[-1])

                        self.index(current_cell).beads.remove(bad_bead)

                        progress = self.walk_data(ID)
                        current_pos = progress[-1][-1]

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
                        
                # -----------------------------------------------------------------------------------
                
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
        return 1

    def graft_chain(self, starting_bead, num_beads, Kval, cutoff, energy, sigma, bead_sequence, mini=1.12234, style='fene', phi=None, theta=None, cell_num=None, allowed_failures=10000):
        """
        This method is used to grow extra beads at a particular point in a given chain.
        Intended to study the effects of different chain architectures on macroscopic properties.

        num_beads: the number of beads on the grafted chain
        bead_types: the bead type sequence on the grafted chain
        starting_bead: the bead that the new chain will be grafted to.

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
        bond = mini*sigma

        # employ the same mechanism as in random_walk to find a suitable position.
        too_close = True
        generation_count = 0        
        while too_close:
            trial = new_position(position, bond, 1, self.cellside*self.cellnums, phi, theta)
            
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
                sigma = self.interactions.return_sigma(bead_sequence[0], j[2])
                if distance < sigma:                        
                    issues += 1
                    break

            if issues > 0:      
                too_close = True
                current_pos = previous
            else:
                too_close = False

            # This has to be here: the failure condition is False when generation_count = 0
            generation_count += 1
            
            if generation_count % allowed_failures == 0:
                print("Position for graft bead not found. Consider reattempting with a sparser box.")
                print("Graft unsuccessful.")
                raise Exception("Program terminated.")

        # the algorithm should by now have returned a valid position for the random walk. (or failed)
        # now, all that's left is to run a random walk from this position.
        graft_pos = current_pos
        
        self.random_walk(num_beads,
                         Kval,
                         cutoff,
                         energy,
                         sigma,
                         bead_sequence,
                         mini=1.12234,
                         style='fene',
                         phi=phi,
                         theta=theta,
                         starting_pos=list(graft_pos),
                         restart=False,
                         termination=None,
                         allowed_failures=10000)

        gbead_0 = [self.num_walks, 0]
        self.graft_coords.append([starting_bead, gbead_0])
        self.num_grafts += 1
        self.num_bonds += 1
            


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


    # ----------------------------- END OF CLASS ----------------------------
