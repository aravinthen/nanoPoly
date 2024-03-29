# Program Name: scf.py
# Author: Aravinthen Rajkumar
# Description: This is the library in which self-consistent field theory routines are included.

import numpy as np
import sys
import os
import os.path
from os import path
from scipy.interpolate import RegularGridInterpolator

class MeanField:
    """
    Used to read in density files into polyboxes
    """
    def __init__(self, polylattice):
        self.polylattice = polylattice
        self.density = False
        self.param_file = None
        self.model_file = None
        self.kgrid_file = None
        self.rgrid_file = None
        self.omega_file = None
        self.guessed = False
        self.unit = None

        self.fhparams = {}
        self.num_fh = 0

        # these attributes are used for the purpose of interpolation
        # they aren't necessary in basic applications, so their use is considered optional
        self.interpolated = False
        self.density_funcs = None
        
        self.max_dranges = {} # the biggest difference within density values per cell

    def setFL(self, type1, type2, fl):
        # sets the flory huggins parameters for a pair of beads
        # entry exists for both a,b and b,a. This is lazy, but it works
        pair1 = (self.polylattice.interactions.typekeys[type1]-1, self.polylattice.interactions.typekeys[type2]-1)
        pair2 = (self.polylattice.interactions.typekeys[type2]-1, self.polylattice.interactions.typekeys[type1]-1)

        self.fhparams[pair1] = fl
        self.fhparams[pair2] = fl

        self.num_fh += 1
        

#--------------------------------------------------------------------------------------------------
# Problem initialisation
#--------------------------------------------------------------------------------------------------
    def parameters(self,
                   pname,
                   unit,
                   crystal,
                   chain_step,
                   group_name,                   
                   *args,
                   cell_param = 2.127667,
                   error_max=1.0,
                   max_itr=100,
                   ncut=100):
        # arguments:
        # unit         - the dimesions of the mean field unit
        #                (this must be a divisible unit of the main box)
        # chi          - flory huggins parameter
        # space_group  - parameter file
        # crystal      - crystal system
        # chain_step   - this should be the inverse of the number of beads in a chain
        # *args        - the number and details of chains in the system

        if self.num_fh == 0:
            raise SystemError("Flory-Huggins parameters must be defined for all types.")

        num_chains = len(args)
        chain_step = format(chain_step, "E")

        for i in range(3):
            if self.polylattice.cellnums % unit[i] != 0:
                raise SystemError("Unit cell indices must divide the polylattice box.")
        
        chains = []
        types = dict()
        for i in args:
            single_chain = []
            for j in i:
                for k in j:
                    if len(j) != 3:
                        single_chain.append((k[0], 1.0, k[2]))
                        types[k[0]] = 1.0
                    else:
                        single_chain.append(k)
                        types[k[0]] = k[1]

                    
            chains.append(single_chain)

        N_monos = len(list(types))
        mono_string = "\t" + "\t".join([format(types[i], "E") for i in types])
        
        self.param_file = pname
        self.kgrid_file = pname + "_kgrid"
        self.rgrid_file = pname + "_rgrid"
        self.rho_file = pname + "_rho.in"
        self.omega_file = pname + ".omega.in"
        f = open(self.param_file, "w")        
        f.write(f"\
format  1  0                                                           \n\
                                                                       \n\
")
        f.write(f"\
MONOMERS                                                               \n\
N_monomer                                                              \n\
        {N_monos}                                                      \n\
kuhn                                                                   \n\
  {mono_string}                                                        \n\
                                                                       \n\
CHAINS                                                                 \n\
N_chain                                                                \n\
        {num_chains}                                                   \n\
")

        # Specifying chain structure        
        # no indexing if there's only one chain within the system

        if num_chains == 1:
            n_block = len(chains[0])
            block_monomer = ""
            block_length = ""
            for bead_type in chains[0]:
                conv_type = self.polylattice.interactions.typekeys[bead_type[0]]
                block_monomer += f"              {conv_type}"
                num = format(bead_type[2], "E")
                block_length +=  f"              {format(num)}"

            f.write(f"\
N_block                                                                \n\
        {n_block}                                                      \n\
block_monomer                                                          \n\
        {block_monomer}                                                \n\
block_length                                                           \n\
        {block_length}                                                 \n\
")

        else:
            chain_num = 0
            for chain in chains:
                chain_num += 1
                n_block = len(chain)
                block_monomer = ""
                block_length = ""
                for bead_type in chain:
                    conv_type = self.polylattice.interactions.typekeys[bead_type[0]]
                    block_monomer += f"              {conv_type}"
                    num = format(bead_type[1], "E")
                    block_length +=  f"              {format(num)}"

            f.write(f"\
N_block({chain_num})                                                   \n\
        {n_block}                                                      \n\
")                
            f.write(f"\
block_monomer                                                          \n\
{block_monomer}                                                        \n\
block_length                                                           \n\
{block_length}                                                         \n\
")


        f.write(f"                                                     \n\
COMPOSITION                                                            \n\
ensemble                                                               \n\
              0                                                        \n\
")

        chain_num = 0
        for chain in chains:
            chain_num += 1            
            f.write(f"\
phi                                                                    \n\
  1.0000000E+00                                                        \n\
                                                                       \n\
")

        f.write(f"                                                     \n\
INTERACTION                                                            \n\
interaction_type                                                       \n\
          'chi'                                                        \n\
chi                                                                    \n\
")
  
        num_types = self.polylattice.interactions.num_types

        # building a list of type pairs
        types_list = []
        for i in range(num_types):
            for j in range(i+1, num_types):
                types_list.append((i,j))

        ftypes_list = [] # the actual list of types
        for i in types_list:
            if i not in ftypes_list and (i[1], i[0]) not in ftypes_list:
                ftypes_list.append(i)

        # putting the types in the correct printing format
        for i in range(1,num_types):    
            print_list = []
            for j in ftypes_list:
                if i in j and all(k<i+1 for k in j):
                    print_list.append(str(format(float(self.fhparams[j]), "E")))
            chi = "\t".join(print_list)
            f.write(f"\
            {chi}                                                          \n\
")

        f.write(f"                                                     \n\
UNIT_CELL                                                              \n\
dim                                                                    \n\
              3                                                        \n\
crystal_system                                                         \n\
        '{crystal}'                                                    \n\
N_cell_param                                                           \n\
              1                                                        \n\
cell_param                                                             \n\
  {cell_param}                                                        \n\
                                                                       \n\
DISCRETIZATION                                                         \n\
ngrid                                                                  \n\
             {unit[0]}             {unit[1]}             {unit[2]}     \n\
chain_step                                                             \n\
        {chain_step}                                                   \n\
                                                                       \n\
BASIS                                                                  \n\
group_name                                                             \n\
     '{group_name}'                                                    \n\
                                                                       \n\
KGRID_TO_RGRID                                                         \n\
input_filename                                                         \n\
        '{self.kgrid_file}'                                            \n\
output_filename                                                        \n\
        '{self.rgrid_file}'                                            \n\
                                                                       \n\
RGRID_TO_FIELD                                                         \n\
input_filename                                                         \n\
        '{self.rgrid_file}'                                            \n\
output_filename                                                        \n\
        '{self.rho_file}'                                              \n\
                                                                       \n\
RHO_TO_OMEGA                                                           \n\
input_filename                                                         \n\
        '{self.rho_file}'                                              \n\
output_filename                                                        \n\
        '{self.omega_file}'                                            \n\
                                                                       \n\
ITERATE                                                                \n\
input_filename                                                         \n\
          '{self.omega_file}'                                          \n\
output_prefix                                                          \n\
          ''                                                           \n\
max_itr                                                                \n\
            {max_itr}                                                  \n\
")

        error_max = format(error_max, "E")
        f.write(f"\
error_max                                                              \n\
            {error_max}                                                \n\
domain                                                                 \n\
              T                                                        \n\
itr_algo                                                               \n\
           'NR'                                                        \n\
N_cut                                                                  \n\
            {ncut}                                                   \n\
                                                                       \n\
FIELD_TO_RGRID                                                         \n\
input_filename                                                         \n\
            'rho'                                                  \n\
input_filename                                                         \n\
            'rgrid'                                                    \n\
                                                                       \n\
FINISH")

        f.close()

#----------------------------------------------------------------------------------------------------

    def model_field(self, model_file, core_type, posn_list):
        # build the model file required to run pscfFieldGen
        # pscfFieldGen will not be run in this function: only the file is generated

        self.guessed =True
        self.model_file = model_file
        
        if self.param_file == None:
            raise EnvironmentError("Run the parameters() function first.")

        f = open(self.model_file, "w")           
        f.write(f"\
software              pscf                         \n\
parameter_file        {self.param_file}            \n\
output_file           {self.kgrid_file}            \n\
structure_type        particle                     \n\
core_monomer          {core_type}                  \n\
coord_input_style     basis                        \n\
N_particles           {len(posn_list)}            \n\
particle_positions                                 \n\
")
        for pos in posn_list:
            f.write(f"\
        {pos[0]}   {pos[1]}   {pos[2]}             \n\
")

        f.write(f"\
finish                                             \n\
")

    def run(self, path_to_pscf, param_file=None, guess_file=None):
        """
        Run both pscf and pscfFieldGen.
        """
        # Guess field generation.
        # pscfFieldGen MUST be included within the python path!!!!
        
        if self.guessed == True:
            os.system(f"python3 -m pscfFieldGen -f {self.model_file}")
        else:
            if guess_file == None:
                raise Exception('You need to provide a guess file for pscf to work!')

        # Density file generation
        if param_file==None:
            os.system(f"{path_to_pscf}/pscf < {self.param_file}")
        else:
            os.system(f"{path_to_pscf}/pscf < {param_file}")
         
    def density_file(self,
                     density_file,
                     unit=None):
        """
        subdivision: allows the unit cell to be split into multiple polylattice
        """
        if self.polylattice.interactions.num_types == 0:
            raise EnvironmentError("Types must be defined before setting densities.")

        if unit==None and self.unit==None:
            raise EnvironmentError("Unit dimensions have to be defined if parameter function has not been used.")

        if self.unit != None:
            unit = self.unit

        count = 0
            # check if the number of beads in file match with the number of beads in interaction data
#             file_beads = len(f.readline().strip().split("\t"))
#             if file_beads != self.polylattice.interactions.num_types:
#                 raise EnvironmentError("Defined bead types and file bead types do not match.")

        unit_index = [] 
        # this process essentially finds the array indices of the line-by-line density file.
        # first argument is the index of the density box
        # second argument is the density at said box.
        numtypes = self.polylattice.interactions.num_types
        density_difference = {}
        for i in range(numtypes):
            for j in range(i+1, numtypes):
                density_difference[(i,j)] = 0.0
                
        with open(density_file, 'r') as f:
            x = 0
            y = 0
            z = 0

            for line in f:
                datum = line.strip().split()

                not_numeric = False
                                
                for i in datum:
                    if not i[0].isnumeric():
                        not_numeric = True
                        break
                
                if not_numeric == True:
                    continue

                if len(datum) == 1:
                    continue
                    
                if [int(float(i)) for i in datum] == unit:
                    continue

                density_data = np.array([float(i) for i in datum])                    
                for i in range(numtypes):
                    for j in range(i+1, numtypes):
                        diff = abs(density_data[i] - density_data[j])
                        if diff > density_difference[(i,j)]:
                            density_difference[(i,j)] = diff

                unit_index.append([[x,y,z], density_data])
               
                x+=1                
                if x > unit[0]-1:
                    x = 0
                    y += 1
                if y > unit[1]-1:
                    x = 0
                    y = 0
                    z += 1
                    
                count+=1

        
        inv_types = dict((v-1, k) for k, v in self.polylattice.interactions.typekeys.items())
        for i in density_difference:
            self.max_dranges[(inv_types[i[1]],
                              inv_types[i[0]])] = density_difference[i]
            self.max_dranges[(inv_types[i[0]],
                              inv_types[i[1]])] = density_difference[i]

        # for self-ranges, just use max difference.
        for i in range(numtypes):
            self.max_dranges[(inv_types[i],
                              inv_types[i])] = 1.0

        num_types = np.size([unit_index[0][-1]])
        density_fields = [np.zeros(unit) for i in range(num_types)]
        
        # this converts the density list calculated above into a pair of arrays.
        for i in unit_index:    
            for j in range(num_types):
                density_fields[j][i[0][0], i[0][1], i[0][2]] = i[-1][j]

        # fills in a single unit cell at each point
        # x,y,z = the coordinates where a unit cell begins or repeats itself.
        # i, j, k = the coordinates of the unit cell that are being repeated over
        for x in range(0, self.polylattice.cellnums, unit[0]):
            for y in range(0, self.polylattice.cellnums, unit[1]):
                for z in range(0, self.polylattice.cellnums, unit[2]):
                    for i in range(unit[0]):
                        for j in range(unit[1]):
                            for k in range(unit[2]):
                                cell = [x+i,y+j,z+k]
                                density_data = [data[i,j,k] for data in density_fields]
                                self.polylattice.index(cell).densities = density_data                                   

        print(f"{count} density values read into box.")
        
        self.density = True


    def interpolate(self,):
        # creates interpolation functions based on the density information within the walk
        # It is necessary for a density file to be read into the system before this can be done!

        # error messages
        if self.density == False:
            raise SystemError("A density file must be read into the box first!")

        # obtain the mesh data
        init_size = self.polylattice.cellnums # the actual size of the box

        # these define the points along the axes upon which the interpolation is defined.
        x = np.array([(0.5+i)*self.polylattice.cellside for i in range(-1, init_size+1)])
        y = np.array([(0.5+i)*self.polylattice.cellside for i in range(-1, init_size+1)])
        z = np.array([(0.5+i)*self.polylattice.cellside for i in range(-1, init_size+1)])

        # build a mesh-grid with the density values
        # the meshgrid has to be a bit larger than normal in order to interpolate at higher values.

        # this is a list that contains N meshgrids, where N is the number of types within the system.
        densities = [np.zeros([init_size+2,
                               init_size+2,
                               init_size+2]) for i in range(-1, self.polylattice.interactions.num_types)]

        for i in range(-1, init_size+1):
            for j in range(-1, init_size+1):
                for k in range(-1, init_size+1):
                    for val in range(self.polylattice.interactions.num_types):
                        xv = i % init_size
                        yv = j % init_size
                        zv = k % init_size
                        densities[val][i,j,k] = self.polylattice.index([xv,yv,zv]).densities[val]

        # create a dictionary for each type that contains an interpolator function
        density_funcs = {}
        for i in range(self.polylattice.interactions.num_types):
            density_function = RegularGridInterpolator((x, y, z), densities[i])
            # note that i+1 corresponds to the enumerated types found in interactions.typekeys
            density_funcs[i+1] = density_function # this can now be called as any standard function

        self.density_funcs = density_funcs
        # set interpolated to true: to be used in random walk functions
        self.interpolated = True

        
    # ----------------------------------------------------------------------------------------------
    def density_search(self, beadtype, upper):
        # returns the list of cell indices that have a minimum of a specified
        good_spots = []
        for i in range(self.polylattice.cellnums):
            for j in range(self.polylattice.cellnums):
                for k in range(self.polylattice.cellnums):
                    cell = [i,j,k]
                    density = self.polylattice.index(cell).densities[self.polylattice.interactions.typekeys[beadtype]-1]
                    if density >= upper:
                        good_spots.append(cell)

        return good_spots

    def density_eval(self, position):
        current_cell = self.polylattice.which_cell(position)
        surround_ind = ([(current_cell[0]+i)%self.polylattice.cellnums,
                         (current_cell[1]+j)%self.polylattice.cellnums,
                         (current_cell[2]+k)%self.polylattice.cellnums] for i in range(-1,2) for j in range(-1,2) for k in range(-1,2))

        region_ds = [0.0 for i in range(self.polylattice.interactions.num_types)]
        for ind in surround_ind:
            for i in range(self.polylattice.interactions.num_types):
                region_ds[i] += self.polylattice.index(ind).densities[i]

        return np.array(region_ds)/(len(region_ds)-1)
            
        
    def assign_limit(self, beadtype, limit):
        valid_cells = 0
        for i in range(self.polylattice.cellnums):
            for j in range(self.polylattice.cellnums):
                for k in range(self.polylattice.cellnums):
                    cell = [i,j,k]
                    density = self.polylattice.index(cell).densities[self.polylattice.interactions.typekeys[beadtype]-1]
                    if density >= limit:
                        valid_cells += 1

        print(f"Fraction of box available for type {beadtype}: {valid_cells/(self.polylattice.cellnums)**3}")

        self.polylattice.interactions.assign_limit(beadtype, limit)
