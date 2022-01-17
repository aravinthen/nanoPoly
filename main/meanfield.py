# Program Name: scf.py
# Author: Aravinthen Rajkumar
# Description: This is the library in which self-consistent field theory routines are included.

import numpy as np
import sys
import os
import os.path
from os import path

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

#--------------------------------------------------------------------------------------------------
# Problem initialisation
#--------------------------------------------------------------------------------------------------
    def parameters(self,
                   pname,
                   unit,
                   chi,
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


        num_chains = len(args)
        chi = format(chi, "E")
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
        {chi}                                                          \n\
                                                                       \n\
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
        with open(density_file, 'r') as f:
            x = 0
            y = 0
            z = 0

            # incrementally
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
                    
                unit_index.append([[x,y,z], density_data])
                
                x+=1                
                if x > unit[0]-1:
                    x = 0
                    y +=1
                if y > unit[1]-1:
                    x = 0
                    y = 0
                    z +=1
                    
                count+=1

        num_types = np.size([unit_index[0][-1]])
        density_fields = [np.zeros(unit) for i in range(num_types)]
        
        for i in unit_index:
            for j in range(num_types):
                density_fields[j][i[0][0], i[0][1], i[0][2]] = i[-1][j]
        # now you have a list of arrays that contain the density information PER type


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
