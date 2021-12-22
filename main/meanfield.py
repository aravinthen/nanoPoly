# Program Name: scf.py
# Author: Aravinthen Rajkumar
# Description: This is the library in which self-consistent field theory routines are included.

import numpy as np

class MeanField:
    """
    Used to read in density files into polyboxes
    """
    def __init__(self, polylattice):
        self.polylattice = polylattice
        self.density = False
        self.paramn_file = None
        self.model_file = None
        self.guess_file = None
        self.guessed = False
        self.unit = None

#----------------------------------------------------------------------------------------------------------------- 
# Problem initialisation
#----------------------------------------------------------------------------------------------------------------- 
    def parameters(self,
                   pname,
                   gname,
                   unit,
                   chi,
                   crystal,
                   chain_step,
                   group_name,                   
                   *args,
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

        types = set()
        for i in args:
            for j in i:
                types.add(j)
        N_monos = len(list(types))        
        
        self.param_file = pname
        self.guess_file = gname
        f = open(self.param_file, "w")           
        f.write(f"\
format  1  0                                                           \n\
")
        f.write(f"\
MONOMERS                                                               \n\
N_monomer                                                              \n\
        {N_monos}                                                      \n\
kuhn                                                                   \n\
  1.0000000E+00  1.0000000E+00  1.0000000E+00                          \n\
                                                                       \n\
CHAINS                                                                 \n\
N_chain                                                                \n\
        {num_chains}                                                   \n\
")

        # Specifying chain structure        
        # no indexing if there's only one chain within the system

        if num_chains == 1:
            n_block = len(args[0])
            block_monomer = ""
            block_length = ""
            for bead_type in args[0]:
                conv_type = self.polylattice.interactions.typekeys[bead_type[0]]
                block_monomer += f"              {conv_type}"
                num = format(bead_type[1], "E")
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
            for chain in args:
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

        # solvents: not supported for nanoPoly
        f.write(f"                                                     \n\
SOLVENTS                                                               \n\
N_solvent                                                              \n\
              0                                                        \n\
                                                                       \n\
COMPOSITION                                                            \n\
ensemble                                                               \n\
              0                                                        \n\
")

        chain_num = 0
        for chain in args:
            chain_num += 1            
            f.write(f"\
phi_chain({chain_num})                                                 \n\
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
        {crystal}                                                      \n\
N_cell_param                                                           \n\
              1                                                        \n\
cell_param                                                             \n\
  2.1276672E+00                                                        \n\
                                                                       \n\
DISCRETIZATION                                                         \n\
ngrid                                                                  \n\
             {unit[0]}             {unit[1]}             {unit[2]}     \n\
chain_step                                                             \n\
        {chain_step}                                                        \n\
                                                                       \n\
BASIS                                                                  \n\
group_name                                                             \n\
     '{group_name}'                                                    \n\
                                                                       \n\
ITERATE                                                                \n\
input_filename                                                         \n\
          {self.guess_file}                                            \n\
output_prefix                                                          \n\
             'out/'                                                    \n\
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
            {ncut}                                                     \n\
                                                                       \n\
FIELD_TO_RGRID                                                         \n\
input_filename                                                         \n\
            'out/rho'                                                  \n\
input_filename                                                         \n\
            'rgrid'                                                    \n\
FINISH")

        f.close()

#----------------------------------------------------------------------------------------------------------------- 

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
output_file           {self.guess_file}            \n\
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

    def run(self,):
        # Run both pscf and pscfFieldGen.
        pass
         
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
        with open(density_file, 'r') as f:
            # check if the number of beads in file match with the number of beads in interaction data
            file_beads = len(f.readline().strip().split("\t"))
            if file_beads != self.polylattice.interactions.num_types:                
                raise EnvironmentError("Defined bead types and file bead types do not match.")

        unit_index = []
        with open(density_file, 'r') as f:
            x = 0
            y = 0
            z = 0

            # incrementally
            for line in f:
                datum = line.strip().split("\t")
                density_data = np.array([float(i) for i in datum])
                
                if any(i < 0.0 for i in density_data):
                    raise EnvironmentError("Unphysical densities contained with density file.")
                
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
