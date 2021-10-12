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

        self.param_file = None
        self.guess_file = None

        self.guessed = False

        self.unit = None

    def guess_field(self, guess_file):
        self.guessed =True
        self.guess_file = guess_file


    def parameters(self,
                   pname,
                   unit,
                   crystal,
                   group_name,
                   error_max=1,
                   max_itr=100,
                   ncut=100,                   
                   *args):
        # arguments:
        # unit - the dimesions of the mean field unit
        #        this must be a divisible unit of the main box
        # space_group - parameter file
        # crystal - crystal system
        # *args = the number and details of chains in the system

        num_chains = len(args)

        for i in range(3):
            if self.polylattice.cellnums % unit[i] != 0:
                raise SystemError("Unit cell indices must divide the polylattice box.")

        types = set()
        for i in args:
            for j in i:
                types.add(j)
        N_monos = len(list(types))        
        
        self.param_file = pname
        f = open(self.param_file, "w")           
        f.write(f"\
format  1  0                                                   \n\
                                                               \n\
MONOMERS                                                       \n\
N_monomer                                                      \n\
        {N_monos}   \n\
kuhn                                                         \n\
  1.0000000E+00  1.0000000E+00  1.0000000E+00                \n\
                                                             \n\
CHAINS                                                       \n\
N_chain                                                      \n\
        {num_chains}                                   \n\
")

        # Specifying chain structure
        
        # no indexing if there's only one chain within the system

        if num_chains == 1:
            f.write(f"\
N_block                                                    \n\
        {n_block}                                            \n\
")

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

            if num_chains > 1:
                f.write(f"\
N_block({chain_num})                                           \n\
        {n_block}                                            \n\
")                
            f.write(f"\
block_monomer                                                \n\
{block_monomer} \n\
block_length    \n\
{block_length}  \n\
")

            # solvents: not included
            
        f.write(f"                                           \n\
SOLVENTS                                                     \n\
N_solvent                                                    \n\
              0                                              \n\
                                                             \n\
COMPOSITION                                                  \n\
ensemble                                                     \n\
              0                                              \n\
")

        f.write(f"                                           \n\
phi_chain                                                    \n\
  1.0000000E+00                                              \n\
                                                             \n\
INTERACTION                                                  \n\
interaction_type                                             \n\
          'chi'                                              \n\
chi                                                          \n\
  1.3000000E+01                                              \n\
  3.5000000E+01  1.3000000E+01                               \n\
                                                             \n\
")

        f.write(f"                                           \n\
UNIT_CELL                                                    \n\
dim                                                          \n\
              3                                              \n\
crystal_system                                               \n\
        {crystal}                                            \n\
N_cell_param                                                 \n\
              1                                              \n\
cell_param                                                   \n\
  2.1276672E+00                                              \n\
                                                             \n\
DISCRETIZATION                                               \n\
ngrid                                                        \n\
             16             16             16                \n\
chain_step                                                   \n\
  1.0000000E-02                                              \n\
                                                             \n\
BASIS                                                        \n\
group_name                                                   \n\
     {group_name}                                            \n\
                                                             \n\
ITERATE                                                      \n\
input_filename                                               \n\
          {self.guess_file}                                  \n\
output_prefix                                                \n\
             'out/'                                          \n\
max_itr                                                      \n\
            {max_itr}                                        \n\
")
        error_max= format(error_max, "E")
        f.write(f"\
error_max                                                    \n\
            {error_max}                                      \n\
domain                                                       \n\
              T                                              \n\
itr_algo                                                     \n\
           'NR'                                              \n\
N_cut                                                        \n\
            {ncut}                                           \n\
                                                             \n\
FINISH")

        f.close()

    def run(self,):
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
                                
                    
        
#                
                
        print(f"{count} density values read into box.")
        
        self.density = True
