# Program Name: simulation.py
# Author: Aravinthen Rajkumar
# Description: Generates the LAMMPS input and output files for LAMMPS

import time
import numpy as np
import random
import os
import os.path
from os import path
import subprocess
import glob
import shutil
from datetime import date
today = date.today()

class Simulation:
    # stages of the simulation

    def __init__(self, polylattice):
        self.polylattice = polylattice # connects the stonestop box together with polylattice
        # details for writing
        self.equibs = 0 # the number of equilibration procedures 
        
        # file info        
        self.lmp_sim_init = ""  # the simulation init file configuration
        self.file_name = None   # name of file. This is set in settings().
        self.data_file = None   # name of datafile. This is set in structure().
        
        self.dumping = 0 # used for dumping files
        self.data_production = 0

        self.set_settings = False

        # variable below is set True when the types dictionary has been ennumerated.

    def global_bead_num(self, bead):
        """
        Returns a unique ID for a system based on it's walk coordinates.
        
        """
        walk = str(bead[0])
        if walk.isnumeric():
            return self.polylattice.walkinfo[bead[0]-1]+bead[1]        
        if walk[0:2] == "bc":
            return self.polylattice.num_walk_beads + self.polylattice.bclinfo[int(bead[0][2::])-1]+bead[1]
        if walk[0:2] == "uc":
            return self.polylattice.num_walk_beads + self.polylattice.num_bclbeads + self.polylattice.uclinfo[int(bead[0][2::])-1]+bead[1]

        
    def structure(self, datafile=None):
        """
        This is where we dump all of the structural data from polylattice into the LAMMPS file.
        NOTE: this comes FIRST in the simulation's order of precedence!
        """            
        self.polylattice.structure_ready = True


        # builds dictionaries for types
        # the self.polylattice.types dictionary holds the string type and the mass.
        # the numbered type dictionary holdes the number type and the string type.
        numbered_string = ""
        for i in self.polylattice.interactions.used_types:
            numbered_string += f"# {self.polylattice.interactions.typekeys[i]} {i}\n"

        self.data_file = str(datafile) # this sets the data in the class so it can be reused.
        f = open(f"{self.data_file}", "w")
        
        # separate the walks and the crosslinkers so they can be treated differently.
        crosslinker_beads = self.polylattice.crosslink_data()
        walk_beads = self.polylattice.walk_data()
        
        # write the initial 
        f.write(f"\
#-----------------------------------------------------------------------------------           \n\
# NANOPOLY - POLYMER NANOCOMPOSITE STRUCTURAL DATA FILE                                        \n\
#-----------------------------------------------------------------------------------           \n\
# FILENAME:  {self.data_file}                                                                  \n\
# DATE: {today}                                                                                \n\
#-----------------------------------------------------------------------------------           \n\
                                                                                               \n\
{self.polylattice.num_beads} atoms                                                             \n\
{self.polylattice.num_bonds} bonds                                                             \n\
                                                                                               \n\
{len(self.polylattice.interactions.used_types)} atom types                                      \n\
{len(self.polylattice.bonds)} bond types                                            \n\
                                                                                    \n\
0.0000 {self.polylattice.cellside*self.polylattice.cellnums:.4f} xlo xhi            \n\
0.0000 {self.polylattice.cellside*self.polylattice.cellnums:.4f} ylo yhi            \n\
0.0000 {self.polylattice.cellside*self.polylattice.cellnums:.4f} zlo zhi            \n\
                                                                                    \n\
                                                                                    \n\
# Types have been converted numerically as follows:                                 \n\
{numbered_string}\
                                                                                    \n\
Masses                                                                              \n\
                                                                                    \n\
")
        # read in the mass data.
        for i in self.polylattice.interactions.used_types:
            f.write(f"{self.polylattice.interactions.typekeys[i]}\t{self.polylattice.interactions.types[i]}\n")
        f.write("\n")


        f.write("\
Atoms                                                                               \n\
\n")
        # --------------------------------------------------------------------------------------
        # Hierarchy: 1. Random walk beads
        #            2. Grafted chains
        #            3. Bonded crosslinker beads
        #            4. Unbonded crosslinker beads
        #            5. Nanoparticles
        #            Note that grafts are pretty much just random walks anyway, so they don't
        #            need to be treated separately.
        #            Bonded crosslinks are ranked higher than unbonded crosslinks automatically
        # --------------------------------------------------------------------------------------
        
        # READ ATOMS
        for bead in walk_beads:
            bead_num = self.polylattice.walkinfo[bead[0]-1]+bead[1]
            bead_type = self.polylattice.interactions.typekeys[bead[2]] # converted beforehand
            x, y, z = bead[-1][0], bead[-1][1], bead[-1][2]
            f.write(f"\t{bead_num+1}\t{bead[0]}\t{bead_type}\t{x:.5f}\t\t{y:.5f}\t\t{z:.5f}\t\n")

            
        for bead in crosslinker_beads:            
            walk_coordinate = int(bead[0][2::])
            if bead[0][0:2] == "bc":
                addon = self.polylattice.num_walks
                walk = walk_coordinate + addon
                
            if bead[0][0:2] == "uc":
                addon = (self.polylattice.num_walks+self.polylattice.num_bclstructs)
                walk = walk_coordinate + addon
                
            bead_num = self.global_bead_num(bead)                
            bead_type = self.polylattice.interactions.typekeys[bead[2]] # converted beforehand
            x, y, z = bead[-1][0], bead[-1][1], bead[-1][2]
            f.write(f"\t{bead_num+1}\t{walk}\t{bead_type}\t{x:.5f}\t\t{y:.5f}\t\t{z:.5f}\t\n")

        # BOND DATA            
        # reading in the bond data        
        f.write("\
Bonds                                                                               \n\
                                                                                    \n")
        # READ BOND DATA: RANDOM WALKS
        bond = 1
        for walk in range(self.polylattice.num_walks+1):
            for bead in self.polylattice.walk_data(walk)[1::]:
                bead_num = self.global_bead_num(bead)
                bondtype = bead[3]                                
                f.write(f"\t{bond}\t{bondtype}\t{(bead_num)}\t{bead_num+1}\n")
                bond+=1

        # READ BOND DATA: GRAFTS
        for graft in self.polylattice.graft_coords:
            beadnum1 = self.global_bead_num(graft[0])+1
            beadnum2 = self.global_bead_num(graft[1])+1

            # returns the bond detail attached to the first bead of the grafted chain
            bondtype = self.polylattice.walk_data(graft[1][0])[graft[1][1]][3]
            f.write(f"\t{bond}\t{bondtype}\t{(beadnum1)}\t{beadnum2}\n")
            bond+=1
            
        # READ BOND DATA: CROSSLINKS
        for crosslink in self.polylattice.crosslinks_loc:
            beadnum1 = self.global_bead_num(crosslink[0])+1
            beadnum2 = self.global_bead_num(crosslink[2])+1
            linkernum = self.global_bead_num(crosslink[1])+1
            bondtype = crosslink[1][3]
            
            f.write(f"\t{bond}\t{bondtype}\t{(beadnum1)}\t{linkernum}\n")
            bond+=1
            f.write(f"\t{bond}\t{bondtype}\t{linkernum}\t{beadnum2}\n")
            bond+=1

        f.close()

#-----------------------------------------------------------------------------------------------------

    def settings(self, filename=None, dielectric=False):
        """ 
        NOTE: This comes SECOND in the simulation order of precedence!
              Will flag an error if sim_structure hasn't run first.
        This is where we define the header, the units and system settings.        
        Variables - fname: name of the simulation input script file.
                    dielectric: whether the material is dielectric or not.
                                WARNING! needs a bit of confirmation.
        """
        if self.polylattice.structure_ready == False:
            raise EnvironmentError("You must create a structure file before settings can be defined.")
        
        def dictionary_shuffle(my_dict, key_of_list):
            """
            Sorts a dictionary of lists by one of the indices shared in the list.
            Arguments: my_dict     - dictionary that is to ben shuffled
                       key_of_list - key of list in dictionary entry that is to be shuffled
            """
            styles_dict = {}
            for i in my_dict:
                if my_dict[i][key_of_list] not in styles_dict:
                    styles_dict[my_dict[i][key_of_list]] = [[i, my_dict[i]]]
                else:
                    styles_dict[my_dict[i][key_of_list]].append([i, my_dict[i]])

            return styles_dict        
        
        if self.data_file == None:
            print("ERROR: A structure file has not yet been defined.")
            print("Please define one using the simulation.structure command.")
            print("Setup aborted.")
            return 0

        if dielectric:
            atom_style = "hybrid bond dipole" 
        else:
            atom_style = "bond"

        # list of variables used in lammps file
        # need to separate these befoe f_str format doesn't allow them in string.
        # NOTE: you /can/ use curly braces by doubling. I should do this, but i'm in too deep
        
        fname = str(filename)
        self.file_name = fname
        f = open(f"{fname}", "w")
        f.write(f"\
#-----------------------------------------------------------------------------------\n\
# NANOPOLY - POLYMER NANOCOMPOSITE SIMULATION FILE                                  \n\
# Full project files can be found at: https://github.com/aravinthen/nanoPoly        \n\
#-----------------------------------------------------------------------------------\n\
# FILENAME: {self.file_name}                                                        \n\
# DATE: {today}                                                                     \n\
# OVERVIEW:                                                                         \n\
#    Number of polymer chains:       {self.polylattice.num_walks}                   \n\
#    Number of bonded crosslinks:    {self.polylattice.num_bclbeads}                \n\
#    Number of unbonded crosslinks:  {self.polylattice.num_uclbeads}                \n\
#    Number of grafted chains:       {self.polylattice.num_grafts}                  \n\
#    Nanostructure:                  {self.polylattice.nanostructure}               \n\
#-----------------------------------------------------------------------------------\n\
                                                                                      \n\
variable structure index {self.data_file}                                             \n\
variable simulation index {self.file_name}                                            \n\
                                                                                      \n\
# Initialization settings                                                             \n\
units              lj                                                                 \n\
boundary           p p p                                                              \n\
atom_style         {atom_style}                                                       \n\
log                log.${{simulation}}.txt                                            \n\
                                                                                      \n\
# read data from object into file                                                     \n\
")
        if self.polylattice.cl_unbonded == True:
# 
            f.write(f"\
read_data          ${{structure}} extra/special/per/atom 100 extra/bond/per/atom 100  \n\
")
        else:
            f.write(f"\
read_data          ${{structure}}                                                     \n\
")
                                                   
        f.write(f"\n\
# define interactions                                                                 \n\
neighbor      0.4 multi                                                               \n\
neigh_modify  every 10 one 10000                                                      \n\
")

        # add pair interactions of different beads
        for i in range(np.shape(self.polylattice.interactions.type_matrix)[0]):
            for j in range(i, np.shape(self.polylattice.interactions.type_matrix)[0]):
                typestring = self.polylattice.interactions.type_matrix[i,j]
                type1, type2 = typestring.split(",")
                num1 = self.polylattice.interactions.typekeys[type1]
                num2 = self.polylattice.interactions.typekeys[type2]
                
                energy = self.polylattice.interactions.return_energy(type1, type2)
                sigma = self.polylattice.interactions.return_sigma(type1, type2)
                cutoff = self.polylattice.interactions.return_cutoff(type1, type2)
                
                f.write(f"\
pair_style    lj/cut {cutoff}                                                         \n\
pair_coeff    {num1} {num2} {round(energy,5)} {sigma}                                  \n\
")

        f.write("\n")
        
        # now it's time to read in the bond information.
        styles_dict = dictionary_shuffle(self.polylattice.bonds, -1)        
        for style in styles_dict:
            f.write(f"\
bond_style    {style}  \n")
            if style == 'fene':                
                f.write(f"\
special_bonds fene  \n")
            for data in styles_dict[style]:
                f.write(f"\
bond_coeff    {data[0]+1} {np.around(data[1][0], decimals=4)} {np.around(data[1][1],decimals=5)} {np.around(data[1][2], decimals=5)} {np.around(data[1][3],decimals=5)} \n")
                
        # ensures that the settings have been set.
        # this must be true for equilibration to take place.
        self.set_settings = True

        f.close()

    def equilibrate(self, steps, timestep, temp, dynamics, bonding=False, final_temp=None, damp=10.0, tdamp=None, pdamp=None, drag=2.0, output_steps=100, dump=0, data=('step','temp','press'), seed=random.randrange(0,99999), reset=True, description=None):
        """
        ARGUMENTS:
        steps      - the number of steps taken in the equilibration
        temp       - the temperature at which the equilibration will take place
        dynamics   - the type of equilibration that'll be taking place
                     'langevin' uses the nve and langevin thermostats
                     'npt' carries out npt dynamics
        bonding    - dependent on whether crosslinking is performed unbonded.
        final_temp - used for heating and cooling equilibrations
        """
        self.equibs+=1
        
        if final_temp==None:            
            final_temp=temp

        if tdamp==None:
            tdamp=100*timestep
        if pdamp==None:
            pdamp=1000*timestep
            
        data_string = " ".join(data)
        
        if not self.set_settings:
            self.equibs-=1
            raise Exception("Settings have not yet been defined.")

        f = open(self.file_name, 'a')
        
        f.write(f"\n\
#---------------------------------------------------------------------------------------------------\n\
# EQUILIBRATION STAGE {self.equibs}                                                                 \n")
                
        if description != None:
            f.write("# Description: {description}\n")

        f.write("\
#---------------------------------------------------------------------------------------------------\n\
")
        if dynamics=='langevin':            
            if dump>0:
                self.dumping = 1
                f.write(f"\
compute         1 all stress/atom NULL \n\
dump            1 all cfg {dump} dump.{self.file_name}_*.cfg mass type xs ys zs fx fy fz c_1[1] c_1[2] c_1[3] \n\
\n\
")
            f.write(f"\
velocity        all create {float(temp)} 1231 \n\
fix             1 all nve/limit {np.amax(self.polylattice.interactions.cutoff_matrix)}\n\
fix             2 all langevin {float(temp)} {float(final_temp)} {damp} {seed}\n\
")
            if bonding == True:
                if self.polylattice.cl_unbonded == False:
                    raise EnvironmentError("Unbonded crosslinks are not present in the Polylattice!\n")
                else:
                    b_data = self.polylattice.cl_bonding
                    # jparam : the allowed beads
                    for jparam in b_data[2]:
                        type1 = self.polylattice.interactions.typekeys[jparam]
                        type2 = self.polylattice.interactions.typekeys[b_data[0]]
                        if b_data[6] == None:
                            f.write(f"\
fix             {type1+2} all bond/create 1 {type2} {type1} {b_data[3]} {b_data[1]} iparam {b_data[4]} {type2} jparam {b_data[5]+2} {type1} \n\
")
                        else:
                            f.write(f"\
fix             {type1+2} all bond/create 1 {type2} {type1} {b_data[3]} {b_data[1]} prob {b_data[6]} {seed} iparam {b_data[4]} {type2} jparam {b_data[5]+2} {type1}\n\
") 
            
            f.write(f"\
thermo_style    custom {data_string} \n\
thermo          {output_steps}\n\
")
            if reset == True:
                f.write(f"\
reset_timestep  0\n\
")
            f.write(f"\
run             {steps} \n\
unfix 1         \n\
unfix 2         \n\
")
            if bonding == True:
                for i in range(1, self.polylattice.interactions.num_types):
                    f.write(f"\
unfix {i+2} \n\
")                    

            f.write(f"\
write_restart   restart.{self.file_name}.polylattice{self.equibs}\n\
")
        if dynamics=='nose_hoover':
            if dump>0:
                self.dumping = 1
                f.write(f"\
dump            1 all cfg {dump} dump.{self.file_name}_*.cfg mass type xs ys zs fx fy fz\n\
\n\
")
            f.write(f"\
fix             1 all npt temp {float(temp)} {float(final_temp)} {tdamp} iso 0 0 {pdamp} drag {drag} \n\
fix             2 all momentum 1 linear 1 1 1\n\
thermo_style    custom {data_string} \n\
thermo          {output_steps}\n\
")
            if reset == True:
                f.write(f"\
reset_timestep  0\n\
")
            f.write(f"\
run             {steps} \n\
unfix 1         \n\
unfix 2         \n\
write_restart   restart.{self.file_name}.polylattice{self.equibs}\n\
")
            f.close()

    def deform(self, steps, timestep, strain, temp, final_temp=None, damp=None, datafile=True, output_steps=100, reset=True, data=('step','temp','lx', 'ly', 'lz', 'pxx','pyy', 'pzz',), seed=random.randrange(0,99999), description = None):
        """
        Carries out the deformation of the box. 
        Note: strain MUST be a six-dimensional list/vector.
        """
        if not self.set_settings:
            self.equibs-=1
            raise Exception("Settings have not yet been defined.") 

        if final_temp == None:
            final_temp = temp

        f = open(self.file_name, 'a')
        f.write(f"\n\
#---------------------------------------------------------------------------------------------------------\n\
# DEFORMATION STAGE                                                                                       \n")
        if description != None:
            self.lmp_sim_init += f"# Description: {description}\n"

        f.write("\
#---------------------------------------------------------------------------------------------------------\n\
")
        data_string = " ".join(data)
        
        if damp==None:
            damp=100*timestep
        
        f.write(f"\
run             0            \n\
velocity        all create {float(temp)} 1231 \n\
fix             1 all langevin {float(temp)} {float(final_temp)} {damp} {seed} \n\
fix             2 all nve/limit {self.polylattice.lj_sigma}\n\
fix		3 all deform 1 x trate {strain[0]} y trate {strain[1]} z trate {strain[2]} units box remap x \n\
")
        if datafile==True:
            self.data_production = 1
            title = "\"" + "\t".join(data) + "\""
            for i in data:
                f.write(f"\
variable        var_{i} equal \"{i}\"              \n\
")
            file_string = "\"${var_"+ "}\t${var_".join(data) + "}\""

            f.write(f"\
fix             datafile all print {output_steps} {file_string} file {self.file_name}.deform.data title {title} screen no \n\
")
            
        f.write(f"\
thermo_style    custom {data_string}                   \n\
thermo          {output_steps}                         \n\
")
        if reset == True:
            f.write(f"\
reset_timestep  0\n\
")
        f.write(f"\
run             {steps}                                \n\
unfix 1                                                \n\
unfix 2                                                \n\
unfix 3                                                \n\
")
        if datafile == True:
            f.write(f"\
unfix datafile \n\
")
            
    def run(self, folder=None, mpi=0):
        """
        Carries out the LAMMPS run.
        """
        correct_dumping = 0
        if self.dumping == 1 or self.data_production == 1:
            if folder==None:
                print("No directory for dumping: files will be dumped in working directory.")
                print("Note that this is usually considered a bad move.")
            else:
                if not path.exists(f"{folder}"):
                    os.system(f"mkdir {folder}")
                    correct_dumping = 1
                else:
                    os.system(f"rm -r {folder}")
                    os.system(f"mkdir {folder}")
                    correct_dumping = 1
        else: 
            if folder != None:
                print("You have not selected any options for dumping: no dump files will be produced.")

        if mpi==0:
            print(f"Running {self.file_name}")
            os.system(f"lmp < {self.file_name}")
        else:
            print(f"Running {self.file_name} with parallel LAMMPS implementation.")
            print(f"Number of cores: {mpi}.")
            os.system(f"mpirun -np {mpi} lmp -in {self.file_name}")

        if correct_dumping == 1:
            if self.data_production == 1:
                os.system(f"mv *.data* {folder}")   
            if self.dumping == 1:
                os.system(f"mv *dump.* {folder}")   
 
            os.system(f"mv *restart.* {folder}")
            os.system(f"mv *log.* {folder}")
            os.system(f"cp {self.file_name} {folder}")
            os.system(f"cp {self.data_file} {folder}")

    def view(self, view_path, sim_file):
        bashcommand = f"{view_path} {sim_file}"

        os.system(bashcommand)
        
