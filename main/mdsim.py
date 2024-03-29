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

class MDSim:
    # stages of the simulation

    def __init__(self, polylattice):
        self.polylattice = polylattice # connects the stonestop box together with polylattice
        # details for writing
        self.equibs = 0 # the number of equilibration procedures 
        
        # file info        
        self.lmp_sim_init = ""  # the simulation init file configuration
        self.file_name = None   # name of file. This is set in settings().
        self.data_file = None   # name of datafile. This is set in structure().
        
        self.dumping = 0 # used to turn on dumping
        self.global_dump = 0 # used to move dump files into a folder
        self.data_production = 0
        self.deform_count = 0 # used to generate multiple deformation_files

        self.pending_mods = []

        self.set_settings = False


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


    def modify_interaction(self, type1, type2,
                           new_sigma = None, new_energy = None, new_cutoff = None,
                           new_n = None, new_alpha = None, new_lambda = None):
        """
        This function will be used to change the properties of the beads during simulation.
        identifier - the interaction that has to be changed, must be in string format       
        new_sigma, new_energy, new_cutoff - self explanatory
        """

        if new_sigma == None and new_energy == None and new_cutoff == None and new_n == None and new_alpha == None and new_lambda == None:
            raise EnvironmentError("No changes are specified.")

        for mod in self.pending_mods:
            if (type1, type2) == (mod[0], mod[1]):
                raise EnvironmentError("Multiple modifications are being made to the same interaction.")

        # retrieve the initial interaction        
        current_vals = [self.polylattice.interactions.return_sigma(type1, type2),
                        self.polylattice.interactions.return_energy(type1, type2),
                        self.polylattice.interactions.return_cutoff(type1, type2),
                        self.polylattice.interactions.return_n(type1, type2),
                        self.polylattice.interactions.return_alpha(type1, type2),
                        self.polylattice.interactions.return_lambda(type1, type2)]        
    
        new_vals = [new_sigma, new_energy, new_cutoff, new_n, new_alpha, new_lambda]

        for i in range(6):
            if new_vals[i] == None:
                new_vals[i] = current_vals[i]

        self.pending_mods.append([type1, type2, current_vals, new_vals])
        

    def run_modifications(self, total_steps, spacing,
                          temp, timestep, final_temp=None, 
                          damp=10, tdamp=None, pdamp=None, drag=2.0,
                          scale=True,
                          data=('step','temp','press', 'pe', 'ke'), output_steps=100,
                          dump=0, seed = random.randrange(0,99999), description=None):
        """
        Runs all of the changes stored up in the 'bead_mod' list.
        steps - total number of steps in which the interaction will be changed,
        spacing - number of steps per interval
        """
        if len(self.pending_mods) == 0:
            raise EnvironmentError("No modifications have been specified.")

        if final_temp==None:            
            final_temp=temp
            
        if tdamp==None:
            tdamp=100*timestep
            
        if pdamp==None:
            pdamp=1000*timestep

        if not self.set_settings:
            self.equibs-=1
            raise Exception("Settings have not yet been defined.")

        data_string = " ".join(data)

        f = open(self.file_name, 'a')

        f.write(f"\n\
#---------------------------------------------------------------------------------------------------\n\
# POTENTIAL MODIFICATION STAGE                                                                 \n")
                
        if description != None:
            f.write("# Description: {description}\n")

        f.write("\
#---------------------------------------------------------------------------------------------------\n\
")

        diffs = []
        for i in self.pending_mods:
            changes = []
            for j in range(6):
                changes.append(spacing*(i[-1][j] - i[-2][j])/total_steps)
            diffs.append(changes)

        for i in range(total_steps//spacing):
            for j in range(len(self.pending_mods)):
                for k in range(6):
                    self.pending_mods[j][-2][k] = round(self.pending_mods[j][-2][k] + diffs[j][k], 5)

                num1 = self.polylattice.interactions.typekeys[self.pending_mods[j][0]]
                num2 = self.polylattice.interactions.typekeys[self.pending_mods[j][1]]
                    
                sigma = self.pending_mods[j][-2][0]
                energy = self.pending_mods[j][-2][1]
                cutoff = self.pending_mods[j][-2][2]
                n = self.pending_mods[j][-2][3]
                alpha = self.pending_mods[j][-2][4]
                lmbda = self.pending_mods[j][-2][5]

                if int(n) == 0 and int(alpha) == 0 and int(lmbda) == 1:
                    f.write(f"\
pair_style    lj/cut {cutoff}                                                         \n\
pair_coeff    {num1} {num2} {round(energy,5)} {sigma}                                  \n\
")
                else:
                    f.write(f"\
pair_style    lj/cut/soft {n} {alpha} {cutoff}                                         \n\
pair_coeff    {num1} {num2} {round(energy,5)} {sigma} {lmbda}                         \n\
")               

                    
            if dump>0:
                self.global_dump = 1
                self.dumping = 1
                f.write(f"\
dump            1 all cfg {dump} dump.{self.file_name}_*.cfg mass type xs ys zs fx fy fz\n\
")

            if scale:
                f.write(f"\
velocity        all scale {temp} \n")
            else:
                f.write(f"\
velocity        all create {float(temp)} {seed} \n")                

            f.write(f"\
fix             1 all nve/limit {np.amax(self.polylattice.interactions.cutoff_matrix)}\n\
fix             2 all langevin {float(temp)} {float(final_temp)} {damp} {seed}\n\
")
            f.write(f"\
thermo_style    custom {data_string} \n\
thermo          {output_steps}\n\
run             {spacing} \n\
unfix 1         \n\
unfix 2         \n\
")

            if self.dumping == 1:
                self.dumping = 0
                f.write(f"\
undump 1        \n\
")
            f.write(f"\n")


        if (total_steps % spacing != 0):
            remaining_steps = total_steps % spacing
            for j in range(len(self.pending_mods)):            
                num1 = self.polylattice.interactions.typekeys[self.pending_mods[j][0]]
                num2 = self.polylattice.interactions.typekeys[self.pending_mods[j][1]]                
                sigma = self.pending_mods[j][-1][0]
                energy = self.pending_mods[j][-1][1]
                cutoff = self.pending_mods[j][-1][2]
                
            f.write(f"\
pair_style    lj/cut/soft {n} {alpha} {cutoff}                                         \n\
pair_coeff    {num1} {num2} {round(energy,5)} {sigma} {lmbda}                         \n\
")               
            
            if dump>0:
                self.global_dump = 1
                self.dumping = 1
                f.write(f"\
dump            1 all cfg {dump} dump.{self.file_name}_*.cfg mass type xs ys zs fx fy fz\n\
")

            if scale:
                f.write(f"\
velocity        all scale {temp} \n")
            else:
                f.write(f"\
velocity        all create {float(temp)} {seed} \n")                

            f.write(f"\
fix             1 all nve/limit {np.amax(self.polylattice.interactions.cutoff_matrix)}\n\
fix             2 all langevin {float(temp)} {float(final_temp)} {damp} {seed}\n\
a")

            f.write(f"\
thermo_style    custom {data_string} \n\
thermo          {output_steps}\n\
run             {remaining_steps} \n\
unfix 1         \n\
unfix 2         \n\
")
            if self.dumping == 1:
                self.dumping = 0
                f.write(f"\
undump 1        \n\
")
            f.write(f"\n")
            
            # resignment of modified values
        for i in range(len(self.pending_mods)):
            self.polylattice.interactions.modify_sigma(self.pending_mods[i][0], 
                                                       self.pending_mods[i][1], 
                                                       self.pending_mods[i][-1][0])
            
            self.polylattice.interactions.modify_energy(self.pending_mods[i][0], 
                                                        self.pending_mods[i][1], 
                                                        self.pending_mods[i][-1][1])
            
            self.polylattice.interactions.modify_cutoff(self.pending_mods[i][0], 
                                                        self.pending_mods[i][1], 
                                                        self.pending_mods[i][-1][2])
                
            self.polylattice.interactions.modify_n(self.pending_mods[i][0], 
                                                   self.pending_mods[i][1], 
                                                   self.pending_mods[i][-1][3])
                
            self.polylattice.interactions.modify_alpha(self.pending_mods[i][0], 
                                                       self.pending_mods[i][1], 
                                                       self.pending_mods[i][-1][4])
            
            self.polylattice.interactions.modify_lambda(self.pending_mods[i][0], 
                                                        self.pending_mods[i][1], 
                                                        self.pending_mods[i][-1][5])
        self.pending_mods = []

    def reapply_interactions(self, soft = False, description=None):
        """
        Used to reapply an interactions instantaneously.
        Best performed immediately prior to a minimisation.
        """
        f = open(self.file_name, 'a')
        f.write("\n\
#---------------------------------------------------------------------------------------------------\n\
# REDEFINE POTENTIAL                                                                                \n")
                
        if description != None:
            f.write("# Description: {description}\n")

        f.write("\
#---------------------------------------------------------------------------------------------------\n\
")


        if soft == True:
            for i in range(np.shape(self.polylattice.interactions.type_matrix)[0]):
                for j in range(i, np.shape(self.polylattice.interactions.type_matrix)[0]):
                    typestring = self.polylattice.interactions.type_matrix[i,j]
                    type1, type2 = typestring.split(",")

                    num1 = self.polylattice.interactions.typekeys[type1]
                    num2 = self.polylattice.interactions.typekeys[type2]                
                    energy = self.polylattice.interactions.return_energy(type1, type2)
                    sigma = self.polylattice.interactions.return_sigma(type1, type2)
                    cutoff = self.polylattice.interactions.return_cutoff(type1, type2)
                    n = self.polylattice.interactions.return_n(type1, type2)
                    alpha = self.polylattice.interactions.return_alpha(type1, type2)
                    lmbda = self.polylattice.interactions.return_lambda(type1, type2)
                
                    f.write(f"\n\
pair_style    lj/cut/soft {n} {alpha} {cutoff}                                         \n\
pair_coeff    {num1} {num2} {round(energy,5)} {sigma} {lmbda}                         \n\
") 

        else:
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

        f.close()
            
        
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
        if self.polylattice.num_bonds > 0:        
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

    def settings(self, 
                 filename=None,
                 nlist=[10,1000],
                 nskin=1.0,
                 dielectric=False,
                 prebuilt=None):
        """ 
        NOTE: This comes SECOND in the simulation order of precedence!
              Will flag an error if sim_structure hasn't run first.
        This is where we define the header, the units and system settings.        
        Variables - nlist: the neighbour list build number
                               if the simulation provides dangerous builds, make this more frequent.
                    filename: name of the simulation input script file.
                    dielectric: whether the material is dielectric or not.
                                WARNING! needs a bit of confirmation.
        """
        if self.polylattice.structure_ready == False and prebuilt==None:
            raise EnvironmentError("You must create a structure file before settings can be defined.")
            
        if prebuilt!=None:
            self.data_file = prebuilt
            self.polylattice.structure_ready = True
        
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
neighbor      {nskin} multi                                                           \n\
neigh_modify  every {nlist[0]} one {nlist[1]}                                         \n\
")


        # add pair interactions of different beads
        for i in range(np.shape(self.polylattice.interactions.type_matrix)[0]):
            for j in range(i, np.shape(self.polylattice.interactions.type_matrix)[0]):
                
                if np.shape(self.polylattice.interactions.type_matrix)[0] == 1:
                    typestring = self.polylattice.interactions.type_matrix[0]
                    type1, type2 = typestring.split(",")
                    num1 = self.polylattice.interactions.typekeys[type1]

                    num2 = num1
                    energy = self.polylattice.interactions.return_energy(type1, type2)
                    sigma = self.polylattice.interactions.return_sigma(type1, type2)
                    cutoff = self.polylattice.interactions.return_cutoff(type1, type2)
                else:
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

        f.write(f"\
compute         1 all stress/atom NULL \n\
")
        # ensures that the settings have been set.
        # this must be true for equilibration to take place.
        self.set_settings = True

        f.close()

    def minimize(self, 
                 etol,
                 ftol,
                 mit=1000,
                 meval=10000,
                 style='cg'):
        """
        This should be used when the LJ parameters used to build the molecular model are different from those that 
        the simulation uses.

        Parameter:
        style   - options are cg, hftn, sd, quickmin, fire/old, spin, spin/cg, spin/lbfgs
        etol    - energy tolerance            
        ftol    - force tolerance
        mit     - max iterations of minimizer 
        meval   - max number of force/energy iterations
        """

        f = open(self.file_name, 'a')
        f.write(f"\
min_style    {style}                        \n\
minimize     {etol} {ftol} {mit} {meval}    \n\
")
        f.close()


    def equilibrate(self, steps, timestep, temp, dynamics,
                    bonding=False, final_temp=None,
                    damp=10.0, tdamp=None, pdamp=None, drag=2.0, scale=False,
                    output_steps=100, dump=0, data=('step','temp','lx', 'ly', 'lz', 'pxx', 'pyy', 'pzz', 'pe', 'ke'),
                    seed=random.randrange(0,99999),
                    rcf=False,
                    reset=False, description=None):
        """
        ARGUMENTS:
        steps      - the number of steps taken in the equilibration
        temp       - the temperature at which the equilibration will take place
        dynamics   - the type of equilibration that'll be taking place
                     'langevin' uses the nve and langevin thermostats
                     'nose_hoover' carries out npt dynamics
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
                self.global_dump = 1
                self.dumping = 1
                f.write(f"\
dump            1 all cfg {dump} dump.{self.file_name}_*.cfg mass type xs ys zs fx fy fz c_1[1] c_1[2] c_1[3]\n\
")

            if scale:
                f.write(f"\
velocity        all scale {temp} \n")
            else:
                f.write(f"\
velocity        all create {float(temp)} {seed} \n")

            f.write(f"\
fix             1 all nve/limit {np.amax(self.polylattice.interactions.cutoff_matrix)}\n\
fix             2 all langevin {float(temp)} {float(final_temp)} {damp} {seed}\n\
")
            if rcf == True:
                print("Not implemented!")
                
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
            if self.dumping == 1:
                self.dumping = 0
                f.write(f"\
undump 1        \n\
")

            if bonding == True:
                for i in range(1, self.polylattice.interactions.num_types):
                    f.write(f"\
unfix {i+2} \n\
")                    

            f.write(f"\
write_restart   restart.{self.file_name}.polylattice{self.equibs}\n\
")
        if dynamics=='npt':
            if dump>0:
                self.global_dump = 1
                self.dumping = 1
                f.write(f"\
dump            1 all cfg {dump} dump.{self.file_name}_*.cfg mass type xs ys zs fx fy fz\n\
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
")
            if self.dumping == 1:
                self.dumping = 0
                f.write(f"\
undump 1        \n\
")
            f.write(f"\
write_restart   restart.{self.file_name}.polylattice{self.equibs}\n\
")
            f.close()

    def deform(self, steps, timestep, deformation, temp, final_temp=None, damp=None, datafile=True, remap=True, output_steps=100, dump=0, reset=True, data=('step','temp','lx', 'ly', 'lz', 'pxx','pyy', 'pzz',), seed=random.randrange(0,99999), description = None):
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
        

        if dump>0:
            self.dumping = 1
            self.global_dump = 1
            f.write(f"\
dump            1 all cfg {dump} dump.{self.file_name}_*.cfg mass type xs ys zs fx fy fz c_1[1] c_1[2] c_1[3]\n\
")

        # add any missing elements to the deformation data

        true_data = [i for i in deformation]
        dimension = []

        for d in deformation:
            dimension.append(d[0])        

        dims = ['x', 'y', 'z']
        for d in dims:
            if d not in dimension:
                true_data.append([d, 'volume'])
                
        defdata = {}
        for i in true_data:
            defdata[i[0]] = i[1]

        deformation_string = ""
        for i in dims:
            deformation_string += f"{i} "
            
            if defdata[i] == 'volume':
                deformation_string += f"volume "
            else:
                deformation_string += f"final 0 {defdata[i]} "

        deformation_string += "units box "
        if remap == True:
            deformation_string += "remap x"

        f.write(f"\
velocity        all create {float(temp)} 1231 \n\
fix             1 all langevin {float(temp)} {float(final_temp)} {damp} {seed} \n\
fix             2 all nve/limit {np.amax(self.polylattice.interactions.cutoff_matrix)}\n\
fix		3 all deform 1 {deformation_string} \n\
")
        if datafile==True:
            self.data_production = 1
            title = "\"" + "\t".join(data) + "\""
            for i in data:
                f.write(f"\
variable        var_{i} equal \"{i}\"              \n\
")
            file_string = "\"${var_"+ "}\t${var_".join(data) + "}\""

            file_name = self.file_name+ "-" +str(self.deform_count) 
            f.write(f"\
fix             datafile all print {output_steps} {file_string} file {file_name}.deform.data title {title} screen no \n\
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
        if self.dumping == 1:
            self.dumping = 0
            f.write(f"\
undump 1        \n\
")
        if datafile == True:
            f.write(f"\
unfix datafile \n\
")
        # increment deformation count
        self.deform_count += 1

    def relax(self, eachstep, timestep, 
              stressdims, temp,
              loop=100,
              damp=None, 
              datafile=True, remap=True,
              output_steps=100, dump=0, reset=True,
              Nevery = 2 , Nrepeat = 10 , Nfreq = 100,
              data=('step','temp','lx', 'ly', 'lz', 'pxx','pyy', 'pzz',),
              seed=random.randrange(0,99999),
              description = None):
        """
        Carries out a relaxation of the box: essentially, deforms towards the original configuration (specified) until
        the stress acting on the box is equal to zero.
        This is encapsulated within an if-jump loop.

        Paramters:
        stressdims: a list of lists, with each sublist including "
        """
        if not self.set_settings:
            self.equibs-=1
            raise Exception("Settings have not yet been defined.") 

        if damp==None:
            damp=100*timestep

        f = open(self.file_name, 'a')
        f.write(f"\n\
#---------------------------------------------------------------------------------------------------------\n\
# RELAXATION STAGE                                                                                       \n")
        if description != None:
            self.lmp_sim_init += f"# Description: {description}\n"

        f.write("\
#---------------------------------------------------------------------------------------------------------\n\
")
        data_string = " ".join(data)    
        if datafile==True:
            self.data_production = 1
            title = "\"" + "\t".join(data) + "\""
            for i in data:
                f.write(f"\
variable        var_{i} equal \"{i}\"              \n\
")
            file_string = "\"${var_"+ "}\t${var_".join(data) + "}\""
            file_name = self.file_name+ "-" +str(self.deform_count) 
            f.write(f"\
fix             datafile all print {output_steps} {file_string} file {file_name}.deform.data title {title} screen no \n\
")        

        if dump>0:
            self.dumping = 1
            self.global_dump = 1
            f.write(f"\
dump            1 all cfg {dump} dump.{self.file_name}_*.cfg mass type xs ys zs fx fy fz c_1[1] c_1[2] c_1[3]\n\
")

        dims = ['x', 'y', 'z']

        deformation_string = ""
        justdims = [i[0] for i in stressdims]
        for d in dims:
            if d in justdims:
                strainrate = [i[1] for i in stressdims if i[0]==d][0]
                deformation_string += f"{d} trate {strainrate} "
            else:
                deformation_string += f"{d} volume "
        
        f.write(f"\
velocity        all create {float(temp)} 1231 \n\
fix             1 all langevin {float(temp)} {float(temp)} {damp} {seed} \n\
fix             2 all nve/limit {np.amax(self.polylattice.interactions.cutoff_matrix)}\n\
fix		3 all deform 1 {deformation_string} \n\
")  

        # ----------------------------------------------------------------------------------------------
        # The loop
        # ----------------------------------------------------------------------------------------------        
        ifcond = "\"${STRESS} > 0.0\" then \"jump SELF break\""
        fdata_str = data_string + " f_4"
        f.write(f"\n\
variable        pressure equal pxx                            \n\
fix             4 all ave/time {Nevery} {Nrepeat} {Nfreq} v_pressure           \n\
variable        STRESS equal f_4                              \n\
thermo_style    custom {fdata_str}                            \n\
thermo          {output_steps}                                \n\
label           loop                                          \n\
variable        a loop {loop}                                 \n\
run             {eachstep}                                    \n\
if              {ifcond}                                      \n\
next            a                                             \n\
jump            SELF loop                                     \n\
label           break                                         \n\
print           \"Relaxation concluded.\"                     \n\
unfix           4                                             \n\
\n")  

        # ----------------------------------------------------------------------------------------------
        if reset == True:
            f.write(f"\
reset_timestep  0\n\
")
        f.write(f"\
unfix 1                                                \n\
unfix 2                                                \n\
unfix 3                                                \n\
")
        if self.dumping == 1:
            self.dumping = 0
            f.write(f"\
undump 1        \n\
")
        if datafile == True:
            f.write(f"\
unfix datafile \n\
")
        # increment deformation count
        self.deform_count += 1



    def run(self, folder=None, lammps_path= None, mpi=0):
        """
        Carries out the LAMMPS run.
        """
        correct_dumping = 0
        if self.global_dump == 1 or self.data_production == 1:
            if folder==None:
                print("No directory for dumping: files will be dumped in working directory.")
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

            if lammps_path==None:
                os.system(f"mpiexec -np {mpi} lmp -in {self.file_name}")
            else:
                os.system(f"mpiexec -np {mpi} {lammps_path} -in {self.file_name}")

        if correct_dumping == 1:
            if self.data_production == 1:
                os.system(f"mv *.data* {folder}")   
            if self.global_dump == 1:
                os.system(f"mv *dump.* {folder}")   
 
            os.system(f"mv *restart.* {folder}")
            os.system(f"mv *log.* {folder}")
            os.system(f"cp {self.file_name} {folder}")
            os.system(f"cp {self.data_file} {folder}")

    def view(self, view_path, sim_file):
        bashcommand = f"{view_path} {sim_file}"

        os.system(bashcommand)
        
