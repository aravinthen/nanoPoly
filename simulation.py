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
import multiprocessing
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
        
    def structure(self, datafile=None):
        """
        This is where we dump all of the structural data from polylattice into the LAMMPS file.
        NOTE: this comes FIRST in the simulation's order of precedence!
        """
        self.data_file = str(datafile) # this sets the data in the class so it can be reused.
        f = open(f"{self.data_file}", "w")


        t0 = time.time()
        all_data = self.polylattice.walk_data() # where all the data is stored
        t1 = time.time()
        print(f"Data read in. Time taken: {t1 - t0}")
        
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
{len(self.polylattice.types)} atom types                                            \n\
{len(self.polylattice.bonds)} bond types                                            \n\
                                                                                    \n\
0.0000 {self.polylattice.cellside*self.polylattice.cellnums:.4f} xlo xhi            \n\
0.0000 {self.polylattice.cellside*self.polylattice.cellnums:.4f} ylo yhi            \n\
0.0000 {self.polylattice.cellside*self.polylattice.cellnums:.4f} zlo zhi            \n\
                                                                                    \n\
Masses                                                                              \n\
                                                                                    \n\
")
        # read in the mass data.
        for i in self.polylattice.types:
            f.write(f"{i}\t{self.polylattice.types[i]}\n")
        f.write("\n\n")

        # this is highly dependent on the bead type.

        # now it's time to read in data.
        f.write("\
Atoms                                                                               \n\
\n")
        cross_posn = []
        for i in self.polylattice.crosslinks_loc:
            bond1 = i[0][-1]
            bond2 = i[-1][-1]
            cross_posn.append(list(np.around(bond1, decimals=4)))
            cross_posn.append(list(np.around(bond2, decimals=4)))
        
        cross_vals = []
        linknums = []

        t0 = time.time()
        for i in range(len(all_data)):
            atom_num = i+1
            chain = all_data[i][0]
            beadtype = all_data[i][2]            

            #------------------ cross links -----------------------
            cp = list(np.around(all_data[i][-1],decimals=4))
            if cp in cross_posn:
                cross_vals.append([list(np.around(all_data[i][-1], decimals=4)), atom_num])

            # note that num_walks is nothing more than the number of chains
            # as initial value is 0, the actual value of the chain 
            if chain == self.polylattice.num_walks:
                linknums.append([list(np.around(all_data[i][-1], decimals=4)), atom_num])
            #------------------------------------------------------
                

            x, y, z = all_data[i][-1][0], all_data[i][-1][1], all_data[i][-1][2]
            f.write(f"\t{atom_num}\t{chain}\t{beadtype}\t{x:.5f}\t\t{y:.5f}\t\t{z:.5f}\t\n")
            
        t1 = time.time()
        print(f"Positions read in. Time taken: {t1 - t0}")
        
        # reading in the bond data        
        f.write("\
Bonds                                                                               \n\
                                                                                    \n")
        
        t0 = time.time()
        bond = 0
        atom_num = 1
        for walk in self.polylattice.walkinfo:
            for bead in range(1,walk[1]):
                # writing the bond
                atom_num += 1
                bond += 1
                bondtype = all_data[atom_num-1][4]
                f.write(f"\t{bond}\t{bondtype}\t{(atom_num-1)}\t{atom_num}\n")
            atom_num+=1
        t1 = time.time()
        print(f"Bonds read in. Total time taken: {t1 - t0}")        
                
        # crosslinker bonds next. This is slightly more complicated.
#        print(cross_vals)        
        for i in self.polylattice.crosslinks_loc:
            # conversion to lists is due to the difficulties in truth values for arrays
            # it is very hard to evaluate two arrays as equal due to floating point error
            bond1 = list(np.around(i[0][-1], decimals=4))
            clinker = list(np.around(i[1][-1], decimals=4))
            bond2 = list(np.around(i[2][-1], decimals=4))
            bondtype = i[1][4]


            bond1_num = [dat[1] for dat in cross_vals if dat[0]==bond1][0]
            clinker_num = [dat[1] for dat in linknums if dat[0]==clinker][0]
            bond2_num = [dat[1] for dat in cross_vals if dat[0]==bond2][0]
            
            f.write(f"\t{bond}\t{bondtype}\t{bond1_num}\t{clinker_num}\n")
            bond+=1
            f.write(f"\t{bond}\t{bondtype}\t{clinker_num}\t{bond2_num}\n")
            bond+=1
            
        f.close()

        
    def settings(self, filename=None, dielectric=False, comms=None):
        """ 
        NOTE: This comes SECOND in the simulation order of precedence!
              Will flag an error if sim_structure hasn't run first.
        This is where we define the header, the units and system settings.        
        Variables - fname: name of the simulation input script file.
                    dielectric: whether the material is dielectric or not.
                                WARNING! needs a bit of confirmation.
        """

        if comms == None:
            comms = 2*self.polylattice.lj_param

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
        self.file_name = str(fname) # this sets the file name in the class so it can be reused.
        self.lmp_sim_init += f"\
#-----------------------------------------------------------------------------------\n\
# NANOPOLY - POLYMER NANOCOMPOSITE SIMULATION FILE                                  \n\
# Full project files can be found at: https://github.com/aravinthen/nanoPoly        \n\
#-----------------------------------------------------------------------------------\n\
# FILENAME: {self.file_name}                                                        \n\
# DATE: {today}                                                                     \n\
# OVERVIEW:                                                                         \n\
#    Number of polymer chains:     {self.polylattice.num_walks}                     \n\
#    Number of crosslinks:         {len(self.polylattice.crosslinks_loc)}           \n\
#    Nanostructure:                {self.polylattice.nanostructure}                 \n\
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
read_data          ${{structure}}                                                     \n\
                                                                                      \n\
# define interactions                                                                 \n\
neighbor      0.4 bin                                                                 \n\
neigh_modify  every 10 one 10000                                                      \n\
comm_modify   mode single cutoff {comms} vel yes                                      \n\
pair_style    lj/cut {self.polylattice.lj_cut}                                        \n\
pair_coeff    * * {self.polylattice.lj_energy} {self.polylattice.lj_param}            \n\
"
        # now it's time to read in the bond information.
        styles_dict = dictionary_shuffle(self.polylattice.bonds, -1)        
        for style in styles_dict:
            self.lmp_sim_init += f"\
bond_style    {style}  \n"
            if style == 'fene':                
                self.lmp_sim_init += f"\
special_bonds fene  \n"
            for data in styles_dict[style]:
                self.lmp_sim_init += f"\
bond_coeff    {data[0]+1} {np.around(data[1][0], decimals=4)} {data[1][1]} {data[1][2]} {data[1][3]} \n"
                

        # ensures that the settings have been set.
        # this must be true for equilibration to take place.
        self.set_settings = True

    def equilibrate(self, steps, timestep, temp, dynamics, bonding=False, final_temp=None, damp=10.0, tdamp=None, pdamp=None, drag=2.0, output_steps=100, dump=0, data=('step','temp','press'), seed=random.randrange(0,99999), reset=True, description=None):
        """
        ARGUMENTS:
        steps      - the number of steps taken in the equilibration
        temp       - the temperature at which the equilibration will take place
        dynamics   - the type of equilibration that'll be taking place
                     'langevin' uses the nve and langevin thermostats
                     'npt' carries out npt dynamics
        final_temp - used for heating and cooling equilibrations
        velocity   - dependent on the 
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
        
        self.lmp_sim_init += f"\n\
#-----------------------------------------------------------------------------------\n\
# EQUILIBRATION STAGE {self.equibs}                                                 \n"
        if description != None:
            self.lmp_sim_init += f"# Description: {description}\n"

        self.lmp_sim_init += "\
#-----------------------------------------------------------------------------------\n\
"
        if dynamics=='langevin':            
            if dump>0:
                self.dumping = 1
                self.lmp_sim_init += f"\
dump            1 all cfg {dump} dump.{self.file_name}_*.cfg mass type xs ys zs fx fy fz\n\
\n\
"            
            self.lmp_sim_init += f"\
velocity        all create {float(temp)} 1231 \n\
fix             1 all nve/limit {self.polylattice.lj_param}\n\
fix             2 all langevin {float(temp)} {float(final_temp)} {damp} {seed}\n\
"
            if bonding == True:
                if self.polylattice.cl_unbonded == False:
                    raise EnvironmentError("Unbonded crosslinks are not present in the Polylattice!\n")                
                else:
                    b_data = self.polylattice.cl_bonding
                    # jparam : the allowed beads
                    for jparam in b_data[2]:
                        if b_data[6] == None:
                            self.lmp_sim_init += f"\
fix             {jparam+2} all bond/create 1 {b_data[0]}  {jparam} {b_data[3]} {b_data[1]} iparam {b_data[4]} jparam {b_data[5]} \n\
"
                        else:
                            self.lmp_sim_init += f"\
fix             {jparam+2} all bond/create 1 {b_data[0]}  {jparam} {b_data[3]} {b_data[1]} prob {b_data[6]} {seed} iparam {b_data[4]} jparam {b_data[5]} \n\
"                            
            
            self.lmp_sim_init += f"\
thermo_style    custom {data_string} \n\
thermo          {output_steps}\n\
"
            if reset == True:
                self.lmp_sim_init += f"\
reset_timestep  0\n\
"
            self.lmp_sim_init += f"\
run             {steps} \n\
unfix 1         \n\
unfix 2         \n\
write_restart   restart.{self.file_name}.polylattice{self.equibs}\n\
"
        if dynamics=='nose_hoover':
            if dump>0:
                self.dumping = 1
                self.lmp_sim_init += f"\
dump            1 all cfg {dump} dump.{self.file_name}_*.cfg mass type xs ys zs fx fy fz\n\
\n\
"  
            self.lmp_sim_init += f"\
fix             1 all npt temp {float(temp)} {float(final_temp)} {tdamp} iso 0 0 {pdamp} drag {drag} \n\
fix             2 all momentum 1 linear 1 1 1\n\
thermo_style    custom {data_string} \n\
thermo          {output_steps}\n\
"
            if reset == True:
                self.lmp_sim_init += f"\
reset_timestep  0\n\
"
            self.lmp_sim_init += f"\
run             {steps} \n\
unfix 1         \n\
unfix 2         \n\
write_restart   restart.{self.file_name}.polylattice{self.equibs}\n\
"            

    def deform(self, steps, timestep, strain, temp, tdamp=None, pdamp=1000, drag=2.0, datafile=True, output_steps=100, reset=True, data=('step','temp','pxx', 'pyy', 'pzz', 'lx', 'ly', 'lz'), description = None):
        """
        Carries out the deformation of the tyres. 
        """
        self.lmp_sim_init += f"\n\
#-----------------------------------------------------------------------------------\n\
# DEFORMATION STAGE                                                                 \n"
        if description != None:
            self.lmp_sim_init += f"# Description: {description}\n"

        self.lmp_sim_init += "\
#-----------------------------------------------------------------------------------\n\
"
        data_string = " ".join(data)
        
        if tdamp==None:
            tdamp=100*timestep

        self.lmp_sim_init += f"\
run             0            \n\
fix		1 all npt temp {temp} {temp} {tdamp} y 0 0 {pdamp} z 0 0 {pdamp} drag {drag}  \n\
fix		2 all deform 1 x erate {strain} units box remap x                             \n\
"
        if datafile==True:
            self.data_production = 1
            title = "\"" + "\t".join(data) + "\""
            for i in data:
                self.lmp_sim_init += f"\
variable        var_{i} equal \"{i}\"              \n\
"
            file_string = "\"${var_"+ "}\t${var_".join(data) + "}\""

            self.lmp_sim_init += f"\
fix             datafile all print {output_steps} {file_string} file {self.file_name}.deform.data title {title} screen no \n\
"
            
        self.lmp_sim_init += f"\
thermo_style    custom {data_string}                   \n\
thermo          {output_steps}                         \n\
"
        if reset == True:
            self.lmp_sim_init += f"\
reset_timestep  0\n\
"
        self.lmp_sim_init += f"\
run             {steps}                                \n\
unfix 1                                                \n\
unfix 2                                                \n\
"   
        if datafile == True:
            self.lmp_sim_init += f"\
unfix datafile \n\
"
        
    def files(self, ):
        with open(f"{self.file_name}", "w") as f:
            f.write(self.lmp_sim_init)
            
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

    def view(self, sim_file):
        bashcommand = f"ovito {sim_file}"
        os.system(bashcommand)
        
