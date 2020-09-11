# Program Name: data_collation_solid.py
# Author: Aravinthen Rajkumar
# Description: This script builds data files for stress and strain from data provided by lammps

import matplotlib.pyplot as plt
from operator import itemgetter
import numpy as np

#----------------------------------------------------------------------------------------
# REPLACE THE PATH WITH THE DATASET PRODUCED BY YOUR CODE.
#----------------------------------------------------------------------------------------
path = os.getcwd() + "/dataset_2020-08-25"

folders = os.listdir(path)
total_data = {}
for folder in folders:
    folder_path = path+"/"+folder
    experiments = os.listdir(folder_path)    
    crosslink_data = []
    for experiment in experiments:        
        datafile = folder_path + "/"+experiment+"/test_lattice.in.deform.data"
        numerics = []
        with open(datafile) as f:
            header = f.readline()
            data = [j.split('\t') for j in ([i.strip() for i in f][1:])]            
            for line in data:
                numerics.append([float(number) for number in line if number != ""])
        crosslink_data.append(numerics)
        
    total_data[folder] = np.array(crosslink_data)

data_points = [ ]
datasets = {}
for dataset in total_data:
    cl_val = int(''.join(filter(str.isdigit, str(dataset))))
    # cycles through each experiment in a crosslinking procedure

    stress_strains = []
    for experiment in total_data[dataset]:
        # these are the stress and strain values for an experiment
        stress = []
        strain = []
        for data in experiment:
            # appends the strain and stress for *each experiment*
            strain.append(data[5])
            stress.append(-400*data[2])

        stress_strains.append([strain, stress])
        
    datasets[cl_val] = stress_strains

# key: datasets[crosslink_number][0] = strain
# key: datasets[crosslink_number][1] = stress
