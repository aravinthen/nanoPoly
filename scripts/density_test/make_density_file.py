# Program Name: make_density_file.py
# Author: Aravinthen Rajkumar
# Description: creates a lamellar density file

import math as m

f = open("density_file.txt", 'w')
ncells = 80
cell_length = 1.25

fill_count = 0
for i in range(ncells):
    for j in range(ncells):
        for k in range(ncells):
            xval = i*1.25
            dens = 0.4*m.cos(0.4*cell_length*i)+0.5
            f.write(f"{i}\t{j}\t{k}\t{dens}\t{1-dens} \n")
            fill_count+=1

print(fill_count)
