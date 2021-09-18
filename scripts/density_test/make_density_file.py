# Program Name: make_density_file.py
# Author: Aravinthen Rajkumar
# Description: creates a lamellar density file

import math as m

f = open("density_file.txt", 'w')
ncells = 50
cell_length = 1.25

fill_count = 0
for z in range(ncells):
    for y in range(ncells):
        for x in range(ncells):
            xval = z*1.25
            dens = 0.4*m.cos(0.4*cell_length*z)+0.5
            f.write(f"{dens}\t{1-dens}\n")
            fill_count+=1

print(fill_count)
