# Program Name: scf.py
# Author: Aravinthen Rajkumar
# Description: This is the library in which self-consistent field theory routines are included.

class MeanField:
    """
    Used to read in density files into polyboxes
    """
    def __init__(self, polylattice):
        self.polylattice = polylattice

        self.density = False

        
    def density_file(self, density_file):
        if self.polylattice.interactions.num_types == 0:
            raise EnvironmentError("Types must be defined before setting densities.")

        count=0
        with open(density_file, 'r') as f:
            # check if the number of beads in file match with the number of beads in interaction data
            file_beads = len(f.readline().strip().split("\t"))

            if file_beads != self.polylattice.interactions.num_types:                
                raise EnvironmentError("Defined bead types and file bead types do not match.")
            
        with open(density_file, 'r') as f:
            x = 0
            y = 0
            z = 0
            for line in f:
                cell = [x,y,z]
                datum = line.strip().split("\t")
                density_data = [float(i) for i in datum]

                if any(i < 0.0 for i in density_data):
                    raise EnvironmentError("Unphysical densities contained with density file.")
                
                self.polylattice.index(cell).densities = density_data

                x+=1                
                if x > self.polylattice.cellnums-1:
                    x = 0
                    y +=1
                if y > self.polylattice.cellnums-1:
                    x = 0
                    y = 0
                    z +=1
                    
                count+=1
                
        print(f"{count} density values read into box.")
        
        self.density = True
