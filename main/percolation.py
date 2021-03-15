# Program Name: percolation.py
# Author: Aravinthen Rajkumar
# Description: Here are some useful bits of code for the percolation part

# starting points: used to gather starting points for the burn algorithm.
starting_points = []        
current_index = 0
while len(starting_points) == 0:
    if face == "xy" or face == "yx":
        for i in range(0, self.polylattice.cellnums):
            for j in range(0, self.polylattice.cellnums):
                cell = self.polylattice.index([current_index, i, j]).beads
                if len(cell) > 0:
                    cell.sort(key=lambda x:x[-1][0])
                    starting_points.append(cell[0])
                        
        if face == "xz" or face == "zx":
            for i in range(0, self.polylattice.cellnums):
                for j in range(0, self.polylattice.cellnums):
                    cell = self.polylattice.index([i, current_index, j]).beads
                    if len(cell) > 0:
                        cell.sort(key=lambda x:x[-1][1])
                        starting_points.append(cell[0])
                    
        if face == "zy" or face == "yz":
            for i in range(0, self.polylattice.cellnums):
                for j in range(0, self.polylattice.cellnums):
                    cell = self.polylattice.index([i, j, current_index]).beads
                    if len(cell) > 0:
                        cell.sort(key=lambda x:x[-1][2])
                        starting_points.append(cell[0])

        current_index += 1


