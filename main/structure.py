# Program Name: structure.py
# Author: Aravinthen Rajkumar
# Description: The classes in this file allow for the easy creation of polymer structures.

import numpy as np
from functools import reduce

def factors(n):    
    return set(reduce(list.__add__, 
                ([i, n//i] for i in range(1, int(n**0.5) + 1) if n % i == 0)))

class Structure:
    class Interactions:
        """All the interactions between beads are stored here."""
        def __init__(self, max_dist):
            self.types = {}
            self.typekeys = {}
            self.type_matrix = None
        
            self.sigma_matrix = None   # distances between types
            self.energy_matrix = None  # interaction energies between types
            self.cutoff_matrix = None  # cutoff between types
            
            # next parameters only to be used in the case of SOFT POTENTIALS            
            self.n_matrix = None
            self.alpha_matrix = None  
            self.lmbda_matrix = None  
            
        
            self.used_types = []        # the types that are actually used in the simulation
            self.max_dist = max_dist
            self.num_types = 0 # the number of types present in the box
        
        def newType(self, type_id, mass, potential, *args):
            """
            type_name: the id of the new type
            potential: TUPLE, three numbers.
            """
            if not type_id.isalnum():
                raise TypeError("Names must be composed of only alphanumeric characters.")
        
            if potential[0]*1.12234 > self.max_dist:
                error1 = "Error! The LJ-distance minimum is greater than the box grid spacing."
                error2 = "This means that random walk algorithm may not detect beads of this type."
                error3 = "Please respecify the number of cells within the box."
                raise SystemError(f"\n{error1}\n{error2}\n{error3}")
        
            if self.num_types == 0:
                if len(args) > 0:
                    raise TypeError("No other types have been defined in the system.")
                self.types[type_id] = mass
                self.typekeys[type_id] = self.num_types+1
                self.type_matrix = np.array([f"{type_id},{type_id}"])
            
                self.sigma_matrix = np.array([potential[0]])
                self.energy_matrix = np.array([potential[1]])
                self.cutoff_matrix = np.array([potential[2]])
            
                if len(potential) == 4:
                    self.n_matrix = np.array([potential[3][0]])
                    self.alpha_matrix = np.array([potential[3][1]])
                    self.lmbda_matrix = np.array([potential[3][2]])
                else:
                    self.n_matrix = np.array([0])
                    self.alpha_matrix = np.array([0])
                    self.lmbda_matrix = np.array([1])
                
                self.num_types+=1
                
            else:
                self.types[type_id] = mass
                self.typekeys[type_id] = self.num_types+1
                self.num_types+=1
            
                # modify the type matrix
                type_matrix = np.empty((self.num_types,self.num_types), dtype=np.dtype('U50'))
                
                # list of types
                types_list = [i for i in self.types]
                # type connections are built in                
                for i in range(self.num_types):
                    for j in range(self.num_types):
                        type_matrix[i,j] = f'{types_list[i]},{types_list[j]}'

                self.type_matrix = type_matrix
                            
                # modify the sigma matrix
                sigma_matrix = self.sigma_matrix
                #  build new columns and rows onto the sigma matrix
                sigma_matrix = np.column_stack((sigma_matrix,
                                                np.array([0 for i in range(self.num_types-1)])))
                sigma_matrix = np.vstack((sigma_matrix,
                                          np.array([0 for i in range(self.num_types)])))
                # assign sigma
                sigma_matrix[self.num_types-1,self.num_types-1] = potential[0]
            
                self.sigma_matrix = sigma_matrix
            
                # modify the energy matrix
                energy_matrix = self.energy_matrix                
                ##  build new columns and rows onto the energy matrix
                energy_matrix = np.column_stack((energy_matrix,
                                                 np.array([0 for i in range(self.num_types-1)])))
                energy_matrix = np.vstack((energy_matrix,
                                           np.array([0 for i in range(self.num_types)])))                
                # assign energy
                energy_matrix[self.num_types-1,self.num_types-1] = potential[1]
            
                self.energy_matrix = energy_matrix
            
                # modify the cutoff matrix
                cutoff_matrix = self.cutoff_matrix
                ##  build new columns and rows onto the cutoff matrix
                cutoff_matrix = np.column_stack((cutoff_matrix,
                                                 np.array([0 for i in range(self.num_types-1)])))
                cutoff_matrix = np.vstack((cutoff_matrix,
                                           np.array([0 for i in range(self.num_types)])))
                # assign cutoff 
                cutoff_matrix[self.num_types-1,self.num_types-1] = potential[2]                
                self.cutoff_matrix = cutoff_matrix
            
                # -----------------------------------------------------------------------
            
                if len(potential) > 3:
                    soft_potential = potential[3]
                else:
                    soft_potential = (0, 0, 1) # n, alpha, lmbda
                    
                # modify the n matrix
                n_matrix = self.n_matrix
                # build new columns and rows onto the n matrix
                n_matrix = np.column_stack((n_matrix,
                                             np.array([0 for i in range(self.num_types-1)])))
                n_matrix = np.vstack((n_matrix,
                                      np.array([0 for i in range(self.num_types)])))
                
                # assign n 
                n_matrix[self.num_types-1,self.num_types-1] = soft_potential[0]                
                self.n_matrix = n_matrix
                
                # modify the alpha matrix
                alpha_matrix = self.alpha_matrix
                # build new columns and rows onto the n matrix
                alpha_matrix = np.column_stack((alpha_matrix,
                                                np.array([0 for i in range(self.num_types-1)])))
                alpha_matrix = np.vstack((alpha_matrix,
                                          np.array([0 for i in range(self.num_types)])))
                
                # assign n 
                alpha_matrix[self.num_types-1,self.num_types-1] = soft_potential[1]                
                self.alpha_matrix = alpha_matrix
                
                # modify the lmbda matrix
                lmbda_matrix = self.lmbda_matrix
                # build new columns and rows onto the n matrix
                lmbda_matrix = np.column_stack((lmbda_matrix,
                                                np.array([0 for i in range(self.num_types-1)])))
                lmbda_matrix = np.vstack((lmbda_matrix,
                                          np.array([0 for i in range(self.num_types)])))
                
                # assign n 
                lmbda_matrix[self.num_types-1,self.num_types-1] = soft_potential[2]                
                self.lmbda_matrix = lmbda_matrix
                
                
                #----------------------------------------------------------------------------------
                # now deal with the interactions for different kinds of beads
                for i in args:
                    name = i[0].replace(" ","")
                    
                    # reverse name has to be broke part, reversed, then joined.
                    revname = i[0].replace(" ","")
                    revname = revname.split(",")
                    revname = ",".join(revname[::-1])
                    properties = i[1]
                    
                    if name not in self.type_matrix or revname not in self.type_matrix:
                        raise TypeError("Type combination does not exist in system.")
                    
                    if type_id not in name:
                        raise TypeError("Unrelated type interaction defined. Interactions must always contain the new type..")
                    
                    ni, nj = np.where(self.type_matrix == name)
                    ri, rj = np.where(self.type_matrix == revname)
                    
                    # now with the array indices, simply slot everything into place.
                    self.sigma_matrix[ri[0], rj[0]] = properties[0]
                    self.sigma_matrix[ni[0], nj[0]] = properties[0]
                    
                    self.energy_matrix[ri[0], rj[0]] = properties[1]
                    self.energy_matrix[ni[0], nj[0]] = properties[1]
                    
                    self.cutoff_matrix[ri[0], rj[0]] = properties[2]
                    self.cutoff_matrix[ni[0], nj[0]] = properties[2]
                    
                    
                    if len(properties) > 3:
                        soft_properties = properties[3]
                    else:
                        soft_properties = (0, 0, 1) # n, alpha, lmbda
                        
                    self.n_matrix[ri[0], rj[0]] = soft_properties[0]
                    self.n_matrix[ni[0], nj[0]] = soft_properties[0]
                
                    self.alpha_matrix[ri[0], rj[0]] = soft_properties[1]
                    self.alpha_matrix[ni[0], nj[0]] = soft_properties[1]
                
                    self.lmbda_matrix[ri[0], rj[0]] = soft_properties[2]
                    self.lmbda_matrix[ni[0], nj[0]] = soft_properties[2]
                
        def return_sigma(self, type1, type2):
            # returns the interaction between two beads
            namestring = f"{type1},{type2}"
            ni, nj = np.where(self.type_matrix == namestring)
            return self.sigma_matrix[ni,nj][0]

        def return_energy(self, type1, type2):
            # returns the interaction between two beads
            namestring = f"{type1},{type2}"
            ni, nj = np.where(self.type_matrix == namestring)
            return self.energy_matrix[ni,nj][0]
        
        def return_cutoff(self, type1, type2):
            # returns the interaction between two beads
            namestring = f"{type1},{type2}"
            ni, nj = np.where(self.type_matrix == namestring)
            return self.cutoff_matrix[ni,nj][0]
        
        def return_n(self, type1, type2):
            # returns the interaction between two beads
            namestring = f"{type1},{type2}"
            ni, nj = np.where(self.type_matrix == namestring)
            return self.n_matrix[ni,nj][0]
        
        def return_alpha(self, type1, type2):
            # returns the interaction between two beads
            namestring = f"{type1},{type2}"
            ni, nj = np.where(self.type_matrix == namestring)
            return self.alpha_matrix[ni,nj][0]
        
        def return_lambda(self, type1, type2):
            # returns the interaction between two beads
            namestring = f"{type1},{type2}"
            ni, nj = np.where(self.type_matrix == namestring)
            return self.lmbda_matrix[ni,nj][0]
        
        # ----------------------------------------------------------------------------------
        
        def modify_sigma(self, type1, type2, new_val):
            # modifys the interaction between two beads
            namestring = f"{type1},{type2}"
            ni, nj = np.where(self.type_matrix == namestring)
            self.sigma_matrix[ni,nj] = new_val
            self.sigma_matrix[nj,ni] = new_val
            
        def modify_energy(self, type1, type2, new_val):
            # modifys the interaction between two beads
            namestring = f"{type1},{type2}"
            ni, nj = np.where(self.type_matrix == namestring)
            self.energy_matrix[ni,nj] = new_val
            self.energy_matrix[nj,ni] = new_val
            
        def modify_cutoff(self, type1, type2, new_val):
            # modifys the interaction between two beads
            namestring = f"{type1},{type2}"
            ni, nj = np.where(self.type_matrix == namestring)
            self.cutoff_matrix[ni,nj] = new_val
            self.cutoff_matrix[nj,ni] = new_val

        def modify_n(self, type1, type2, new_val):
            # modifys the interaction between two beads
            namestring = f"{type1},{type2}"
            ni, nj = np.where(self.type_matrix == namestring)
            self.n_matrix[ni,nj] = new_val
            self.n_matrix[nj,ni] = new_val

        def modify_alpha(self, type1, type2, new_val):
            # modifys the interaction between two beads
            namestring = f"{type1},{type2}"
            ni, nj = np.where(self.type_matrix == namestring)
            self.alpha_matrix[ni,nj] = new_val
            self.alpha_matrix[nj,ni] = new_val
            
        def modify_lambda(self, type1, type2, new_val):
            # modifys the interaction between two beads
            namestring = f"{type1},{type2}"
            ni, nj = np.where(self.type_matrix == namestring)
            self.lmbda_matrix[ni,nj] = new_val
            self.lmbda_matrix[nj,ni] = new_val


            

