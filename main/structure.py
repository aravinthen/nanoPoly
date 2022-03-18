# Program Name: structure.py
# Author: Aravinthen Rajkumar
# Description: The classes in this file allow for the easy creation of polymer structures.

import numpy as np
import time
import random
import math as m
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
            # modifies the interaction between two beads
            namestring = f"{type1},{type2}"
            ni, nj = np.where(self.type_matrix == namestring)
            self.sigma_matrix[ni,nj] = new_val
            self.sigma_matrix[nj,ni] = new_val
            
        def modify_energy(self, type1, type2, new_val):
            # modifies the interaction between two beads
            namestring = f"{type1},{type2}"
            ni, nj = np.where(self.type_matrix == namestring)
            self.energy_matrix[ni,nj] = new_val
            self.energy_matrix[nj,ni] = new_val
            
        def modify_cutoff(self, type1, type2, new_val):
            # modifies the interaction between two beads
            namestring = f"{type1},{type2}"
            ni, nj = np.where(self.type_matrix == namestring)
            self.cutoff_matrix[ni,nj] = new_val
            self.cutoff_matrix[nj,ni] = new_val

        def modify_n(self, type1, type2, new_val):
            # modifies the interaction between two beads
            namestring = f"{type1},{type2}"
            ni, nj = np.where(self.type_matrix == namestring)
            self.n_matrix[ni,nj] = new_val
            self.n_matrix[nj,ni] = new_val

        def modify_alpha(self, type1, type2, new_val):
            # modifies the interaction between two beads
            namestring = f"{type1},{type2}"
            ni, nj = np.where(self.type_matrix == namestring)
            self.alpha_matrix[ni,nj] = new_val
            self.alpha_matrix[nj,ni] = new_val
            
        def modify_lambda(self, type1, type2, new_val):
            # modifies the interaction between two beads
            namestring = f"{type1},{type2}"
            ni, nj = np.where(self.type_matrix == namestring)
            self.lmbda_matrix[ni,nj] = new_val
            self.lmbda_matrix[nj,ni] = new_val

    #---------------------------------------------------------------------------------------
        def pw_energy(self, particles, index1, index2, xdim, ydim, zdim):
            """
            Calculates a pairwise energy between two particles in a system
            """
            pos1 = [particles[index1][0], particles[index1][1], particles[index1][2]]
            pos2 = [particles[index2][0], particles[index2][1], particles[index2][2]]

            sigma = self.polylattice.interactions.return_sigma(particles[index1][3], particles[index2][3])
            energy = self.polylattice.interactions.return_energy(particles[index1][3], particles[index2][3])
                    
            displacement = [pos2[0]-pos1[0],
                            pos2[1]-pos1[1],
                            pos2[2]-pos1[2]]
            
            # correct for periodicity
            displacement = periodize(displacement, xdim, ydim, zdim)
            dist2 = sum([i**2 for i in displacement])
            
            # calculate (1/r)^12 - (1/r)^6
            return energy*((sigma/dist2)**6 - (sigma/dist2)**3)

        def lj_energy(self, particles, xdim, ydim, zdim):
            energy = 0.0

            for i in range(len(particles)-1):
                for j in range(i+1, len(particles)):                    
                    energy+= self.pw_energy(particles, i, j, xdim, ydim, zdim)

            energy = 4*energy
            return energy

        def ec_disp(self, particles, part, new_pos, xdim, ydim, zdim):
            """
            Energy change that results from displacement.
            """
            energy_change = 0.0
            for p in particles:
                if p == part:
                    continue
                else:
                    # calcuate the displacement from the particle being iterated over and the new/old position 
                    # of the particle being changed.
                    # then, periodize that new displacement and get the distance for the new and old particle.

                    disp = periodize([new_pos[0] - p[0], new_pos[1] - p[1], new_pos[2] - p[2]],
                                             xdim, ydim, zdim)


                    dist = sum([i**2 for i in disp])

                    # obtain sigma and energy for the particle interaction
                    # these are dependent on type, so you don't have to calculate it for the new energy.
                    sigma = self.polylattice.interactions.return_sigma(part[-1], p[-1])
                    energy = self.polylattice.interactions.return_energy(part[-1], p[-1])
                    
                    # calculate the energies
                    new_energy = energy*((sigma/dist)**6 - (sigma/dist)**3)

                    energy_change += new_energy 
            
            return 4*energy_change

        def ec_swap(self, particles, part1, part2, xdim, ydim, zdim):
            """
            Energy change that results from swapping particle types.
            """            
            
            # first check whether the particles types are actually different.
            if part1[-1] == part2[-1]:
                raise EnvironmentError("Particles are of the same type. These cannot be swapped.")
        
            echange1 = 0.0
            # ---------------------------------------------------------------------------------------------
            # the energy contribution for the first particle
            # ---------------------------------------------------------------------------------------------
            for p in particles:
                if p == part1:
                    continue

                # the distances are unchanged.
                disp = periodize([part1[0] - p[0], part1[1] - p[1], part1[2] - p[2]],
                                 xdim, ydim, zdim)
                dist = sum([i**2 for i in disp])                
                
                # obtain sigma and energy for the new energy
                # all that's going on here is that the type of part1 is switched out for the type of part2,
                # with p taking on the type of part1 if p == part2
                if p == part2:
                    sigma = self.polylattice.interactions.return_sigma(part2[-1], part1[-1])
                    energy = self.polylattice.interactions.return_energy(part2[-1], part1[-1])
                else:
                    sigma = self.polylattice.interactions.return_sigma(part2[-1], p[-1])
                    energy = self.polylattice.interactions.return_energy(part2[-1], p[-1])

                # calculate the new energy
                echange1 += energy*((sigma/dist)**6 - (sigma/dist)**3)
                
            # ---------------------------------------------------------------------------------------------
            # the energy contribution for the second particle
            # ---------------------------------------------------------------------------------------------
            echange2 = 0.0
            for p in particles:
                if p == part2:
                    continue

                # the distances are unchanged.
                disp = periodize([part2[0] - p[0], part2[1] - p[1], part2[2] - p[2]],
                                 xdim, ydim, zdim)
                dist = sum([i**2 for i in disp])                
                
                # obtain sigma and energy for the new energy
                # all that's going on here is that the type of part2 is switched out for part1, with p taking 
                # on the type of part2 if p == part1
                if p == part1:
                    sigma = self.polylattice.interactions.return_sigma(part1[-1], part2[-1])
                    energy = self.polylattice.interactions.return_energy(part1[-1], part2[-1])
                else:
                    sigma = self.polylattice.interactions.return_sigma(part1[-1], p[-1])
                    energy = self.polylattice.interactions.return_energy(part1[-1], p[-1])

                # calculate the new energy
                echange2 += energy*((sigma/dist)**6 - (sigma/dist)**3)

            return 4*(echange1 + echange2)


        def scale_particles(self, particles, scaled_length):
            """
            Scales the particles for use in the ec_move.
            The volume change should be inputted automatically as part of a random number calculation.
            """
            # We need to save the box size and the particle coordinates

            new_particles = []
            for p in particles:
                distance = m.sqrt(p[0]*p[0] + p[1]*p[1] + p[2]*p[2])                
                if distance > 0.01: # ignore the particles already near the center of the box
                    # break down vectors to their unit contributions
                    # this is just x_i/|x|
                    dx = p[0]/distance
                    dy = p[1]/distance
                    dz = p[2]/distance

                    # multiply the units to the full scale
                    new_particles.append([scaled_length*dx, scaled_length*dy, scaled_length*dz, p[-1]])

            return new_particles

        def enc_vol(self, scaled_particles, current_energy,
                    xdim, ydim, zdim, vol_change, pressure):
            """
            ENTHALPY change that results from changes in volume.
            Scaled particles should be obtained from the scale_particles function.
            """            

            old_vol = xdim*ydim*zdim
            new_vol = old_vol + vol_change
            scale = (new_vol/old_vol)**(1/3) 
            # calculate the energy of the scaled particles             
            new_energy = self.lj_energy(scaled_particles, scale*xdim, scale*ydim, scale*zdim)
            
            energy_change = new_energy - current_energy
            num_atoms = len(scaled_particles)
            
            # change in enthalphy
            en_change = (energy_change + pressure*vol_change) + num_atoms*(m.log(new_vol) - m.log(old_vol))
            
            return en_change, energy_change
            
                                        
        # --------------------------------------------------------------------------------------------------------
        # Composite functions
        # --------------------------------------------------------------------------------------------------------

        def flory_huggins(self, temp, type1, type2, num_particles = 8192,
                          maxdx=0.1, maxdv=0.01, beta=1.0, pressure=0.0,
                          ecycles=40000, scycles=10000, sfreq=10, vis=False,
                          pfreq=1000):
            """
            This function uses the Monte Carlo method to calculate the number fraction of a type A bead within a 
            sample.
            The number fraction is then employed in the formula derived in Chremos (2014) to calculate the Flory
            Huggins parameter.

            PARAMETERS:
            temp:          temperature
            type1/type2:   the types that will be used in the Monte Carlo simulation.
                           these have to be defined in the interactions object beforehand.
            num_particles: this follows the process detailed Chremos's paper and is automatically set to 8192.
                           however, it can be changed.
            
            """
            print("\nThe routine used in flory_huggins was developed with reference to the")
            print("work by Chremos, Nikoubashman and Panagiotopolous (2014). Please cite")
            print("these authors if you make use of this function for published work!\n")
            # -----------------------------------------------------------------------------
            # Helper functions

            # -----------------------------------------------------------------------------

            # these will change likely during the simulation
            
            total_time0 = time.time()

            ydim=5.0
            zdim=5.0
            xdim=10.0

            particles = []
            # place the type1 beads
            for i in range(num_particles//2):
                xpos = random.uniform(-xdim//2, 0)
                ypos = random.uniform(-ydim/2, ydim/2)
                zpos = random.uniform(-zdim/2, zdim/2)
                particles.append([xpos, ypos, zpos, type1])

            # place the type2 beads
            for i in range(num_particles//2):
                xpos = random.uniform(0, xdim//2)
                ypos = random.uniform(-ydim/2, ydim/2)
                zpos = random.uniform(-zdim/2, zdim/2)
                particles.append([xpos, ypos, zpos, type2])

            if vis==True:
                view_list(particles, type1, type2)

            # Calculate the initial Lennard-Jones energy
            print("Calculating initial energy...")            
            energy = self.lj_energy(particles, xdim, ydim, zdim) 
            print(f"Energy calculation concluded. Initial energy: {energy}")

            # Begin equilibration steps
            t0 = time.time()
            accepted = 0

            mov_num = 0
            mov_acc = 0 # particle move acceptance

            vol_num = 0
            vol_acc = 0 # volume acceptance

            swa_num = 0
            swa_acc = 0 # swap acceptance

            for step in range(ecycles):
                if pfreq!=0 and step % pfreq == 0 and step!=0: 
                    t1 = time.time()
                    print(f"{step}: Vol: {np.round(xdim*ydim*zdim, 4)} Energy: {np.round(energy, 6)} \t| P: {mov_acc}/{mov_num} V: {vol_acc}/{vol_num} S: {swa_acc}/{swa_num} \t| Acc.: {accepted}/{step}\t ({np.round(accepted/step, 2)}) \t Time: {np.round(t1 - t0, 2)}")
                    t0 = time.time()
                # pick the move to make
                move_pick = random.uniform(0, 1.0)
                if move_pick < 0.6:
                    # PARTICLE MOVE -------------------------------------------------
                    mov_num+=1
                    # pick particle at random
                    index = random.randrange(0, num_particles)
                    particle = particles[index]                    
                    # Generate a new position for it         
                    #  1. Pick a random number for the displacement magnitude
                    #  2. Generate three values for the random displacement vector
                    #  3. Reduce the generated random displacement vector to unit 
                    #     length, then muliply it by the random displacement.
                    #  4. Add it to the position vector of the selected bead.
                    rand_mag = random.uniform(0, maxdx)
                    rand_disp = np.random.randn(1,3)
                    rand_disp = rand_mag*rand_disp/np.linalg.norm(rand_disp)
                    rand_disp = rand_disp[0]
                    
                    # calculate the new energy
                    energy_change = self.ec_disp(particles, particle, rand_disp, xdim, ydim, zdim)
                    
                    # perform particle test
                    test = random.uniform(0, 1.0)
                    if test < np.exp(-beta*energy_change):
                        new_position = periodize([particle[0]+rand_disp[0],
                                                  particle[1]+rand_disp[1],
                                                  particle[2]+rand_disp[2]], xdim, ydim, zdim)

                        accepted += 1
                        mov_acc += 1

                        particles[index][0] = new_position[0]
                        particles[index][1] = new_position[1]
                        particles[index][2] = new_position[2]
                        
                        energy += energy_change

                elif move_pick > 0.8:
                    # VOLUME MOVE ---------------------------------------------------
                    # pick a change in volume and calculate the change in scale
                    vol_num += 1
                    volume_change = random.uniform(-maxdv, maxdv)                    
                    old_vol = xdim*ydim*zdim
                    new_vol = old_vol + volume_change            
                    scale = (new_vol/old_vol)**(1/3) 

                    # scale atomic positions and lengths
                    new_particles = [[scale*i[0], scale*i[1], scale*i[2], i[3]] for i in particles]
                    new_xdim =  scale*xdim
                    new_ydim =  scale*ydim
                    new_zdim =  scale*zdim

                    # calculate the enthalpy change
                    enthalpy_change,  energy_change = self.enc_vol(new_particles, energy, new_xdim, new_ydim, new_zdim, volume_change, pressure)
                    # perform enthalpy test
                    test = random.uniform(0, 1.0)
                    if test < np.exp(-beta*enthalpy_change):
                        accepted += 1
                        vol_acc += 1                        
                        particles = new_particles
                        xdim = new_xdim
                        ydim = new_ydim
                        zdim = new_zdim

                        energy += energy_change

                else: 
                    # SWAP MOVE -----------------------------------------------------
                    # pick two particles at random
                    swa_num+=1
                    index1 = random.randrange(0, num_particles)
                    index2 = random.randrange(0, num_particles)
                    
                    # ensure that particles are not the same
                    while particles[index1][-1] == particles[index2][-1]:
                        index2 = random.randrange(0, num_particles)

                    # calculate the energy change of the swap
                    energy_change = self.ec_swap(particles, particles[index1], particles[index2], xdim, ydim, zdim)
                    
                    # perform energy test
                    test = random.uniform(0, 1.0)
                    if test < np.exp(-beta*energy_change):
                        energy += energy_change
                        # the type to be swapped
                        swap = particles[index1][-1]
                        particles[index1][-1] = particles[index2][-1]
                        particles[index2][-1] = swap
                        accepted += 1
                        swa_acc += 1

            if vis==True:
                view_list(particles, type1, type2)

            total_time1 = time.time()
            
            print(f"Total time: {total_time1 - total_time0}")

            # Begin sampling steps
            print("SAMPLING STAGE")

            # for step in range(scycles):
            #     if step % pfreq == 0: 
            #         print(f"Move: {step} of sampling procedure.")
            #     pass

            # Obtain the number fraction of the A block

            # Calculate the Flory Huggins parameter via the formula in Chremos, Nikoubashman and Panagiotopolous


