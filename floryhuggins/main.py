# Program name: main.py
# Description: This is a recreation of the Flory-Huggins calculation method described in
#              Chremos, Nikoubashman and Panagiotopolous 2014.
#              The program by itself is extremely inefficient for large numbers of 
#              particles: the simulation outlined in the paper, with 8186 particles over
#              fifty thousand moves, would take weeks upon weeks to run.

import time
import math as m
import copy
import random
import numpy as np
import itertools
import multiprocessing
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

print("MC SIMULATION FOR FLORY-HUGGINS PARAMETER OF LENNARD-JONES LIQUID")
results = open(f"Results{random.randint(0,10000)}.txt", 'w')

def view_list(particles, epsilon1, epsilon2):
    type1_beads = [i for i in particles if i[-1] == epsilon1]                
    type2_beads = [i for i in particles if i[-1] == epsilon2]
    
    type1_posx = [i[0] for i in type1_beads]
    type1_posy = [i[1] for i in type1_beads]
    type1_posz = [i[2] for i in type1_beads]

    type2_posx = [i[0] for i in type2_beads]
    type2_posy = [i[1] for i in type2_beads]
    type2_posz = [i[2] for i in type2_beads]
                
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter3D(type1_posx, type1_posy, type1_posz, c='r', s=200)
    ax.scatter3D(type2_posx, type2_posy, type2_posz, c='b', s=200)
    plt.show()    

lj_procs = 20
pw_procs = 5
results.write("-----------------------------------------------------------------\n")
results.write("MC SIMULATION FOR FLORY-HUGGINS PARAMETER OF LENNARD-JONES LIQUID\n")
results.write("-----------------------------------------------------------------\n")
results.write("The routine used in flory_huggins was developed with reference to the work by Chremos, Nikoubashman and Panagiotopolous (2014). Please cite these authors if you make use of this function for published work!\n")

# Variables to be defined
temp = 1.0
epsilon11 = 1.0
epsilon22 = 0.5
epsilon12 = 0.2

if (epsilon11 == epsilon22):
    raise ValueError("Setting the same value for the energy parameter and different values for the distance parameter will result in an incorrect calculation. This is easy to fix, but as of writing it is unnecessary for me.'")

# simulation details
attempts = 1 # attempts per cycle
ecycles = 25000
scycles = 5000
sfreq = 100
pfreq = 100

vis = True

mov_prob = 0.6 # 0.6
vol_prob = 0.8 # 0.8

# dimensions of cuboidal box
base = 16.0
ydim = base
zdim = base
xdim = 2.0*base

# physical details
num_particles = 8000
maxdx = 0.01*base
maxdv = 0.005*xdim*ydim*zdim
swap_cut = 0.4*base
cutoff_tol = 10000 # the maximum number of attempts for cut off selections
beta = 1.0
pressure = 0.0

dims = [xdim, ydim, zdim] # will be used occasionally for periodization

results.write(f"\nNumber of particles: {num_particles} ")
results.write(f"\nVolume: {xdim*ydim*zdim} ")
results.write(f"\nPressure: {pressure} ")
results.write(f"\nBeta: {beta} \n")

total_time0 = time.time()

# create an object that permanently houses the particle list
class ParticleList:
    def __init__(self,):
        self.original = []
        self.particles = []
    
pl = ParticleList()

def distance_between(index1, index2):
    particles = pl.particles    
    pos1 = np.array([particles[index1][0], particles[index1][1], particles[index1][2]])
    pos2 = np.array([particles[index2][0], particles[index2][1], particles[index2][2]])

    return np.linalg.norm(pos2-pos1)

def place_bead(epsilon, x_range, y_range, z_range, minimum):
    """ 
    Place beads whilst avoiding bead that have already been placed.
    """
    particles = pl.particles
    xpos = random.uniform(x_range[0], x_range[1])
    ypos = random.uniform(y_range[0], y_range[1])
    zpos = random.uniform(z_range[0], z_range[1])
    
    valid_position = False
    failure_count = 0
    while valid_position == False:
        issues = 0
        for p in particles:
            new_pos = [xpos, ypos, zpos]
            bead_pos = np.array([p[0], p[1], p[2]])
            
            # sigma: fixed to be 1.0
            # this function will cause issues for systems in which the sigmas are not set to unity
            if np.linalg.norm(new_pos - bead_pos) < minimum:
                issues += 1
                break
        if issues > 0:
            xpos = random.uniform(x_range[0], x_range[1])
            ypos = random.uniform(y_range[0], y_range[1])
            zpos = random.uniform(z_range[0], z_range[1])

            failure_count += 1            
            if failure_count > 1000:
                # relax the distance condition a little bit: monte carlo will sort out most issues anyway
                print(f"Minimum distance dropped to {minimum}.")
                minimum = 0.99*minimum 
        else:
            valid_position = True

            
    pl.particles.append([xpos, ypos, zpos, 1.0, epsilon])
    

# place the type1 beads
for i in range(num_particles//2):
    place_bead(epsilon11, (-xdim/2, 0), (-ydim/2, ydim/2), (-zdim/2, zdim/2), 0.8)    
    print(f"{i} beads successfully placed!")
    
# place the type2 beads
beads_in_box = len(pl.particles)
for i in range(num_particles//2):
    place_bead(epsilon22, (0, xdim/2), (-ydim/2, ydim/2), (-zdim/2, zdim/2), 0.8)
    if i % 10 == 0:
        print(f"{i+beads_in_box} beads successfully placed!")

if vis == True:
    view_list(pl.particles, epsilon11, epsilon22)

a_block = [i for i in pl.particles if i[0] < 0]
a_count = 0
for i in a_block:
    if i[-1] == epsilon11:
        a_count +=1

e1_count = 0
e2_count = 0
for i in pl.particles:
    if i[-1] == epsilon11:
        e1_count+=1
    if i[-1] == epsilon22:
        e2_count+=1

# ----------------------------------------------------------------------------------------------
# PLACE FUNCTIONS HERE: this is done to ensure that implicit arguments are defined beore the
#                       function is given.
#                       Implicit functions are used very heavily in this script for the purpose
#                       of making the use of multiprocessor more streamlined.

def pairwise_energy(iteration, xlen=xdim, ylen=ydim, zlen=zdim):    
    particles = pl.particles

    i, j = iteration[0], iteration[1]
    pos1 = np.array([particles[i][0], particles[i][1], particles[i][2]])
    pos2 = np.array([particles[j][0], particles[j][1], particles[j][2]])

    # the calculation will fail. 
    en1 = particles[i][-1]
    en2 = particles[j][-1]
    if en1 != en2: 
        # if the types are different:
        sigma = 1.0
        epsilon = epsilon12
    else: 
        # if the types are the same, you can choose the parameters of any of the particles involved
        sigma = particles[i][-1]
        epsilon = particles[i][-2]
        
    displacement = pos2 - pos1

    # correct for periodicity
    dims = [xlen, ylen, zlen]
    for dim in range(3):
        if displacement[dim] > dims[dim]/2:
            displacement[dim] = displacement[dim]-dims[dim]
        elif displacement[dim] < -dims[dim]/2:
            displacement[dim] = displacement[dim]+dims[dim]
    distsq = np.linalg.norm(displacement)

    return epsilon*((sigma/distsq)**6 - (sigma/distsq)**3) 

def swap_pairwise_energy(function_data, xlen=xdim, ylen=ydim, zlen=zdim):
    # function data:
    # 1. itr - the index of the iterated particle
    # 2. sw1 - the index of the first particle being swapped
    # 3. sw2 - the index of the second particle being swapped

    particles = pl.particles

    itr, sw1, sw2 = function_data[0], function_data[1], function_data[2]
    pos1 = np.array([particles[itr][0], particles[itr][1], particles[itr][2]])
    pos2 = np.array([particles[sw1][0], particles[sw1][1], particles[sw1][2]])

    # retrieve the value of the positions
    displacement = pos2 - pos1
    # correct for periodicity
    dims = [xlen, ylen, zlen]
    for dim in range(3):
        if displacement[dim] > dims[dim]/2:
            displacement[dim] = displacement[dim]-dims[dim]
        elif displacement[dim] < -dims[dim]/2:
            displacement[dim] = displacement[dim]+dims[dim]
    distsq = np.linalg.norm(displacement)

    # if the first particle being swapped is the same as particle being iterated over, return 0
    if itr == sw1:
        return 0.0

    # The energy BEFORE the swap --------------------------------------------------------------------
    ben1 = particles[sw1][-1]
    ben2 = particles[sw2][-1]
    if ben1 != ben2: 
        # if the types are different:
        bsigma = 1.0
        bepsilon = epsilon12
    else: 
        # if the types are the same, you can choose the parameters of any of the particles involved
        bsigma = particles[itr][-1]
        bepsilon = particles[itr][-2]

    old_energy = bepsilon*((bsigma/distsq)**6 - (bsigma/distsq)**3)    

    # The energy AFTER the swap ---------------------------------------------------------------------
    sigma_sw1 = particles[sw2][-2]
    epsilon_sw1 = particles[sw2][-1] 
    if itr == sw2:
        # in this case, you need to change the type of k as well.
        sigma_itr = particles[sw1][-2]
        epsilon_itr = particles[sw1][-1]
    else:
        sigma_itr = particles[itr][-2]
        epsilon_itr = particles[itr][-1]        

    if epsilon_itr != epsilon_sw1:
        sigma = 1.0
        epsilon = epsilon12
    else:
        sigma = sigma_sw1
        epsilon = sigma_sw1       

    new_energy = epsilon*((sigma/distsq)**6 - (sigma/distsq)**3)
    
    return new_energy - old_energy

    
def position_energy(function_data, xlen=xdim, ylen=ydim, zlen=zdim):
    """
    function_data includes
    1. the iterated index
    2. the index of the particle being moved
    3. the new position
    """

    # set the initialized particles to those saved within the ParticleList object
    particles = pl.particles
    
    i = function_data[0]
    j = function_data[1]

    if i == j:
        return 0.0

    ind_pos = np.array([particles[i][0], particles[i][1], particles[i][2]]) # the index position
    old_pos = np.array([particles[j][0], particles[j][1], particles[j][2]]) # initial position of the particle being moved
    new_pos = function_data[2]

    # Implicitly assuming that the sigma values are the same. 
    # This means that, if the epsilons for different types are the same and the sigmas are different,
    # the calculation will fail. 
    en1 = particles[i][-1]
    en2 = particles[j][-1]
    if en1 != en2: 
        # if the types are different:
        sigma = 1.0
        epsilon = epsilon12
    else: 
        # if the types are the same, you can choose the parameters of any of the particles involved
        sigma = particles[i][-1]
        epsilon = particles[i][-2]
        
    old_displacement = old_pos - ind_pos
    new_displacement = new_pos - ind_pos

    # periodise the old displacement vector
    dims = [xlen, ylen, zlen]
    for dim in range(3):
        if old_displacement[dim] > dims[dim]/2:
            old_displacement[dim] = old_displacement[dim]-dims[dim]
        elif old_displacement[dim] < -dims[dim]/2:
            old_displacement[dim] = old_displacement[dim]+dims[dim]

    # periodize the new displacement vector
    for dim in range(3):
        if new_displacement[dim] > dims[dim]/2:
            new_displacement[dim] = new_displacement[dim]-dims[dim]
        elif new_displacement[dim] < -dims[dim]/2:
            new_displacement[dim] = new_displacement[dim]+dims[dim]

    o_distsq = np.linalg.norm(old_displacement)
    n_distsq = np.linalg.norm(new_displacement)
    
    o_en = epsilon*((sigma/o_distsq)**6 - (sigma/o_distsq)**3)
    n_en = epsilon*((sigma/n_distsq)**6 - (sigma/n_distsq)**3)
    
    return n_en - o_en


# ----------------------------------------------------------------------------------------------
# Calculate the initial Lennard-Jones energy
t0 = time.time()
results.write("Calculating initial energy...\n")            
pool = multiprocessing.Pool(processes=lj_procs)    
params = itertools.combinations(range(len(pl.particles)), 2)
energy = 4*sum(pool.map(pairwise_energy, params))
pool.close()
t1 = time.time()
results.write(f"Energy calculation concluded in {t1 - t0} seconds. Initial energy: {energy}\n\n")

# Begin equilibration steps
t0 = time.time()
accepted = 0

mov_num = 0 
mov_acc = 0 # particle move acceptance

vol_num = 0

vol_acc = 0 # volume acceptance

swa_num = 0
swa_acc = 0 # swap acceptance

number_frac = []
energy_list = [energy]
volume_list = [xdim*ydim*zdim]
step_list = [0]

for step in range(ecycles+scycles):
    # simulation details
    if pfreq!=0 and step % pfreq == 0 and step!=0: 
        t1 = time.time()        
        results.write(f"{step}: Vol: {np.round(xdim*ydim*zdim, 4)} Energy: {np.round(energy, 6)} \t| P: {mov_acc}/{mov_num} V: {vol_acc}/{vol_num} S: {swa_acc}/{swa_num} \t| Acc.: {accepted}/{step*attempts}\t ({np.round(accepted/step*attempts, 2)}%) \t Time: {np.round(t1 - t0, 2)}\n")

        print(f"Move: {step}, Energy: {energy}, Volume: {xdim*ydim*zdim}")
        t0 = time.time()

    for attempt in range(attempts):
        # pick the move to make
        move_pick = random.uniform(0, 1.0)
        if move_pick < mov_prob:
            # PARTICLE MOVE -------------------------------------------------
                        
            mov_num+=1
            # pick particle at random
            index = random.randrange(0, num_particles)
            particle = pl.particles[index]        
            
            # Generate a new position for the picked particle
            #  1. Pick a random number for the displacement magnitude
            #  2. Generate three values for the random displacement vector
            #  3. Reduce the generated random displacement vector to unit 
            #     length, then muliply it by the random displacement.
            #  4. Add it to the position vector of the selected bead.
            rand_mag = random.uniform(0, maxdx)
            rand_disp = np.random.randn(1,3)
            rand_disp = rand_mag*rand_disp/np.linalg.norm(rand_disp)
            rand_disp = rand_disp[0]
            
            new_position = np.array([particle[0], particle[1], particle[2]]) + rand_disp
            for dim in range(3):
                if new_position[dim] > dims[dim]/2:
                    new_position[dim] = new_position[dim]-dims[dim]
                elif new_position[dim] < -dims[dim]/2:
                    new_position[dim] = new_position[dim]+dims[dim]

            # calculate the energy of the new position
            pool = multiprocessing.Pool(processes=pw_procs)    
            params = ((p, index, new_position) for p in range(len(pl.particles)))
            energy_change = 4*sum(pool.map(position_energy, params))        
            pool.close()

            # perform particle test
            test = random.uniform(0, 1.0)
            if test < np.exp(-beta*energy_change):

                pl.particles[index][0] = new_position[0]
                pl.particles[index][1] = new_position[1]
                pl.particles[index][2] = new_position[2]

                accepted += 1
                mov_acc += 1
                energy += energy_change

        elif move_pick > vol_prob:
            # VOLUME MOVE ---------------------------------------------------
            # pick a change in volume and calculate the change in scale
            vol_num += 1
            volume_change = random.uniform(-maxdv, maxdv)                    
            old_vol = xdim*ydim*zdim
            new_vol = old_vol + volume_change            
            scale = (new_vol/old_vol)**(1/3) 

            # scale atomic positions and lengths
            old_particles = copy.deepcopy(pl.particles)
            new_xdim =  scale*xdim
            new_ydim =  scale*ydim
            new_zdim =  scale*zdim
            pl.particles = [[scale*i[0], scale*i[1], scale*i[2], i[3], i[4]] for i in pl.particles]

            # calculate the new energy
            pool = multiprocessing.Pool(processes=lj_procs)    
            params = itertools.combinations(range(len(pl.particles)), 2)
            new_energy = 4*sum(pool.map(pairwise_energy, params))
            pool.close()
            
            energy_change = new_energy - energy
            num_atoms = len(pl.particles)

            enthalpy_change = (energy_change + pressure*volume_change) + num_atoms*(m.log(new_vol) - m.log(old_vol))
            # perform enthalpy test
            test = random.uniform(0, 1.0)
            if test < np.exp(-beta*enthalpy_change):
                accepted += 1
                vol_acc += 1                        
                xdim = new_xdim
                ydim = new_ydim
                zdim = new_zdim

                energy = new_energy

            else:
                pl.particles = old_particles

        else: 
            # SWAP MOVE -----------------------------------------------------
            # pick two particles at random
            swa_num+=1
            index1 = random.randrange(0, num_particles)
            index2 = random.randrange(0, num_particles)

            
            cutoff_cond = (distance_between(index1, index2) > swap_cut)
            cutoff_count = 0
            cflag = False
            
            # ensure that particles are not the same and aren´t within the cutoff
            while (pl.particles[index1][-1] == pl.particles[index2][-1]) or cutoff_cond:
                index1 = random.randrange(0, num_particles)
                index2 = random.randrange(0, num_particles)

                if cutoff_cond == True:
                    cutoff_count+=1
                if cutoff_count > cutoff_tol:
                    cflag = True
                    break;

            if cflag == True:
                continue
            
            # save the old configuration
            old_particles = copy.deepcopy(pl.particles)

            # get the energies of the particles to be swapped. These should be different anyway, 
            # but below is the easiest configuration
            i1_en = pl.particles[index1][-1]
            i2_en = pl.particles[index2][-1]

            # swap the particle energies
            pl.particles[index1][-1] = i2_en
            pl.particles[index2][-1] = i1_en

            # calculate the new energy
            pool = multiprocessing.Pool(processes=lj_procs)    
            params = itertools.combinations(range(len(pl.particles)), 2)
            swapped_energy = 4*sum(pool.map(pairwise_energy, params))
            pool.close()

            energy_change = swapped_energy - energy

            # perform energy test
            test = random.uniform(0, 1.0)
            if test < np.exp(-beta*energy_change):
                accepted += 1
                swa_acc += 1       
                energy = swapped_energy
            else:
                pl.particles = old_particles
              

    if step>ecycles and step%sfreq==0:
        # calculate the number fraction of the A block: the first half of the box in the x dimension
        a_block = [i for i in pl.particles if i[0] < 0]
        
        a_count = 0
        for i in a_block:
            if i[-1] == epsilon11:
                a_count +=1

        number_frac.append(a_count/len(a_block))

    step_list.append(step)
    energy_list.append(energy)
    volume_list.append(xdim*ydim*zdim)
            
# recalculate the complete energy
pool = multiprocessing.Pool(processes=lj_procs)    
params = itertools.combinations(range(len(pl.particles)), 2)
calc_energy = 4*sum(pool.map(pairwise_energy, params))
pool.close()
results.write(f"ENERGY CHECK:\nSimulation energy: {energy} \t Calculated energy: {calc_energy} \t Diff: {energy-calc_energy} \n")            

# stop the timer
total_time1 = time.time()

if vis==True:
    view_list(pl.particles, epsilon11, epsilon22)    

# average out the number fraction and calculate the flory huggins parameter

avg_nf = np.mean(number_frac)

flory_huggins = np.log(1/avg_nf - 1)/(1 - 2*avg_nf)

if pl.particles == pl.original:
    print("Nothing has moved! Something has gone wrong!")

results.write(f"\nRESULTS:")
results.write(f"\nNumber fraction: {avg_nf} \t Flory-Huggins Parameter: {flory_huggins}\n")
results.write(f"Total time: {total_time1 - total_time0}\n")

print(f"\n SIMULATION CONCLUDED. {total_time1 - total_time0} \n")

fig, (ax1, ax2) = plt.subplots(2, sharex=True)
fig.suptitle("Energy and volume change with cycles")
ax1.plot(step_list, energy_list)
ax2.plot(step_list, volume_list)
plt.show()

results.close()
