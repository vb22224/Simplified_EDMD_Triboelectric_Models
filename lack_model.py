#!/usr/bin/env python3

'''
Lacks model implemented with a simple time driven molecular dynamic simulation under the single electrontranfer assumption
2nd model iteration aimed to increse speed with only re-computing relevant colission times in the colission time matrix
'''

import numpy as np
import math as m
from tqdm import tqdm
import lack_plots as lp
import lack_functions as lf
import csv



def main(**args):

    '''Main run of the simulation'''

    # Passing parameters then setting size, mass, position and velcity of all the particles

    target_packing, n_particles = args['target_packing'], args['n_particles']
    n_eqil_steps, n_sim_steps = args['n_eqil_steps'], args['n_sim_steps']
    radius_l, radius_u = args['radius_l'], args['radius_u']
    speed_l, speed_u = args['speed_l'], args['speed_u']
    mass_dependent, density = args['mass_dependent'], args['density']
    lower_radius_frac, poly_disperse = args['lower_radius_frac'], args['poly_disperse']
    charge_density = args['charge_density']
    log_mean, log_sd = args['log_mean'], args['log_sd']
    
    # Debugging options
    position_check = False
    energy_momentum_check = False
    
    # Create animation?
    animate = False
    save_animation = False
    animation_save_dir = "..\\Animation\\test2\\" # Directory to save file for animation
    charge_range=[-5, 20]
    dpi = 600

    # Setting the size distribution depending ong the opton chosen

    if poly_disperse == "bi": # Bidisperse
        radii = np.concatenate([np.ones(m.ceil(n_particles * lower_radius_frac)) * radius_l, np.ones(m.floor(n_particles * (1 - lower_radius_frac))) * radius_u])
        
    elif poly_disperse == "lin": # Linear
        radii = np.random.rand(n_particles) * (radius_u - radius_l) + radius_l
        
    elif poly_disperse == "lognorm": # Lognormal
        radii = (10 ** np.random.normal(loc=np.log10(log_mean), scale=log_sd, size=n_particles)) / 2 # divied by 2 as radii
        
    elif poly_disperse == "multimodal": # Edit here custom distribution, example case trimodal lognormal distribution
        
        mode_sizes = 0.9998908343003384, 0.00010916569966157586
        mode_means = 140.00269129864694, 1999.999926146656
        mode_stds = 0.2822161924076099, 0.1779400474177674
    
        total_size = np.sum(mode_sizes)
        radii = np.array([])
        
        for number, mode in enumerate(mode_sizes):
            mode_particles = int(n_particles * mode / total_size)
            mode_radii = (10 ** np.random.normal(loc=np.log10(mode_means[number]), scale=mode_stds[number], size=mode_particles)) / 2 # divied by 2 as radii
            radii = np.concatenate((radii, mode_radii))
        
        truncation = float('inf') # used to truncate the size distribution function
        radii = radii[radii <= truncation]
        
        n_particles = len(radii) # In case a particle has been lost / gained in rounding or truncating
    
    elif poly_disperse == "convert vol":
        
        dp_fit = np.logspace(np.log10(0.1), np.log10(10000), 1000, base=10) # Continua for fit calculation
        vp = np.pi * np.power(dp_fit, 3) / 6 # Volume at each size
        truncate = float('inf')
        vol_density, radii = [], np.array([])
        
        # Information about the volume distribution to be converted
        mode_sizes = [0.2702522105709556, 0.5377299225945315, 0.14637881818382076]
        mode_means = [47.68674473289051, 129.87119736614284, 600.1805862890564]
        mode_stds = [0.16414445691995336, 0.1917548803130691, 0.37970119431980565]
        
        for d in dp_fit: # Calculating multimodal volume density
            vol_density.append(lf.calc_size_freq(d, mode_sizes, mode_means, mode_stds, truncate))
        
        vol_density = np.array(vol_density)
        num_density = vol_density / vp # Converting to number density
        
        vol_density /= lf.log_trap_int(dp_fit, vol_density) # Normalisations
        num_density /= lf.log_trap_int(dp_fit, num_density)
        vol_cum = np.cumsum(vol_density) / sum(vol_density) # Cumulative Dstributions
        num_cum = np.cumsum(num_density) / sum(num_density)
        
        # Information on the diffenternt distribution averages
        print(f'Max volume density: {dp_fit[np.argmax(vol_density)]}')
        print(f'Max number density: {dp_fit[np.argmax(num_density)]}') 
        print(f'Half volume density: {lf.get_half(dp_fit, vol_cum)}')
        print(f'Half number density: {lf.get_half(dp_fit, num_cum)}')
        
        lp.plot_size_dists(dp_fit, vol_density, num_density, y_label='Normalised Frequency Density', dpi=200) # Plotting distributions
        lp.plot_size_dists(dp_fit, vol_cum, num_cum, y_label='Cumulative Frequency Density', dpi=200) # Plotting cumulative distributions

        for n in range(n_particles):
            radii = np.concatenate((radii, dp_fit[np.argwhere(num_cum > np.random.rand())[0]] / 2))

        n_particles = len(radii) # In case a particle has been lost / gained in rounding or truncating

    
    elif poly_disperse == "dustyboi": # David's distribution
        
        quantity = np.round(n_particles * np.array([0.128315105, 0.183572807, 0.124311959, 0.087425835, 0.062477661, 0.051611981, 0.135821002, 0.208449496, 0.018014154]))
        particle_sizes = np.array([1.0, 3.0, 6.0, 12.0, 23.5, 47.0, 94.0, 187.5, 375.0])
        ranges = np.array([1.0, 1.0, 2.0, 4.0, 7.5, 16.0, 31.0, 62.5, 125.0])
        psq = np.vstack((particle_sizes, quantity, ranges)).T
        radii = [0.5 * (size + ((2 * (np.random.random() - 0.5)) * rad_range)) for size, q, rad_range in psq for _ in range(int(q))]
                
        n_particles = np.size(radii)

    else:
        raise ValueError("poly_disperse should be lognormal (lognorm), linear (lin), bidisperse (bi), or custom (multimodal).")
        
    # Setting Particle Mass
    
    if mass_dependent == True:
        masses = lf.v_spherical_mass(radii, density)
    elif mass_dependent == False:
        masses = np.ones(n_particles)

    # Calculating the number of initial high energy states
    
    high_energy_states = lf.v_surface_charge_calc(radii, charge_density)
    energy_states = np.stack((high_energy_states, np.zeros(high_energy_states.size, dtype=int)), axis=1)
    
    # Calculates the required box size for the specified density and puts the particles in the box
    
    box_length = lf.calc_box_len(target_packing, radii)
    print(f'In a box of length: {round(box_length / 1000, 2)} mm')

    p_overlaps, w_overlaps = [], []
    n_p_overlaps,  n_w_overlaps = 1, 1
    r0 = np.random.rand(n_particles, 3) * box_length #initial positions

    while n_p_overlaps != 0 and n_w_overlaps != 0: #checks for overlapping particles or particles not in box and replaces them
        p_overlaps = lf.overlap_check(r0, radii, n_particles)
        n_p_overlaps = len(p_overlaps)
        p_overlaps = np.sort(p_overlaps)[:: -1 ]

        for o in p_overlaps:
            r0 = np.delete(r0, o, axis=0)

        r0 = np.concatenate([r0, (np.random.rand(n_p_overlaps, 3) * box_length)])

        w_overlaps = lf.out_of_bounds(r0, radii, n_particles, box_length)
        n_w_overlaps = len(w_overlaps)
        w_overlaps = np.sort(w_overlaps)[:: -1 ]

        for o in w_overlaps:
            r0 = np.delete(r0, o, axis=0)

        r0 = np.concatenate([r0, (np.random.rand(n_w_overlaps, 3) * box_length)])
    
    if position_check:
        print(f'Number of coordinates initially less than 0: {np.count_nonzero(r0 < 0)}')
        print(f'Number of coordinates initially greater than the length of box: {np.count_nonzero(r0 > box_length)}')
        print(f'Number of particles: {n_particles}')
    
    
    # Sets the speeds of the particles

    speed = (np.random.rand(n_particles) - 0.5) * 2 * (speed_u - speed_l)

    for s in range(n_particles):
        if speed[s] < 0:
            speed[s] -= speed_l
        else:
            speed[s] += speed_l

    direction =  (np.random.rand(n_particles, 3) - 0.5) * 2
    norm_vel = np.empty(0)

    for d in range(n_particles):
        x, y, z = direction[d][0], direction[d][1], direction[d][2]
        norm = m.sqrt(x ** 2 + y ** 2 + z ** 2)
        norm_vel = np.append(norm_vel, [x / norm, y / norm, z / norm])

    speed = np.reshape(speed, (n_particles, 1))
    norm_vel = np.reshape(norm_vel, (n_particles, 3))
    v0 = norm_vel * speed
    #speed_histogram(v0, 10)
    
    # Intiallising collision times
    
    coll_times = np.full((n_particles, n_particles), np.inf) # Particle collision matrix
    wall_times = np.full((n_particles, 6), np.inf) # 6 sides of the cube
    
    for i in range(n_particles):
        wall_times[i] = lf.t_wall(r0[i], v0[i], radii[i], box_length) # Times until each wall is hit
        for j in range(i + 1, n_particles):
            coll_times[i][j] = lf.t_coll(r0[i], r0[j], v0[i], v0[j], radii[i], radii[j])
            
    # print(np.sum(~np.isinf(coll_times))) # Can check number of potential collisions
    print(f'Initial number of high energy states: {np.sum(high_energy_states)}')
    
    energy_list, px_list, py_list, pz_list = [], [], [], []
    
    escape, equilibrated = False, False
    frame, csv_name = 0, animation_save_dir + 'timings.csv'
    
    while escape == False: # Loop for the main simualation
        
        if equilibrated == False:
            n_sim_steps += n_eqil_steps
        
        for step in tqdm(range(n_sim_steps)): # With progress bar
        # for step in range(n_eqil_steps + n_sim_steps): # Without progress bar
            
            t_wall, t_coll = np.min(wall_times), np.min(coll_times) # First collsions

            if t_wall <= t_coll: # Wall collision
                # print('Wall!')
                time_step = t_wall
                r0 += (v0 * t_wall) # Update positions
                particle_index = np.where(wall_times == t_wall)[0][0] # Which partilcle hit
                wall_index = np.where(wall_times == t_wall)[1][0] # Which wall was hit
                wall_times -= t_wall # Update wall collision times until wall hit
                coll_times -= t_wall # Update particle collision times until wall hit
                v0[particle_index][wall_index // 2] *= -1 # Particle bounce off wall
                recalc_p = [particle_index] # Particle to recalulate the particle collions for
                
            else: # Partilcle collsion
                # print('Coll!')
                time_step = t_coll
                r0 += (v0 * t_coll) # Update positions
                i, j = int(np.where(coll_times == t_coll)[0][0]), int(np.where(coll_times == t_coll)[1][0]) # Which particles hit
                wall_times -= t_coll # Update wall collision times until particle hit
                coll_times -= t_coll # Update particle collision times until particle hit
            
                # v0[i], v0[j] = lf.calc_coll_traj(v0, r0, masses, i, j) # Maths to find new particle trajectories conserving energy but not momentum (Alex's way)
                v0[i], v0[j] = lf.calc_coll_traj_cons(v0, r0, masses, i, j) # Maths to find new particle trajectories conserving energy and momentum (David's way)
                
                recalc_p = [i, j] # Particles to recalulate the particle collions for
                wall_index = 'inf' # Not a wall hit
                
                if step > n_eqil_steps or equilibrated == True: # If not equilibrating -> do the charge transfer
                    equilibrated = True
                    ei, ej = energy_states[i], energy_states[j]
        
                    if ei[0] > 0:
                        energy_states[j][1] += 1
                        energy_states[i][0] -= 1
                    if ej[0] > 0:
                        energy_states[i][1] += 1
                        energy_states[j][0] -= 1

                if np.sum(energy_states[:, 0]) <= 0:
                    print("Simulation complete as no high-energy electrons remain")
                    escape = True
                    break
            
            for p in recalc_p: # Particles that need collsions recalculating
                # Recaculating relevant wall collsions
                wall_times[p] = lf.t_wall(r0[p], v0[p], radii[p], box_length, hit=wall_index)
                
                # Recaculating relevant particle collsions
                for a in range(0, p): # Vertically in collsion matrix
                    coll_times[a][p] = lf.t_coll(r0[a], r0[p], v0[a], v0[p], radii[a], radii[p])
                for b in range(p + 1, n_particles): # Horizontally in collsion matrix
                    coll_times[p][b] = lf.t_coll(r0[p], r0[b], v0[p], v0[b], radii[p], radii[b])
                
                if wall_index == 'inf': # If particle collsion
                    coll_times[i][j] = 'inf' # Particles cant immidiately re-collide
            
            if energy_momentum_check: # For checking that energy and momentum are conserved
                energy, px, py, pz = lf.calc_energy_momentum(masses, v0)
                energy_list.append(energy)
                px_list.append(px)
                py_list.append(py)
                pz_list.append(pz)
                # print(f'Total kinetic energy and momentum (x, y, z) in the system: {energy} and {px}, {py}, {pz}')
            
            if animate or save_animation: # Showing the simulation as it runs
                name = str(frame)
                save_name = animation_save_dir + name
                charges = energy_states[:,0] + energy_states[:,1] - high_energy_states
                lp.plot_3D(r0, -charges, radii, box_length=box_length/1000, charge_range=charge_range, show=animate, save=save_animation, save_path=save_name, dpi=dpi)
                
                if save_animation: # Saving the frames as the simulation goes
                    data = [f'{name}.png', time_step]
                    if frame == 0:
                        with open(csv_name, 'w', newline='') as file:
                            writer = csv.writer(file)
                            writer.writerow(['files', 'times'])
                            writer.writerow(data)
                    else:
                        with open(csv_name, 'a', newline='') as file:
                            writer = csv.writer(file)
                            writer.writerow(data)
                        
                    frame += 1
            
        escape2 = False
        
        while escape2 == False and escape == False: # Giving the option for the simulation to stop early
            if np.sum(energy_states[:, 0]) > 0:
                continue_option = input(f"The specified number of steps have been completed, however, there are still {np.sum(energy_states[:, 0])} high-energy electrons remaining out of the original {np.sum(high_energy_states)}, would you like to continue (Y/N): ")
                
                if continue_option == 'Y':
                    n_sim_steps = int(input("How many more simulation steps would you like?: "))
                    escape2 = True
                                      
                elif continue_option == 'N':
                    escape, escape2 = True, True
                
                else:
                    print(f'Error "Y" or "N" was expected but received: {continue_option}')
                
    charges = energy_states[:,0] + energy_states[:,1] - high_energy_states # This means its the change in the number of electrons (- of charge)
    lf.write_output(args, radii, masses, r0, v0, high_energy_states, energy_states, charges)

    # lp.size_histogram(radii, n_bins=15)
    # lp.speed_histogram(v0, 15)
    # lp.scatter_speeds(v0, radii)
    lp.plot_3D(r0, -charges, radii, box_length / 1000)
    
    if energy_momentum_check:
        lp.track_energy_momentum(energy_list, px_list, py_list, pz_list)
    
    if position_check:
        print(f'Number of coordinates fianlly less than 0: {np.count_nonzero(r0 < 0)}')
        print(f'Number of coordinates finally greater than the length of box: {np.count_nonzero(r0 > box_length)}')
        print(f'Number of particles: {n_particles}')

    return



if __name__ == "__main__":

    params = {
        'target_packing': 0.1, # Packing faction to target (Alex used 1.68%)
        'n_particles': 300,
        'n_eqil_steps': 100000, # Usually around 100000
        'n_sim_steps': 500000, 
        'radius_l': 0.3,
        'lower_radius_frac': 0.5,
        'radius_u': 1.7,
        'speed_l':1.0,
        'speed_u':2.0,
        'mass_dependent': True,
        'poly_disperse': "convert vol", # Options: lognormal (lognorm), linear (lin), bidisperse (bi), custom (multimodal), multimodal volume distribution to convert (convert vol), David's distribution (dustyboi)
        'density': 1.0, # Default of 1.0
        'charge_density': 0.05,
        'log_mean': 325.0976857912173, # Mean of the distribution if lognormal
        'log_sd': 0.09053671829707784 # Standard deviation of the distribution if lognormal
        }

    output = main(**params)
    #print(output)
    
    
    