"""Functions required for the running of the Lack model simulation"""

import numpy as np
import math as m
import pandas as pd



def mag(mat0):
    
    '''squares and sums the elements of a 1x3 matrix'''
    
    mag = m.sqrt(float(mat0[0]) ** 2 + float(mat0[1]) ** 2 + float(mat0[2]) ** 2)
    
    return mag
    
    


def overlap_check(r, radii, n_particles):

    '''Checks is particles are overlapping and if so thier indexes'''
    
    overlaps = []
    
    for i in range(0, n_particles):
        for j in range(i + 1, n_particles):
            
            Rij = radii[i] + radii[j]
            rij = r[i] - r[j]
            
            if mag(rij) < Rij and j not in overlaps:
                overlaps.append(j)
       
    return np.array(overlaps)



def out_of_bounds(r, radii, n_particles, box_length):
    
    '''Checks if particles are out of the box'''
    
    outside = []
    
    for i in range(0, n_particles):
    
        dx, dy, dz = r[i][0], r[i][1], r[i][2]
    
        if dx < radii[i] or dy < radii[i] or dz < radii[i]:
            outside.append(i)
        elif dx > box_length - radii[i] or dy > box_length - radii[i] or dz > box_length - radii[i]:
            outside.append(i)

    return np.array(outside)



def spherical_mass(radii, density):
    
    '''Takes array of raddi and returns radii of their spherical masses'''
    
    volume = (4 / 3) * m.pi * (radii ** 3)
    mass = volume * density
    return mass

v_spherical_mass = np.vectorize(spherical_mass)



def t_wall(position, velocity, box_length):
    
    '''Calculates time until all particles hits each wall'''
    
    if velocity < 0:
        t = (0.0 - position) / velocity
    else:
        t = (box_length - position) / velocity
    return t

vt_wall = np.vectorize(t_wall)



def t_coll(r0, v0, radii, n_particles):
    
    '''Calculates time until particles collide'''
    
    times = []
    
    for i in range(0, n_particles):
        for j in range(i + 1, n_particles):
            
            ri0, rj0, vi, vj, Ri, Rj = r0[i], r0[j], v0[i], v0[j], radii[i], radii[j]
            vij, rij0, Rij = vi - vj, ri0 - rj0, Ri + Rj
            discrim = np.dot(rij0, vij) - (mag(vij) ** 2) * ((mag(rij0) ** 2) - Rij ** 2)

            if discrim > 0 and np.dot(rij0, vij) < 0:
                t = (1 / (mag(vij) ** 2)) * (np.dot(rij0, vij) + m.sqrt(discrim))
                times.append([i, j, t])

    return np.array(times)



def calc_den(box_length, radii):
    
    '''Returns the packing fraction from inputs of the box length and radii of all particles'''

    box_volume = box_length ** 3
    particle_volume = 0.0
    
    for r in radii:
        particle_volume =+ (4 / 3) * m.pi * (r ** 3)
    
    packing_frac = particle_volume / box_volume

    return packing_frac
    


def surface_charge_calc(r, charge_density):
    
    '''Calculates the surface area and then surface charge of particle from given radii and charge density'''
    
    surface_area = 4 * m.pi * (r ** 2)
    surface_charge = int(surface_area * charge_density)
    # print(surface_area, surface_charge)
    
    return surface_charge

v_surface_charge_calc = np.vectorize(surface_charge_calc)



def write_output(params, radii, masses, r0, v0, high_energy_states, energy_states, charges):
    
    '''Writes the key data and parameters used to an output file'''
    
    with open('lack_model_output.txt', 'w') as f:
        
        f.write('Parameters: \n')
        for key,value in params.items():
            f.write(str(key) + '\t' + str(value) + '\n')
        
        f.write('\nradii,masses,position(x),position(y),position(z),velocity(x),velocity(y),velocity(z),initial high e,final high e,final low e,charge\n')
        df = pd.DataFrame(data=[radii, masses, r0[:,0], r0[:,1], r0[:,2], v0[:,0], v0[:,1], v0[:,2], high_energy_states, energy_states[:,0], energy_states[:,1], charges]).transpose()
        df_string = df.to_csv(index=False, header=False, sep=',')
        f.write(df_string)
        
    return



