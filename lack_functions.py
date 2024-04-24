# -*- coding: utf-8 -*-
'''Functions required for the running of the Lack model simulation'''

import numpy as np
import math as m
import pandas as pd
import statistics as stats
from scipy.optimize import minimize




def round_up_to_1sf(number):
    
    '''Rounds a number up to its highest power of 10'''
    
    result = np.ceil(number / 10**np.floor(np.log10(np.abs(number)))) * 10**np.floor(np.log10(np.abs(number)))
    
    return result



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
        ri = radii[i]
    
        if dx < ri or dy < ri or dz < ri:
            outside.append(i)
        elif dx > box_length - ri or dy > box_length - ri or dz > box_length - ri:
            outside.append(i)

    return np.array(outside)



def spherical_mass(radii, density):
    
    '''Takes array of raddi and returns radii of their spherical masses'''
    
    volume = (4 / 3) * m.pi * (radii ** 3)
    mass = volume * density
    
    return mass

v_spherical_mass = np.vectorize(spherical_mass)



def t_wall(pos, vel, rad, b_len, hit='inf'):
    
    '''
    Calculates time until each wall is hit by a particle for version 2.0
    Takes position (pos) and velocity (vel) as lists/arrays and take the radii (rad) and box length (b_l) as integers/floats
    Returns an array of times in the order [x_lower, x_upper, y_lower, ...]
    If a particle is in a wall colision the parameter "hit" indicates this, having index of the wall hit as above, "inf" if not in collision
    '''

    if len(pos) != len(vel):
        raise ValueError('Dimension ERROR!') # Raise an error if the dimensions of position and velocity differ
        
    times = []
    
    for n1, p in enumerate(pos): # Iterate over position dimentions
        for n2 in range(2): # Two sides to box per dimension i.e. up/down
            d = b_len * n2 - pos[n1] - rad * (n2 * 2 - 1) # Distance to box wall from edge of radius
            t = d / vel[n1] # Time = Displacement / Velocity
            if t < 0 or hit == n1 * 2 + n2:
                t = 'inf' # If the particle is moving away or this is the particle in the collision: Time = Infinity
            times.append(t)

    return np.array(times)



def t_coll(pos_i, pos_j, vel_i, vel_j, rad_i, rad_j):
    
    '''
    Calculates time until particles (i and j) collide for version 2.0
    Takes position (pos) and velocity (vel) as lists/arrays and take the radii (rad) as integers/floats
    Returns the time until collision (t) as a float
    '''
    
    if not len(pos_i) == len(pos_j) == len(vel_i) == len(vel_j):
        raise ValueError('Dimension ERROR!') # Raise an error if the dimensions of position and velocity differ
    
    pos_ij = pos_i - pos_j # Relative position
    vel_ij = vel_i - vel_j # Relative velocity
    rad_ij = rad_i + rad_j # Combined radii
    
    # Descriminant of collision equation, if positive their paths meet
    discrim = np.dot(pos_ij, vel_ij) - (mag(vel_ij) ** 2) * ((mag(pos_ij) ** 2) - rad_ij ** 2)
    
    t = 'inf' # No collision
    
    if discrim > 0:
        t = (1 / (mag(vel_ij) ** 2)) * (np.dot(pos_ij, vel_ij) + m.sqrt(discrim)) # Calculate time unill collision
        if t < 0: # If particles moving away from each other
            t = 'inf'
        
    return t
    


def calc_box_len(target_packing, radii):
    
    '''Returns the packing box length required to acheive the target packing fraction for a specified radii array'''

    particle_volume = 0.0
    
    for r in radii:
        particle_volume += (4 / 3) * m.pi * (r ** 3)
    
    box_volume = particle_volume / target_packing
    box_length = box_volume ** (1 / 3)

    return box_length
    


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
            f.write(str(key) + ': \t' + str(value) + '\n')
        
        f.write('\nradii,masses,position(x),position(y),position(z),velocity(x),velocity(y),velocity(z),initial high e,final high e,final low e,charge\n')
        df = pd.DataFrame(data=[radii, masses, r0[:,0], r0[:,1], r0[:,2], v0[:,0], v0[:,1], v0[:,2], high_energy_states, energy_states[:,0], energy_states[:,1], charges]).transpose()
        df_string = df.to_csv(index=False, header=False, sep=',')
        f.write(df_string)
        
    return



def calc_charge_function(x, a=1, b=1.5, c=0):
    
    '''Calculates function used to fit the charge '''
    
    y = a * x ** b + c
    
    return y


    
def calc_charge_res(x_list, y_list, a, b, c):
    
    '''Calculates the total residual and R² value between a set of values and a fit'''
    
    TR, SSE, SST = 0, 0, 0
    x_av = stats.mean(x_list)
    
    for n, x in enumerate(x_list):
        y_fit = calc_charge_function(x, a, b, c)
        R = np.abs(y_fit - y_list[n])
        TR += R
        SSE += R ** 2
        SST += (x - x_av) ** 2
        
    R2 = 1 - SSE / SST
    
    return TR, R2
    


def objective_function(variables, *args):
    
    '''Used for the function minimisation with scipy.minimize'''
    
    x_list, y_list, minimise_op = args

    a, b, c = variables[0], variables[1] ,variables[2]
    TR, R2 = calc_charge_res(x_list, y_list, a, b, c)
    
    if minimise_op == "TR":
        result = TR
    elif minimise_op == "R2":
        result = 1 - R2
    else:
        raise ValueError(f"minimise_op should be either total residual (TR), or R² value (R2). Not: {minimise_op}.")

    return result



def get_fit(diameters, charge, initial_abc_guess=[0.003, 2, -50], minimise_op="R2"):
    
    '''Gets the optimised fit for the charge'''
    
    result = minimize(objective_function, initial_abc_guess, args=(diameters, charge, minimise_op))
    a, b, c = result.x[0], result.x[1], result.x[2]
    
    return a, b, c
    
    
    
    
    