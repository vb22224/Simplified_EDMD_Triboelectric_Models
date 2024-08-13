# -*- coding: utf-8 -*-
'''Functions required for the running of the Lack model simulation'''

import numpy as np
import math as m
import pandas as pd
import statistics as stats
from scipy.optimize import minimize
from scipy.stats import norm



def round_up_to_1sf(number):
    
    '''Rounds a number up to its highest power of 10'''
    
    result = np.ceil(number / 10**np.floor(np.log10(np.abs(number)))) * 10**np.floor(np.log10(np.abs(number)))
    
    return result



def round_down_to_1sf(number):
    
    '''Rounds a number down to its highest power of 10'''
    
    result = np.floor(number / 10**np.floor(np.log10(np.abs(number)))) * 10**np.floor(np.log10(np.abs(number)))
    
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
    discrim = np.dot(pos_ij, vel_ij) ** 2 - (mag(vel_ij) ** 2) * ((mag(pos_ij) ** 2) - rad_ij ** 2)
    
    t = 'inf' # No collision
    
    if discrim > 0:
        t = - (1 / (mag(vel_ij) ** 2)) * (np.dot(pos_ij, vel_ij) + m.sqrt(discrim)) # Calculate time unill collision
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



def calc_charge_function_complex(x, a=130, d=-30):
    
    '''Calculates function used to fit the charge '''
    
    y = a * x ** (2) + d * x ** (-1) # If using the ^-1 fitting
    # y = a * x ** (2) + d * x ** (-3/2) # If -3/2 is decided as better than -1
    
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



def calc_charge_res_complex(x_list, y_list, a, d):
    
    '''Calculates the total residual and R² value between a set of values and a fit for the complex distribution shape'''
    
    TR, SSE, SST = 0, 0, 0
    x_av = stats.mean(x_list)
    
    for n, x in enumerate(x_list):
        y_fit = calc_charge_function_complex(x, a, d)
        R = np.abs(y_fit - y_list[n])
        TR += R
        SSE += R ** 2
        SST += (x - x_av) ** 2
        
    R2 = 1 - SSE / SST
    
    return TR, R2
    


def calc_energy_momentum(masses, v0):
    
    '''Calculates the total momentum and energy in the system'''
    
    energy, px, py, pz = 0, 0, 0, 0
    
    if len(masses) != len(v0):
        raise ValueError('The there are a different number of values for masses and velocites: {masses} and {v0}')
        
    for i, mass in enumerate(masses):
        speed = mag(v0[i])
        energy += 0.5 * mass * speed ** 2 
        px += mass * v0[i][0]
        py += mass * v0[i][1]
        pz += mass * v0[i][2]
    
    return energy, px, py, pz



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



def objective_function_complex(variables, *args):
    
    '''Used for the function minimisation with scipy.minimize for the comxplex distribution shape'''
    
    x_list, y_list, minimise_op = args

    a, d = variables[0], variables[1]
    TR, R2 = calc_charge_res_complex(x_list, y_list, a, d)
    
    if minimise_op == "TR":
        result = TR
    elif minimise_op == "R2":
        result = 1 - R2
    else:
        raise ValueError(f"minimise_op should be either total residual (TR), or R² value (R2). Not: {minimise_op}.")

    return result



def get_fit_complex(diameters, charge, initial_ad_guess=[130, -33], minimise_op="R2"):
    
    '''Gets the optimised fit for the charge for the comxplex distribution shape'''
    
    result = minimize(objective_function_complex, initial_ad_guess, args=(diameters, charge, minimise_op))
    a, d = result.x[0], result.x[1]
    
    return a, d
    


def check_modes(mode_sizes, mode_means, mode_stds):
    
    '''Checks the modes are the same length and given then sums them'''
    
    if len(mode_sizes) < 1:     
        raise ValueError("No mode sizes given")

    elif len(mode_sizes) != len(mode_means) or len(mode_sizes) != len(mode_stds):   
        raise ValueError("Mode parameter lengths differ")
    
    total_size = np.sum(mode_sizes)
    
    return total_size



def calc_size_freq(d, mode_sizes, mode_means, mode_stds, truncate=float('inf')):
    
    '''Function to calulate the frequency denisity for a given particle size'''

    frequency_density = 0
    total_size = check_modes(mode_sizes, mode_means, mode_stds)
    
    for number, mode_size in enumerate(mode_sizes):
        frequency_density += (mode_size / total_size) * norm.pdf(np.log10(d), np.log10(mode_means[number]), mode_stds[number])
    
    if d > truncate: # truncates the fequency density function
        frequency_density = 0
    
    return frequency_density



def linear_interp(x, x1, x2, y1, y2):
    
    '''Performas a linear interpolation'''

    y = y1 + (x - x1) * (y2 - y1) / (x2 - x1)

    return y



def log_trap_int(x_array, y_array, x1=0, x2=float('inf')):
    
    '''Performs a numerical integration in logspace using the trapezium rule bewteen the limits x1 and x2'''

    x_array = np.log10(x_array)  # Normalising in logspace
    
    if x1 == 0:
        x1 = min(x_array)
    if x2 == float('inf'):
        x2 = max(x_array)
    
    total_integral = 0

    for i in range(1, len(x_array)):
        x = x_array[i]
        if x > x1 and x < x2:
            total_integral += (x - x_array[i - 1]) * \
                (y_array[i] + y_array[i - 1]) / 2
            if x_array[i - 1] > x:  # correction at beginning of trapezium
                y_x1 = linear_interp(
                    x_array[i - 1], x1, x, y_array[i - 1], y_array[i])
                total_integral -= (x1 - x_array[i - 1]) * \
                    (y_x1 + y_array[i - 1]) / 2

    return total_integral



def get_half(x_array, y_array):
    
    '''Gets the x value for y = 0.5 in a cumulative distribution'''
    
    for n, y in enumerate(y_array[:-1]):
        if y_array[n + 1] >= 0.5:
            x_at_half = linear_interp(0.5, y,y_array[n + 1], x_array[n], x_array[n + 1])
            break
    
    return x_at_half


def calc_coll_traj(v0, r0, masses, i, j):
    
    '''Calculates the trajectories of two particles i and j after a collision conserving energy'''
    
    ui, uj, Rij, Rji = v0[i], v0[j], r0[j] - r0[i], r0[i] - r0[j]
    mi, mj, Rij_hat, Rji_hat = masses[i], masses[j], Rij / mag(Rij), Rji / mag(Rji)
    ui_para, uj_para = (np.dot(ui, Rij_hat)) * Rij_hat, (np.dot(uj, Rji_hat)) * Rji_hat
    vi_perp, vj_perp = ui - ui_para, uj - uj_para
    vi_para = (ui_para * (mi - mj) + 2 * mj * uj_para) / (mi + mj)
    vj_para = (uj_para * (mj - mi) + 2 * mi * ui_para) / (mj + mi)
    vi, vj = vi_para + vi_perp, vj_para + vj_perp
    
    return vi, vj
    


def calc_coll_traj_cons(v0, r0, masses, i, j):
    
    '''Calculates the trajectories of two particles i and j after a collision conserving energy and momentum'''
    
    m1, m2 = masses[i], masses[j]
    u1, u2, r1, r2 = v0[i], v0[j], r0[i], r0[j]
    e = 1 #coefficient of restitution
     
    r_rel = r2 - r1 # Calculate the relative position vectors
    r_rel_norm = r_rel / np.linalg.norm(r_rel) # Calculate the relative velocity along the line of impact
    m_eff = 1 / ((1/m1) + (1/m2))
    v_imp = np.dot(r_rel_norm, (u1 - u2))
    J = (1 + e) * m_eff * v_imp
    delta_v1 = (- J / m1) * r_rel_norm
    delta_v2 = (J / m2) * r_rel_norm
     
    # Calculating the new velocities of particles i and j
    vi = u1 + delta_v1
    vj = u2 + delta_v2
     
    return vi, vj

    