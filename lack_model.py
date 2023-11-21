#!/usr/bin/env python3

'''Lacks model implimented with a simple time driven molecular dynamic simulation under teh single electrontranfer assumption'''

import numpy as np
import math as m
from tqdm import tqdm
import lack_plots as lp
import lack_functions as lf
import lack_functions as lf



def main(**args):

    '''Main run of the simulation'''

    #passing parameters then setting size, mass, position and velcity of all the particles

    box_length, n_particles = args['box_length'], args['n_particles']
    n_eqil_steps, n_sim_steps = args['n_eqil_steps'], args['n_sim_steps']
    radius_l, radius_u = args['radius_l'], args['radius_u']
    speed_l, speed_u = args['speed_l'], args['speed_u']
    mass_dependent, density = args['mass_dependent'], args['density']
    lower_radius_frac, poly_disperse = args['lower_radius_frac'], args['poly_disperse']
    charge_density = args['charge_density']
    log_mean, log_sd = args['log_mean'], args['log_sd']

    if poly_disperse == "bi":
        radii = np.concatenate([np.ones(m.ceil(n_particles * lower_radius_frac)) * radius_l, np.ones(m.floor(n_particles * (1 - lower_radius_frac))) * radius_u])
    elif poly_disperse == "lin":
        radii = np.random.rand(n_particles) * (radius_u - radius_l) + radius_l
    elif poly_disperse == "lognorm":
        radii = 10 ** np.random.normal(loc=np.log10(log_mean), scale=log_sd, size=n_particles)
    else:
        raise ValueError("poly_disperse should be lognormal (lognorm), linear (lin), or bidisperse (bi)")
        

    if mass_dependent == True:
        masses = lf.v_spherical_mass(radii, density)
    elif mass_dependent == False:
        masses = np.ones(n_particles)

    high_energy_states = lf.v_surface_charge_calc(radii, charge_density)
    energy_states = np.stack((high_energy_states, np.zeros(high_energy_states.size, dtype=int)), axis=1)

    overlaps = []
    n_overlaps = 1
    r0 = np.random.rand(n_particles, 3) * box_length #initial positions

    while n_overlaps != 0: #checks for overlaps and replaces them
        overlaps = lf.overlap_check(r0, radii, n_particles)
        n_overlaps = len(overlaps)
        overlaps = np.sort(overlaps)[:: -1 ]

        for o in overlaps:
            r0 = np.delete(r0, o, axis=0)

        r0 = np.concatenate([r0, (np.random.rand(n_overlaps, 3) * box_length)])

    while n_overlaps != 0: #checks for particles overlapping with the box and replaces them
        overlaps = lf.out_of_bounds(r0, radii, n_particles, box_length)
        n_overlaps = len(overlaps)
        overlaps = np.sort(overlaps)[:: -1 ]

        for o in overlaps:
            r0 = np.delete(r0, o, axis=0)

        r0 = np.concatenate([r0, (np.random.rand(n_overlaps, 3) * box_length)])

    speed = (np.random.rand(n_particles) - 0.5) * 2 * (speed_u - speed_l)

    for s in range(n_particles):
        if speed[s] < 0:
            speed[s] -= speed_l
        else:
            speed[s] += speed_l

    direction =  (np.random.rand(n_particles, 3) - 0.5) * 2
    norm_vel = np.empty (0)

    for d in range(n_particles):
        x, y, z = direction[d][0], direction[d][1], direction[d][2]
        norm = m.sqrt(x **2 + y ** 2 + z ** 2)
        norm_vel = np.append(norm_vel, [x / norm, y / norm, z / norm])

    speed = np.reshape(speed, (n_particles, 1))
    norm_vel = np.reshape(norm_vel, (n_particles, 3))
    v0 = norm_vel * speed
    #speed_histogram(v0, 10)
    phase = 0 # In the equilibration phase

    for step in tqdm(range(n_eqil_steps + n_sim_steps)):

        wall_times = lf.vt_wall(r0, v0, box_length) #checking how long untill a wall hit
        wall_time_index = wall_times.argmin()
        wall_time = wall_times[m.floor(wall_time_index / 3)][wall_time_index % 3]

        coll_times = lf.t_coll(r0, v0, radii, n_particles) #checking how long until a particle collision

        if len(coll_times) > 0:
            coll_time_index = coll_times[:,2].argmin()
            coll_time = coll_times[coll_time_index][2]
        else:
            coll_time = np.array([wall_time + 1]) #in case of no collisions


        if wall_time <= coll_time:
            #print('Wall!')
            r0 += (v0 * wall_time)
            v0[m.floor(wall_time_index / 3)][wall_time_index % 3] *= -1
        else:
            #print('Coll!')
            r0 += (v0 * coll_time)
            i, j = int(coll_times[coll_time_index][0]), int(coll_times[coll_time_index][1])
            ui, uj, Rij, Rji = v0[i], v0[j], r0[j] - r0[i], r0[i] - r0[j]
            mi, mj, Rij_hat, Rji_hat = masses[i], masses[j], Rij / lf.mag(Rij), Rji / lf.mag(Rji)

            ui_para, uj_para = (np.dot(ui, Rij_hat)) * Rij_hat, (np.dot(uj, Rji_hat)) * Rji_hat
            vi_perp, vj_perp = ui - ui_para, uj - uj_para
            vi_para = (ui_para * (mi - mj) + 2 * mj * uj_para) / (mi + mj)
            vj_para = (uj_para * (mj - mi) + 2 * mi * ui_para) / (mj + mi)
            vi, vj = vi_para + vi_perp, vj_para + vj_perp
            v0[i], v0[j] = vi, vj
            #print(mag(Rij))

            ei, ej = energy_states[i], energy_states[j]

            if ei[0] > 0:
                energy_states[j][1] += 1
                energy_states[i][0] -= 1
            if ej[0] > 0:
                energy_states[i][1] += 1
                energy_states[j][0] -= 1

        if step == n_eqil_steps:
            #print("\n Equilibriated")
            phase = 1 # In the simulation phase

    charges = energy_states[:,0] + energy_states[:,1] - high_energy_states

    lf.write_output(args, radii, masses, r0, v0, high_energy_states, energy_states, charges)

    # lp.speed_histogram(v0, 15)
    # lp.scatter_speeds(v0, radii)
    # lp.plot_3D(r0)
    print("\n", lf.calc_den(box_length, radii))

    #print("\n Simulation Complete")

    return



if __name__ == "__main__":

    params = {
        'box_length': 500,
        'n_particles': 100,
        'n_eqil_steps': 1000,
        'n_sim_steps': 10000,
        'radius_l': 0.3,
        'lower_radius_frac': 0.5,
        'radius_u': 1.7,
        'speed_l':1.0,
        'speed_u':2.0,
        'mass_dependent': True,
        'poly_disperse': "lognorm", # Should be lognormal (lognorm), linear (lin), or bidisperse (bi)
        'density': 1.0,
        'charge_density': 0.001,
        'log_mean': 30, # Mean of the distribution if lognormal
        'log_sd': 0.2 # Standard deviation of the distribution if lognormal
        }

    output = main(**params)
    #print(output)