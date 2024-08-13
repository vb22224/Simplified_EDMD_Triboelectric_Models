# -*- coding: utf-8 -*-
'''Functions required for plotting graphs in the Lack's model'''

import numpy as np
import matplotlib.pyplot as plt
import lack_functions as lf
from matplotlib.ticker import ScalarFormatter
from scipy.optimize import curve_fit



def plot_3D(data, scale, size, box_length=0, charge_range=[0, 0], show=True, save=False, save_path='', dpi=600):
    
    '''Plots the data as a scatter plot in 3D, sized by radii and coloured by scale'''
    
    fig = plt.figure(dpi=dpi)
    
    # Create a grid of subplots with 1 row and 2 columns
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 0.05])

    # Plot the 3D scatter plot
    ax = fig.add_subplot(gs[0], projection='3d')
    if box_length != 0:
        size /= box_length

    scatter = ax.scatter(data[:,0]/1000, data[:,1]/1000, data[:,2]/1000, s=size, c=scale, cmap='Spectral', alpha=0.6)

    ax.set_xlabel('X / mm')
    ax.set_ylabel('Y / mm')
    ax.set_zlabel('Z / mm')
    
    if box_length != 0:
        ax.set_xlim([0, box_length])
        ax.set_ylim([0, box_length])
        ax.set_zlim([0, box_length]) 

    # Ensure equal aspect ratio
    ax.set_box_aspect([1,1,1])

    # Add colorbar to the right of the plot
    cax = fig.add_subplot(gs[1])
    cbar = plt.colorbar(scatter, cax=cax, pad=0.1)  # Adjust the pad parameter to set the distance
    if charge_range != [0, 0]:
        scatter.set_clim(charge_range[0], charge_range[1])  # Set the clim directly on the ScalarMappable object

    cbar.set_label('Charge / a.u.')

    if save:
        plt.savefig(save_path + '.png', bbox_inches='tight')  # Ensure tight bounding box

    if show:
        plt.show()

    plt.close()

    return



def size_histogram(radii, n_bins=15, dpi=600):
    
    """Pots a gistogram for the size distribution of particles from their radii"""
    
    size_array = radii * 2
    
    plt.figure(dpi=dpi)
    
    log_bins = np.logspace(np.log10(size_array.min()), np.log10(size_array.max()), n_bins)
    plt.hist(size_array, bins=log_bins, edgecolor='black')
    
    plt.xlabel('d$_p$ / $\mathrm{\mu}$m')
    plt.ylabel('Frequency')
    plt.xscale('log')
    plt.gca().xaxis.set_major_formatter(ScalarFormatter())
    
    plt.show()
    
    return
    


def speed_histogram(vel_data, n_bins=15, dpi=600):
    
    '''Plots histogram of speeds of particles from array of 3D velocities'''
    
    speed_data = []
    
    for v in vel_data:
        speed_data.append(lf.mag(v))
    
    plt.figure(dpi=dpi)
    plt.hist(speed_data, bins=n_bins)
    
    plt.xlabel('Particle Speed / $\lambda$s$^{-1}$')
    plt.ylabel('Number of Particles')
    
    plt.show()

    return    


def maxwell_boltzmann(x, A, T):
    
    '''Calcullates a point on a maxwell-boltzmann distribution'''
    
    return A * np.exp(-x / T) * (x ** 2)



def ke_histogram(vel_data, masses, n_bins=15, dpi=600):
    '''Plots histogram of kinetic energies of particles from array of 3D velocities'''
    speed_data = np.array([lf.mag(v) for v in vel_data])
    ke = 0.5 * np.multiply(masses, np.square(speed_data))
    
    plt.figure(dpi=dpi)
    bin_heights, bins, patches = plt.hist(ke, bins=n_bins, label='Simulation Data')
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    
    # Provide better initial guesses for parameters
    A_fit = 1000 /  (np.mean(ke) ** 2)
    T_fit = np.mean(ke) / 4

    params, covariance = curve_fit(maxwell_boltzmann, bin_centers, bin_heights, p0=[A_fit, T_fit])
    A_fit, T_fit = params
    
    x_fit = np.linspace(0, max(bin_centers), 1000) 
    plt.plot(x_fit, maxwell_boltzmann(x_fit, A_fit, T_fit), color='red', label='Maxwell-Boltzmann Fit')

    plt.xlabel('Kinetic Energy / *UNITS*')
    plt.ylabel('Number of Particles')
    plt.legend()
    plt.show()

    return



def scatter_speeds(vel_data, radii, dpi=600):
    
    '''Plots a scatter of V^(-2/3) against radii'''
    
    speed_data = []
    
    for v in vel_data:
        speed_data.append(lf.mag(v))
    
    powers = np.ones(len(radii)) * (-2 / 3)
    y = np.power(speed_data, powers)
    
    plt.figure(dpi=dpi)
    plt.scatter(radii, y)
    
    plt.xlabel('Radius / $\lambda$')
    plt.ylabel('<V>$^{-2/3}$')
    
    plt.show()
    
    return



def track_energy_momentum(energy_list, px_list, py_list, pz_list, dpi=600):
    
    '''Plots the normalised system momentum and energy over the steps of the simulation to check conservation'''
    
    if len(energy_list) != len(px_list) or len(energy_list) != len(py_list) or len(energy_list) != len(pz_list):
        raise ValueError(f'There is a mismatch in the energy and momnetum list lengths: {len(energy_list)} and {len(px_list)}, {len(py_list)}, {len(pz_list)}')
        
    plt.figure(dpi=dpi)
    
    steps = range(1, len(energy_list) + 1)
    plt.plot(steps, np.array(energy_list) / energy_list[0], label='energy')
    plt.plot(steps, np.array(px_list) / px_list[0], label='momentum (x)')
    plt.plot(steps, np.array(py_list) / py_list[0], label='momentum (y)')
    plt.plot(steps, np.array(pz_list) / pz_list[0], label='momentum (z)')
    plt.xlabel('Number of steps')
    plt.ylabel('Normalised value')
    
    plt.legend()
    plt.show()
    
    return



def plot_charge(radii, charge, name="Radii", dpi=600):
    
    '''Plots charge normalised as a dimensionless quantity against radii'''
    
    plt.figure(dpi=dpi)
    
    plt.plot(radii, charge, "x", color='r')
    
    plt.xlabel('$d_p$ / $\mu$m')
    # plt.ylabel('Charge / 4$\pi p_{H0}\lambda^2$e')
    plt.ylabel('Charge / a.u.')
    
    # plt.xlim(0, 500)
    # plt.ylim(-200, 40)
    
    return



def plot_fit(a, b, c, x_min, x_max, N=1000):
    
    '''Plots a function of y = a * x ^ b + c'''
    
    x_list = np.linspace(x_min, x_max + 1, num=N).tolist()
    y_list = []
    
    for x in x_list:
        y_list.append(lf.calc_charge_function(x, a, b, c))
    
    plt.plot(x_list, y_list, "--", color='black')
    
    return
    


def plot_fit_complex(a, d, x_min, x_max, N=1000):
    
    '''Plots a function of y = a * x ^ 2 + d * x ^ - 1'''
    
    x_list = np.linspace(x_min, x_max + 1, num=N).tolist()
    y_list = []
    
    for x in x_list:
        y_list.append(lf.calc_charge_function_complex(x, a, d))
    
    plt.plot(x_list, y_list, "--", color='black')
    
    return



def plot_size_dists(dp_fit, vol_density, num_density, y_label='Normalised Frequency Density', dpi=600):
    
    '''Plots the size distributions'''
    
    plt.figure(dpi=dpi)
    plt.plot(dp_fit, vol_density, label='Volume')
    plt.plot(dp_fit, num_density, label='Number')
    
    plt.xscale('log')
    plt.xlabel('d$_p$ / $\mu$m')
    plt.ylabel(y_label)
    plt.legend()
    plt.show()
    
    return



