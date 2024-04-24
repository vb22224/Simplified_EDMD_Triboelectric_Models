# -*- coding: utf-8 -*-
'''Functions required for plotting graphs in the Lack's model'''

import numpy as np
import matplotlib.pyplot as plt
import lack_functions as lf
from matplotlib.ticker import ScalarFormatter



def plot_3D(data, scale, size, box_length=0, charge_range=[0, 0], show=True, save=False, save_path='', dpi=600):
    
    '''Plots the data as a scatter plot in 3D, sized by radii and coloured by scale'''
    
    fig = plt.figure(dpi=dpi)
    
    # Create a grid of subplots with 1 row and 2 columns
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 0.05])

    # Plot the 3D scatter plot
    ax = fig.add_subplot(gs[0], projection='3d')
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



def ke_histogram(vel_data, masses, n_bins=15, dpi=600):
    
    '''Plots histogram of kinetic energies of particles from array of 3D velocities'''
    
    speed_data = []
    
    for v in vel_data:
        speed_data.append(lf.mag(v))
    
    speed_data = np.array(speed_data)
    ke = 0.5 * np.multiply(masses, np.square(speed_data))
    
    plt.figure(dpi=dpi)
    plt.hist(ke, bins=n_bins)
    
    plt.xlabel('Kinetic Energy / *UNITS*')
    plt.ylabel('Number of Particles')
    
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
    
    



