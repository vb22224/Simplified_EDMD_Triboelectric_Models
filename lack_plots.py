'''Functions required for plotting graphs in the Lack's model'''

import numpy as np
import matplotlib.pyplot as plt
import lack_functions as lf
from matplotlib.ticker import ScalarFormatter



def plot_3D(data, scale, size):
    
    '''Plots the data as a scatter plot in 3D, sized by radii and coloured by scale'''

    fig = plt.figure(dpi=600)
    
    # Create a grid of subplots with 1 row and 2 columns
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 0.05])

    # Plot the 3D scatter plot
    ax = fig.add_subplot(gs[0], projection='3d')
    scatter = ax.scatter(data[:,0]/1000, data[:,1]/1000, data[:,2]/1000, s=size, c=scale, cmap='Spectral', alpha=0.6)

    ax.set_xlabel('X / mm')
    ax.set_ylabel('Y / mm')
    ax.set_zlabel('Z / mm')

    # Add colorbar to the right of the plot
    cax = fig.add_subplot(gs[1])
    cbar = plt.colorbar(scatter, cax=cax, pad=0.1)  # Adjust the pad parameter to set the distance

    cbar.set_label('Charge / e')

    plt.show()



def size_histogram(radii, n_bins=15):
    
    """Pots a gistogram for the size distribution of particles from their radii"""
    
    size_array = radii * 2
    
    plt.figure(dpi=600)
    
    log_bins = np.logspace(np.log10(size_array.min()), np.log10(size_array.max()), n_bins)
    plt.hist(size_array, bins=log_bins, edgecolor='black')
    
    plt.xlabel('d$_p$ / $\mathrm{\mu}$m')
    plt.ylabel('Frequency')
    plt.xscale('log')
    plt.gca().xaxis.set_major_formatter(ScalarFormatter())
    
    plt.show()
    


def speed_histogram(vel_data, n_bins=15):
    
    '''Plots histogram of speeds of particles from array of 3D velocities'''
    
    speed_data = []
    
    for v in vel_data:
        speed_data.append(lf.mag(v))
    
    plt.figure(dpi=600)
    plt.hist(speed_data, bins=n_bins)
    
    plt.xlabel('Particle Speed / $\lambda$s$^{-1}$')
    plt.ylabel('Number of Particles')
    
    plt.show()



def scatter_speeds(vel_data, radii):
    
    '''Plots a scatter of V^(-2/3) against radii'''
    
    speed_data = []
    
    for v in vel_data:
        speed_data.append(lf.mag(v))
    
    powers = np.ones(len(radii)) * (-2 / 3)
    y = np.power(speed_data, powers)
    
    plt.figure(dpi=600)
    plt.scatter(radii, y)
    
    plt.xlabel('Radius / $\lambda$')
    plt.ylabel('<V>$^{-2/3}$')
    
    plt.show()
    


def plot_charge(radii, charge):
    
    '''Plots charge normalised as a dimensionless quantity against radii'''
    
    plt.figure(dpi=600)
    
    plt.plot(radii, charge, "x", color='r')
    
    plt.xlabel('Radius / d$_p$')
    plt.ylabel('Charge / 4$\pi p_{H0}\lambda^2$e')
    
    plt.show()
    
    
