'''Functions required for plotting graphs in the Lack's model'''

import numpy as np
import matplotlib.pyplot as plt
import math as m
import lack_functions as lf



def plot_3D(data):
    
    '''Plots the data as a scatter plot in 3D'''
    
    fig = plt.figure(dpi=600)
    ax = fig.add_subplot(projection='3d')
    ax.scatter(data[:,0], data[:,1], data[:,2], color='b')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()
    


def speed_histogram(vel_data, n_bins):
    
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
    



