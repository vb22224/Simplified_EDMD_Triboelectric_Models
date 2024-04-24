# -*- coding: utf-8 -*-
'''Reads the outputs from my Lack model simulations so the data can be used in graphs etc'''

import numpy as np
import pandas as pd
import lack_plots as lp
import lack_functions as lf



file_name = 'lack_model_output.txt'
# file_name = '..\\MAIN\\Ash_Charge_Distributions\\Output_Files\\Grimsvotn_Output_2.txt'

df = pd.read_csv(file_name, sep=',', header=16) #header may need to be altered for different versions of the code's ouptut file
# print(df.columns.values)
# print(max(df['final high e']))

vel = df[['velocity(x)', 'velocity(y)', 'velocity(z)']].to_numpy()
pos = df[['position(x)', 'position(y)', 'position(z)']].to_numpy()
radii, charge, mass = df['radii'], -df['charge'], df['masses']

electron_surface_density = 1
diameters = radii * 2
dimentionless_charge = charge / (4 * np.pi * electron_surface_density)

# lp.size_histogram(diameters, n_bins=15)
# lp.speed_histogram(vel, 10)
# lp.scatter_speeds(vel, radii)
lp.plot_3D(pos, charge, radii)
# lp.ke_histogram(vel, mass, 50)

lp.plot_charge(diameters, dimentionless_charge, "Diameter", dpi=600)
a, b, c = lf.get_fit(diameters, dimentionless_charge, initial_abc_guess=[0.00015620966892319275, 2.023552458071091, -0.6237543866506734], minimise_op="R2")
lp.plot_fit(a, b, c, 0, lf.round_up_to_1sf(max(diameters)))
print(f'a, b, c = {a}, {b}, {c}')


