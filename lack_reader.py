"""Reads the outputs from my Lack model simulations so the data can be used in graphs etc"""

import numpy as np
import pandas as pd
import seaborn as sns
import lack_plots as lp



file_name = 'lack_model_output.txt'

df = pd.read_csv(file_name, sep=',', header=16) #header may need to be altered for different versions of the code's ouptut file
# print(df.columns.values)
# print(max(df['final high e']))

vel = df[['velocity(x)', 'velocity(y)', 'velocity(z)']].to_numpy()
pos = df[['position(x)', 'position(y)', 'position(z)']].to_numpy()
radii, charge = df['radii'], -df['charge']

# sns.scatterplot(x = "radii", y = "charge", data = df) # Plot charge against radii

# lp.size_histogram(radii, n_bins=15)
# lp.speed_histogram(vel, 10)
# lp.scatter_speeds(vel, radii)
lp.plot_3D(pos, charge, radii)
# lp.plot_charge(radii, charge)


