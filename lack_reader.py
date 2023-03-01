"""Reads the outputs from my Lack model simulations so the data can be used in graphs etc"""

import pandas as pd
import seaborn as sns


file_name = 'lack_model_output.txt'

df = pd.read_csv(file_name, sep=',', header=14)
print(df.columns.values)

sns.scatterplot(x = "radii", y = "charge", data = df)

