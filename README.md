
# EDMD_Triboelectric_Models

 My simple event-driven molecular dynamics models for triboelectric charging of aerosols

 The main single electron model is the "lack_model.py", this then calls "lack_functions.py" for various functions and uses "lack_plots.py" to produce plots. The outputs of the simulation are written to "lack_model_output.txt" which can be read back in for plotting using "lack_reader.py"

 The simulation implements a naive EDMD simulation of hard spheres with only single electrons transferred, in an effort to simulate triboelectric charging. This work is based upon the approach of previous works by Daniel J. Lacks.
