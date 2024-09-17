
# EDMD_Triboelectric_Models

A simple event-driven molecular dynamics models for triboelectric charging of aerosols, used to produce the EDMD output files in: https://www.doi.org/10.18126/n3dx-dh16.


 The main single electron model is the "lack_model.py", this then calls "lack_functions.py" for various functions and uses "lack_plots.py" to produce plots. The outputs of the simulation are written to "lack_model_output.txt" which can be read back in for plotting using "lack_reader.py". Animations can also be prosuced from "lack_animator.py" provided the associated option is turned on in the main "lack_model.py" before running to produce the nessecary output files. There is also a C++ version "charging_sim.cpp" that runs more quickly for larger simulations.

 The simulation implements a simplified EDMD simulation of hard spheres with only single electrons transferred, in an effort to simulate triboelectric charging. This work is based upon the approach of previous works by Daniel J. Lacks.
