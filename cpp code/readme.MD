cpp code compiles and runs the same code as the python script (though cpp)

the matlab code runs the previous version of the code (it doesn't have the box sizing fix yet) and plots the particles live using matlab link
- to run this you need to have Matlab installed, and MinGW64 installed and on the PATH (e.g. from https://github.com/niXman/mingw-builds-binaries)

the compile command is:
'g++ lack_model_matlab.cpp -I "C:\Program Files\MATLAB\R2024a\extern\include" -L "C:\Program Files\MATLAB\R2024a\extern\lib\win64\mingw64" -lMatlabEngine -lMatlabDataArray
 
