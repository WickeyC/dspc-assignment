## Setting Up OpenCV
1. Download and install the ![OpenCV library](https://opencv.org/releases/) if you haven't already. 

2. In the Project "Properties" window, in Configuration, Select 'All Configuration'

3. In the "Configuration Properties" section, select "VC++ Directories".

4. Add the include directory of your OpenCV installation in the "Include Directories" field.

5. Next, go to "Linker" > "General" and add the path to OpenCV's "lib" directory in the "Additional Library Directories" field.

6. Click 'OK'

7. In the Project "Properties" window, in Configuration, perform step 8 below for both 'Debug' and 'Release'.
 
8. go to "Linker" > "Input", add the OpenCV libraries (the .lib file in the lib directory) in the "Additional Dependencies" field.

(Note: the .lib file ends with 'd' is for Debug, without d is for Release.)

## For OpenMP project
project name: openMP

### Enabling OpenMP
1. Must have a .cpp file first in the project directory.

2. In the Project "Properties" window, in Configuration, Select 'All Configuration'

3. go to "Configuration Properties" > "C/C++" > "Language".

4. Set "Open MP Support" to "Yes (/openmp)".

## For pThread project
project name: pThread

## For CUDA project
project name: CUDA
