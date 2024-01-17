3D World Reconstruction Project

Overview

This project aims to reconstruct a 3D model using images captured from four different cameras, focusing on creating a detailed three-dimensional representation of an object (a fish) by processing masked images to calculate the object's volume.

Requirements

Python 3.12
Numpy library
OpenCV library
Matplotlib library

Setup

Ensure that Python 3 and the required libraries (Numpy, OpenCV, and Matplotlib) are installed on your system. If not, they can be installed using pip:
pip install numpy opencv-python matplotlib

Running the Program

To run the program, follow these steps:

Clone or download this repository to your local machine.
Ensure that the folder containing the masked images is in an accessible directory.
Run the program
When prompted, enter the full path to the folder containing the masked images.

Input Data

The input data for this project consists of masked images of an object taken from four different camera angles, essential for the 3D reconstruction process.

Important Notes on Input Data:
The masked images should be in a single folder.
You will be prompted to enter the path to this folder after starting the script. Ensure you enter the correct path.
The images must be named as specified by Professor Furkan Kıraç. It is critical to maintain the naming convention for the program to correctly process the images.

Output

The program will produce:

The calculated volume of the object in cubic centimeters.
A 3D plot of the voxel grid that represents the object.

Contact

For any queries or issues regarding the code, please feel free to contact Güneş Altıner / gunes.altiner@ozu.edu.tr
