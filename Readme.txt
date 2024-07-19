CUDA border detection software


This zip file contains the source code "borders_cuda.cu", with an implementation of border tracking in GPUs written in CUDA, 
by Victor Manuel Garcia Molla, Pedro Alonso Jordá and Ricardo García. 
It has been tested in linux (UBUNTU) machines with CUDA toolkit up to 10.2.
It includes a Makefile with these two lines:

CUDAFLAGS=-lineinfo  -O3  -D OPENCV -std=c++11 -I/usr/local/include/opencv4 -L/usr/local/lib -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_imgcodecs
CUDAFLAGS=-lineinfo  -O3 -std=c++11 -L/usr/local/lib 


The first one is if you have a OpenCV installation (version tested 4.5) and want to compare the GPU code with the "findcontours" function of OpenCV; the second line is meant for the case when you do not 
have OpenCV installation (or are not interested in comparing OpenCV with our CUDA code). Please comment the line that you are not interested in.

It includes two test images, the Frame_08.bin image is input for the CUDA code and the Frame_08.bmp is the input for the OpenCV code. The output image is written 
in the file h_Asal.bin.
The "imagen.m" file is a Matlab/Octave file that allows visualizing the output image when run in the same directory.

The package is supplied "as is", without any accompanying support services, maintenance, or future updates. 
We make no warranties, explicit or implicit, that the software contained in this package is free of error 
or that it will meet your requirements for any particular application. It should not be relied on for any purpose 
where incorrect results could result in loss of property, personal injury, liability or whatsoever. If you do 
use our software for any such purpose, it is at your own risk. The authors disclaim all liability of any kind,
 either direct or consequential, resulting from your use of these programs.


