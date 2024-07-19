 ##############################################################################
#
# Makefile to generate borders_cuda command. 
#
# To generate the GPU + OpenCV " version (An OpenCV installation should be already exist in your computer;)
# Uncomment the first CUDAFLAGS line, and comment out the second.
#     ( you probably need to adjust the paths to your installation)
# Type just "make" to generate the GPU version alone.
# 
# Authors: Victor Manuel Garcia Molla, Pedro Alonso Jorda and Ricardo Garci­a.
# September, 2021.
# 
##############################################################################


CUDAFLAGS=-lineinfo  -O3  -D OPENCV -I/usr/local/include/opencv4 -L/usr/local/lib -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_imgcodecs
#CUDAFLAGS=-lineinfo  -O3 -std=c++11  
NVCC=nvcc

all: borders_cuda

borders_cuda : borders_cuda.cu 
	$(NVCC) borders_cuda.cu -o borders_cuda $(CUDAFLAGS) 

clean:
	rm -f *.o borders_cuda.linkinfo  borders_cuda

