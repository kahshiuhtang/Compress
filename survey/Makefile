NVCC= nvcc
CC  = g++

CUDA_INCLUDEPATH=/usr/local/cuda-12/include

NVCC_OPTS=-O3 -arch=sm_70 -Xcompiler -m64 -g -G

GCC_OPTS=-O3 -Wall -Wextra -m64

OBJ = main.o compress.o decompress.o
SRC = main.cc
EXE = survey

ifdef TESTING 
override TESTING = -DTESTING
endif

ifdef CACHECWLUT
override TESTING = -DCACHECWLUT
endif

survey: $(OBJ) 
	$(NVCC) $(TESTING) $(CACHECWLUT) $(NVCC_OPTS) $(OBJ) -o $(EXE) 

decompress.o: decompress.cu 
	$(NVCC) $(TESTING) -c decompress.cu $(NVCC_OPTS)

compress.o: compress.cu 
	$(NVCC) -c compress.cu $(NVCC_OPTS)

pack_kernels.o: pack_kernels.cu 
	$(NVCC) -c pack_kernels.cu $(NVCC_OPTS)

main.o: main.cc 
	$(NVCC) $(TESTING) -c main.cc $(NVCC_OPTS) -I $(CUDA_INCLUDEPATH) 

.o:.cpp
	$(CC) ++ $(GCC_OPTS) -c $@ -o $<

clean:
	rm -f *.o $(EXE) 