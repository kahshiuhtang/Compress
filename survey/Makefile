NVCC= nvcc
CC  = g++

CUDA_INCLUDEPATH=/usr/local/cuda-12/include

NVCC_OPTS=-O3 -arch=sm_70 -Xcompiler -m64 -g -G

GCC_OPTS=-O3 -Wall -Wextra -m64

SRCD=src
BIND=bin
BLDD=build

OBJ = main.o compress.o decompress.o
SRC = main.cc
OBJ := $(addprefix $(BLDD)/, $(OBJ))
EXE = survey

ifdef TESTING 
override TESTING = -DTESTING
endif

ifdef CACHECWLUT
override TESTING = -DCACHECWLUT
endif

all: setup survey 

setup : $(BIND) $(BLDD)
$(BIND):
	mkdir -p $(BIND)
$(BLDD):
	mkdir -p $(BLDD)

survey: $(OBJ) 
	$(NVCC) $(TESTING) $(CACHECWLUT) $(NVCC_OPTS) $(OBJ) -o $(BIND)/$(EXE) 

$(BLDD)/decompress.o: decompress.cu 
	$(NVCC) $(TESTING) -c decompress.cu $(NVCC_OPTS) -o $(BLDD)/decompress.o

$(BLDD)/compress.o: compress.cu 
	$(NVCC) -c compress.cu $(NVCC_OPTS) -o $(BLDD)/compress.o

$(BLDD)/main.o: main.cc 
	$(NVCC) $(TESTING) -c main.cc $(NVCC_OPTS) -I $(CUDA_INCLUDEPATH) -o $(BLDD)/main.o

.o:.cpp
	$(CC) ++ $(GCC_OPTS) -c $@ -o $<

clean:
	rm -f *.o $(BIND)/$(EXE) 
	rm -rf *.o $(BLDD)