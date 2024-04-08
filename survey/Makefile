NVCC= nvcc
CC  = g++

CUDA_INCLUDEPATH=/usr/local/cuda-12/include

NVCC_OPTS=-O3 -arch=sm_70 -Xcompiler -m64 -g -G

GCC_OPTS=-O3 -Wall -Wextra -m64

SRCD=src
BIND=bin
INCD=include
BLDD=build

OBJ = main.o search.o huff.o
SRC = main.cc
OBJ := $(addprefix $(BLDD)/, $(OBJ))
EXE = survey

INC := -I $(INCD)

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

$(BLDD)/decompress.o: $(SRCD)/decompress.cu 
	$(NVCC) $(TESTING) -c $(SRCD)/decompress.cu $(NVCC_OPTS) -o $(BLDD)/decompress.o

$(BLDD)/search.o: $(SRCD)/search.cu 
	$(NVCC) -c $(SRCD)/search.cu $(NVCC_OPTS) $(INC) -o $(BLDD)/search.o

$(BLDD)/main.o: $(SRCD)/main.cc 
	$(NVCC) $(TESTING) -c $(SRCD)/main.cc $(NVCC_OPTS) -I $(CUDA_INCLUDEPATH) $(INC) -o $(BLDD)/main.o

$(BLDD)/huff.o: $(SRCD)/huff.cc 
	$(NVCC) $(TESTING) -c $(SRCD)/huff.cc $(NVCC_OPTS) -I $(CUDA_INCLUDEPATH) $(INC) -o $(BLDD)/huff.o

.o:.cpp
	$(CC) ++ $(GCC_OPTS) -c $@ -o $<

clean:
	rm -f *.o $(BIND)/$(EXE) 
	rm -rf *.o $(BLDD)