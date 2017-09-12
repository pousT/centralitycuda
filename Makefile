NVCCFLAGS = --ptxas-options="-v" -arch sm_21
INCLUDES  = -I. -I/usr/local/cuda/samples/common/inc -I/usr/local/cuda/include
HEADER_FILES = graph_indexed.h cuBCStruct.h
SOURCE_FILES = BC_cpu.cpp timing.cu
OBJECT_FILES = BC_cpu.o
all: main

main: main.cu $(HEADER_FILES) $(SOURCE_FILES) $(OBJECT_FILES)
	nvcc $(NVCCFLAGS) $(INCLUDES) -o main main.cu $(OBJECT_FILES)

#BC_cpu.o : BC_cpu.cpp
#	nvcc -g $(NVCCFLAGS) -c BC_cpu.cpp -o BC_cpu.o $(INCLUDES)

%.o : %.cpp
	nvcc -g $(NVCCFLAGS) -c $< $(INCLUDES)

clean:
	rm -f *.o main main.linkinfo
