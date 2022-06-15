NVFLAGS  := -std=c++11 -O3 -Xptxas="-v" -arch=sm_61 
LDFLAGS  := -lm
EXES     := hw4-1

all: $(EXES)

clean:
	rm -f $(EXES)

hw4-1: hw4-1.cu
	nvcc $(NVFLAGS) $(LDFLAGS) -o $@ $?
