NVFLAGS := -std=c++11 -Xptxas="-v" -arch=sm_61 
CFLAGS  := -O3
XCFLAGS := -fopenmp
LDFLAGS := -lm
EXES    := hw4-2

all: $(EXES)

clean:
	rm -f $(EXES)

hw4-2: hw4-2.cu
	nvcc $(NVFLAGS) $(CFLAGS) $(LDFLAGS) -Xcompiler="$(XCFLAGS)" -o $@ $?
