TARGETS = hw2seq hw2a hw2b hw2a_basic hw2b_basic
CC = gcc
CXX = g++
LDLIBS = -lpng
OPTFLAGS = -march=native -fopt-info-vec-all
CFLAGS = -lm -O3
hw2a: CFLAGS += -pthread
hw2b: CC = mpicc
hw2b: CXX = mpicxx
hw2b: CFLAGS += -fopenmp
hw2a_basic: CFLAGS += -pthread
hw2b_basic: CC = mpicc
hw2b_basic: CXX = mpicxx
hw2b_basic: CFLAGS += -fopenmp
CXXFLAGS = $(CFLAGS)

.PHONY: all
all: $(TARGETS)

.PHONY: clean
clean:
	rm -f $(TARGETS)
