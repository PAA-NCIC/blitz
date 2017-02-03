#variables
.PHONY: clean bins

#project
BINS := convolution pack
CC := g++

bins: $(BINS)

#dirs
PREFIX ?= /home/zkr/github/blitz
INC := -I$(PREFIX)/include

#LIBS
LIB := -L$(PREFIX)/lib

#flags
LDFLAGS := -Wl,--no-as-needed $(LIB) -lblitz
CXXFLAGS := -Wall -Wno-unused-parameter -O3 -fPIC -fopenmp -mavx

#mkl for compare
BLAS ?= openblas
ifeq ($(BLAS), mkl)
	CXXFLAGS += -DUSE_MKL
endif

$(BINS): % : %.cc
	$(CC) $(CXXFLAGS) $(INC) -o $@ $^ $(LDFLAGS) 

clean:
	-rm -rf $(BINS)

#utils
print-% : ; $(info $* is $(flavor $*) variable set to [$($*)]) @true
