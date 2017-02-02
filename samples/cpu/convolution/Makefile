#variables
.PHONY: clean bins

#project
BINS := convolution pack
CC := g++

bins: $(BINS)

#dirs
DIR := ../../..
INC := -I$(DIR)/include

#LIBS
LIBS := $(DIR)/lib/libblitz.a

#flags
LDFLAGS := -Wl,--no-as-needed -lglog -lboost_thread -lboost_system
CXXFLAGS := -Wall -Wno-unused-parameter -O3 -fPIC -fopenmp -mavx

#mkl for compare
LDFLAGS += -lopenblas

$(BINS): % : %.cc $(OBJECTS) $(LIBS)
	$(CC) $(CXXFLAGS) $(INC) $(LDFLAGS) -o $@ $^

clean:
	-rm -rf $(BINS)

#utils
print-% : ; $(info $* is $(flavor $*) variable set to [$($*)]) @true
