#variables
.PHONY: clean bins

#project
BINS := convolution shuffle
CC := g++

bins: $(BINS)

#dirs
DIR := ../../..
INC := -I$(DIR)/include

#libs
LIBS := $(DIR)/lib/libblitz.a

#flags
LDFLAGS := -Wl,--no-as-needed -lyaml-cpp -lhdf5 -lglog -lcudart -lcuda -lcublas -lcudnn -lcurand \
  -lboost_chrono -lboost_thread -lboost_date_time -lboost_system
CXXFLAGS := -Wall -Wno-unused-parameter -O3 -fPIC -fopenmp
CXXFLAGS += -DBLITZ_RELEASE
CXXFLAGS += -DBLITZ_NUM_THREADS=16
CXXFLAGS += -DBLITZ_ALIGNMENT_SIZE=64

#mkl for compare
LDFLAGS += -lmkl_intel_lp64 -lmkl_core -lmkl_gnu_thread -lpthread -ldl
CXXFLAGS += -DUSE_MKL
INC += -I/opt/intel/mkl/include

$(BINS): % : %.cc $(OBJECTS) $(LIBS)
	$(CC) $(CXXFLAGS) $(INC) $(LDFLAGS) -o $@ $^

clean:
	-rm -rf $(BINS)

#utils
print-% : ; $(info $* is $(flavor $*) variable set to [$($*)]) @true
