#variables
.PHONY: clean bins

#project
BINS := convolution
CC := icc

bins: $(BINS)

#dirs
DIR := ../../..
INC := -I$(DIR)/include

#LIBS
LIBS := $(DIR)/lib/libblitz.a

#flags
LDFLAGS := -Wl,--no-as-needed -lyaml-cpp -lhdf5 -lglog -lboost_chrono -lboost_thread -lboost_date_time -lboost_system -lxsmm
CXXFLAGS := -Wall -Wno-unused-parameter -O1 -fPIC -fopenmp -xMIC-AVX512 -g
CXXFLAGS += -DBLITZ_RELEASE
CXXFLAGS += -DBLITZ_NUM_THREADS=64
CXXFLAGS += -DBLITZ_ALIGNMENT_SIZE=64
CXXFLAGS += -DBLITZ_USE_MIC

#mkl for compare
LDFLAGS += -lmkl_intel_lp64 -lmkl_core -lmkl_gnu_thread -lpthread -ldl
CXXFLAGS += -DUSE_MKL
INC += -I/opt/intel/mkl/include

$(BINS): % : %.cc $(OBJECTS)
	$(CC) $(CXXFLAGS) $(INC) $^ $(LIBS) $(LDFLAGS) -o $@

clean:
	-rm -rf $(BINS)

#utils
print-% : ; $(info $* is $(flavor $*) variable set to [$($*)]) @true
