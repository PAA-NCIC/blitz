#project
PROJECT := blitz
BINS := convolution_implicit_asm
NVCC := nvcc
CC := g++

#dirs
DIR := ../../..
INC := -I$(DIR)/include
BUILD_DIR := $(DIR)/build
SRC_ROOT := $(DIR)/src
SRCS := $(shell find $(SRC_ROOT) -maxdepth 4 -name "*.cc" ! -name $(PROJECT).cc)
NVCC_SRCS := $(shell find $(SRC_ROOT) -maxdepth 4 -name "*.cu" ! -name $(PROJECT).cc)

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

OBJECTS := $(addprefix $(BUILD_DIR)/, $(patsubst %.cc, %.o, $(SRCS:$(SRC_ROOT)/%=%)))
NVCC_OBJECTS := $(addprefix $(BUILD_DIR)/, $(patsubst %.cu, %.o, $(NVCC_SRCS:$(SRC_ROOT)/%=%)))

$(BINS): % : %.cc $(OBJECTS) $(NVCC_OBJECTS)
	$(CC) $(CXXFLAGS) $(INC) $(LDFLAGS) -o $@ $^

clean:
	-rm -rf $(BINS)

#utils
print-% : ; $(info $* is $(flavor $*) variable set to [$($*)]) @true
