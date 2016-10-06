#project
PROJECT := blitz

#config
CONFIG_FILE := Makefile.config
include $(CONFIG_FILE)

#variables
.PHONY: clean all dirs bins objects

#dirs
BIN_DIR := bin
BUILD_DIR := build
SRC_ROOT := src

#compilers
OPTIMIZE_OPTIONS := -O3
#avx types
ifeq ($(BLITZ_AVX), 512)
    OPTIMIZE_OPTIONS += -DBLITZ_AVX_WIDTH=64 -xMIC-AVX512
else ifeq ($(BLITZ_AVX), 3)
  OPTIMIZE_OPTIONS += -DBLITZ_AVX_WIDTH=64 -xCORE-AVX512
else ifeq ($(BLITZ_AVX), 2)
  OPTIMIZE_OPTIONS += -DBLITZ_AVX_WIDTH=32 -mavx2
else
  OPTIMIZE_OPTIONS += -DBLITZ_AVX_WIDTH=32 -mavx
endif

OPENMP_OPTIONS := -fopenmp
CXXFLAGS := -Wall -Wno-unused-parameter -fPIC $(OPENMP_OPTIONS) $(OPTIMIZE_OPTIONS) 
INC := -Iinclude/

ifeq ($(CPU_ONLY), 1)
  CXXFLAGS += -DBLITZ_CPU_ONLY
else
  NVCC := nvcc
  NVCC_INC := -Iinclude/
  NVCC_XCOMPILE := -O3 -Wall -fopenmp -fPIC
  NVCC_FLAGS := -O3 $(CUDA_ARCH) --use_fast_math -ccbin $(CC)
endif

#dependency
DEPDIR := $(BUILD_DIR)
$(shell mkdir -p $(DEPDIR) >/dev/null)
DEPFLAGS = -MT $@ -MMD -MP -MF $(DEPDIR)/$*.Td
POSTCOMPILE = mv -f $(DEPDIR)/$*.Td $(DEPDIR)/$*.d

#libraries
ifeq ($(CPU_ONLY), 1)
  LDFLAGS := -Wl,--no-as-needed -lyaml-cpp -lhdf5 -lglog -lboost_chrono -lboost_thread -lboost_date_time \
    -lboost_system
else
  LDFLAGS := -Wl,--no-as-needed -lyaml-cpp -lhdf5 -lglog -lcudart -lcuda -lcublas -lcudnn -lcurand \
    -lboost_chrono -lboost_thread -lboost_date_time -lboost_system
endif

#blitz
ifeq ($(BLITZ_MODE), release)
  CXXFLAGS += -DBLITZ_RELEASE
  NVCC_XCOMPILE += -DBLITZ_RELEASE
else ifeq ($(BLITZ_MODE), performance)
  CXXFLAGS += -DBLITZ_PERFORMANCE
  NVCC_XCOMPILE += -DBLITZ_PERFORMANCE
else ifeq ($(BLITZ_MODE), develop)
  CXXFLAGS += -DBLITZ_DEVELOP -g
  NVCC_XCOMPILE += -DBLITZ_DEVELOP -g
endif

CXXFLAGS += -DBLITZ_NUM_THREADS=$(BLITZ_NUM_THREADS)
CXXFLAGS += -DBLITZ_ALIGNMENT_SIZE=$(BLITZ_ALIGNMENT_SIZE)

#blas
BLAS ?= atlas
ifeq ($(BLAS), mkl)
  #MKL
  LDFLAGS += -lmkl_intel_lp64 -lmkl_core -lmkl_gnu_thread -lpthread -ldl
  CXXFLAGS += -DUSE_MKL
  MKL_DIR ?= /opt/intel/mkl
  BLAS_INCLUDE ?= $(MKL_DIR)/include
  BLAS_LIB ?= $(MKL_DIR)/lib/intel64
else ifeq ($(BLAS), atlas)
  #ATLAS
  LDFLAGS += -lcblas -latlas
endif

ifdef BLAS_INCLUDE
  INC += -I$(BLAS_INCLUDE)
endif

ifdef BLAS_LIB
  LIBRARY_DIR += -L$(BLAS_LIB)
endif

ifeq ($(STATIC_LINK), 1)
  LDFLAGS := -Wl, --start-group $(LDFLAGS) -Wl, --end-group
endif

#variables
SRCS := $(shell find $(SRC_ROOT) -maxdepth 4 -name "*.cc" ! -name $(PROJECT).cc ! -path "*backup*" ! -path "*samples*")

OBJECTS := $(addprefix $(BUILD_DIR)/, $(patsubst %.cc, %.o, $(SRCS:$(SRC_ROOT)/%=%)))
OBJECTS_DIR := $(sort $(addprefix $(BUILD_DIR)/, $(dir $(SRCS:$(SRC_ROOT)/%=%))))

BINS := $(BIN_DIR)/$(PROJECT)

AUTODEPS:= $(patsubst %.o, %.d, $(OBJECTS)) $(patsubst %.o, %.d, $(NVCC_OBJECTS))

ifeq ($(CPU_ONLY), 0)
  NVCC_SRCS := $(shell find $(SRC_ROOT) -maxdepth 4 -name "*.cu" ! -name $(PROJECT).cc ! -path "*backup*" ! -path "*samples*")
  NVCC_OBJECTS := $(addprefix $(BUILD_DIR)/, $(patsubst %.cu, %.o, $(NVCC_SRCS:$(SRC_ROOT)/%=%)))
  NVCC_OBJECTS_DIR := $(sort $(addprefix $(BUILD_DIR)/, $(dir $(NVCC_SRCS:$(SRC_ROOT)/%=%))))
  AUTODEPS:= $(AUTODEPS) $(patsubst %.o, %.d, $(NVCC_OBJECTS))
endif

#rules
#mkdir first
all: dirs bins objects 

ifeq ($(CPU_ONLY), 1)
  ALL_OBJECTS_DIR := $(OBJECTS_DIR)
  ALL_BINS_DIR := $(BIN_DIR)
else
  ALL_OBJECTS_DIR := $(sort $(OBJECTS_DIR) $(NVCC_OBJECTS_DIR))
  ALL_BINS_DIR := $(BIN_DIR)
endif

dirs: $(ALL_BINS_DIR) $(ALL_OBJECTS_DIR)

$(ALL_BINS_DIR):
	mkdir -p $@

$(ALL_OBJECTS_DIR):
	mkdir -p $@

bins: $(BINS)

ifeq ($(CPU_ONLY), 1)
  $(BINS): $(BIN_DIR)/% : $(SRC_ROOT)/%.cc $(OBJECTS)
	  $(CC) $(CXXFLAGS) $(INC) $(LIBRARY_DIR) $(LDFLAGS) -o $@ $^

  objects: $(OBJECTS)

  $(OBJECTS): $(BUILD_DIR)/%.o : $(SRC_ROOT)/%.cc $(BUILD_DIR)/%.d 
	  $(CC) $(DEPFLAGS) $(CXXFLAGS) $(INC) -o $@ -c $<
	  $(POSTCOMPILE)
else
  $(BINS): $(BIN_DIR)/% : $(SRC_ROOT)/%.cc $(OBJECTS) $(NVCC_OBJECTS)
	  $(CC) $(CXXFLAGS) $(INC) $(LIBRARY_DIR) $(LDFLAGS) -o $@ $^

  objects: $(OBJECTS) $(NVCC_OBJECTS)

  $(OBJECTS): $(BUILD_DIR)/%.o : $(SRC_ROOT)/%.cc $(BUILD_DIR)/%.d 
	  $(CC) $(DEPFLAGS) $(CXXFLAGS) $(INC) -o $@ -c $<
	  $(POSTCOMPILE)

  $(NVCC_OBJECTS): $(BUILD_DIR)/%.o : $(SRC_ROOT)/%.cu $(BUILD_DIR)/%.d 
	  $(NVCC) $(NVCC_FLAGS) -Xcompiler "$(NVCC_XCOMPILE)" $(NVCC_INC) -M $< -o ${@:.o=.d} -odir $(@D)
	  $(NVCC) $(NVCC_FLAGS) -Xcompiler "$(NVCC_XCOMPILE)" $(NVCC_INC) -c $< -o $@
endif

clean:
	-rm -rf $(BUILD_DIR) $(BIN_DIR)

#include dependency
$(AUTODEPS): ;
.PRECIOUS: $(AUTODEPS)

-include $(AUTODEPS)

#utils
print-% : ; $(info $* is $(flavor $*) variable set to [$($*)]) @true
