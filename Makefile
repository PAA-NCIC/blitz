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
LIB_DIR := lib

#libs
LIBS := $(LIB_DIR)/libblitz.a

#compilers
OPTIMIZE_OPTIONS := -O1
#avx types
ifeq ($(BLITZ_AVX), 512)
	OPTIMIZE_OPTIONS += -DBLITZ_AVX_WIDTH=64 -xMIC-AVX512
else ifeq ($(BLITZ_AVX), 3)
	OPTIMIZE_OPTIONS += -DBLITZ_AVX_WIDTH=64 -xCORE-AVX512
else ifeq ($(BLITZ_AVX), 2)
	OPTIMIZE_OPTIONS += -DBLITZ_AVX_WIDTH=32 -march=core-avx2
else
	OPTIMIZE_OPTIONS += -DBLITZ_AVX_WIDTH=32 -mavx
endif

ifeq ($(BLITZ_ICC), icc)
	OPENMP_OPTIONS := -qopenmp
else
	OPENMP_OPTIONS := -fopenmp
endif
CXXFLAGS := -g -Wall -Wno-unused-parameter -Wunknown-pragmas -fPIC $(OPENMP_OPTIONS) $(OPTIMIZE_OPTIONS) 
INC := -Iinclude/

#libraries
LDFLAGS := -Wl,--no-as-needed -lyaml-cpp -lhdf5 -lglog -lboost_chrono -lboost_thread -lboost_date_time -lboost_system

ifeq ($(BLITZ_USE_GPU), 1)
	NVCC := nvcc
	NVCC_INC := -Iinclude/
	NVCC_XCOMPILE := -O3 -Wall -fopenmp -fPIC -DBLITZ_USE_GPU -Wunknown-pragmas
	NVCC_FLAGS := -O3 $(CUDA_ARCH) --use_fast_math -ccbin $(BLITZ_CC)
	CXXFLAGS += -DBLITZ_USE_GPU
endif

ifeq ($(BLITZ_USE_MIC), 1)
	INC += -I/home/scy/software/libxsmm/include
	CXXFLAGS += -DBLITZ_USE_MIC
	LDFLAGS += -lxsmm
	XSMM_LIB := $(LIBRARY_DIR)/xsmm
	LIBRARY_DIR += -L$(XSMM_LIB)
endif

#dependency
DEPDIR := $(BUILD_DIR)
$(shell mkdir -p $(DEPDIR) >/dev/null)
DEPFLAGS = -MT $@ -MMD -MP -MF $(DEPDIR)/$*.Td
POSTCOMPILE = mv -f $(DEPDIR)/$*.Td $(DEPDIR)/$*.d

ifeq ($(BLITZ_USE_GPU), 1)
	LDFLAGS += -lcudart -lcuda -lcublas -lcudnn -lcurand
endif

#blitz
ifeq ($(BLITZ_MODE), release)
	CXXFLAGS += -DBLITZ_RELEASE
	NVCC_XCOMPILE += -DBLITZ_RELEASE
else ifeq ($(BLITZ_MODE), performance)
	CXXFLAGS += -DBLITZ_PERFORMANCE -g
	NVCC_XCOMPILE += -DBLITZ_PERFORMANCE -g
else ifeq ($(BLITZ_MODE), DEVELOP)
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

#variables
ifeq ($(BLITZ_USE_MIC), 0)
	SRCS := $(shell find $(SRC_ROOT) -maxdepth 4 -name "*.cc" ! -name $(PROJECT).cc ! -path "*backup*" ! -path "*samples*" ! -path "*xsmm*" ! -path "*mic*")
else
	SRCS := $(shell find $(SRC_ROOT) -maxdepth 4 -name "*.cc" ! -name $(PROJECT).cc ! -path "*backup*" ! -path "*samples*")
endif

OBJECTS := $(addprefix $(BUILD_DIR)/, $(patsubst %.cc, %.o, $(SRCS:$(SRC_ROOT)/%=%)))
OBJECTS_DIR := $(sort $(addprefix $(BUILD_DIR)/, $(dir $(SRCS:$(SRC_ROOT)/%=%))))

BINS := $(BIN_DIR)/$(PROJECT)

AUTODEPS:= $(patsubst %.o, %.d, $(OBJECTS)) $(patsubst %.o, %.d, $(NVCC_OBJECTS))

ifeq ($(BLITZ_USE_GPU), 1)
	NVCC_SRCS := $(shell find $(SRC_ROOT) -maxdepth 4 -name "*.cu" ! -name $(PROJECT).cc ! -path "*backup*" ! -path "*samples*")
	NVCC_OBJECTS := $(addprefix $(BUILD_DIR)/, $(patsubst %.cu, %.o, $(NVCC_SRCS:$(SRC_ROOT)/%=%)))
	NVCC_OBJECTS_DIR := $(sort $(addprefix $(BUILD_DIR)/, $(dir $(NVCC_SRCS:$(SRC_ROOT)/%=%))))
	AUTODEPS:= $(AUTODEPS) $(patsubst %.o, %.d, $(NVCC_OBJECTS))
endif

#rules
#mkdir first
all: dirs bins objects libs

#dirs
ALL_BINS_DIR := $(BIN_DIR)
ifeq ($(BLITZ_USE_GPU), 1)
	ALL_OBJECTS_DIR := $(sort $(OBJECTS_DIR) $(NVCC_OBJECTS_DIR))
else
	ALL_OBJECTS_DIR := $(OBJECTS_DIR)
endif

dirs: $(ALL_BINS_DIR) $(ALL_OBJECTS_DIR) $(LIB_DIR)

$(ALL_BINS_DIR):
	mkdir -p $@

$(ALL_OBJECTS_DIR):
	mkdir -p $@

$(LIB_DIR):
	mkdir -p $@

bins: $(BINS)

libs: $(LIBS)

$(BINS): $(BIN_DIR)/% : $(LIBS) $(SRC_ROOT)/%.cc 
	$(BLITZ_CC) $(CXXFLAGS) $(INC) $(LIBRARY_DIR) $^ $(LIBS) $(LDFLAGS) -o $@

ifeq ($(BLITZ_USE_GPU), 1)
  $(LIBS): $(OBJECTS) $(NVCC_OBJECTS)
	ar rcs $@ $^
else
  $(LIBS): $(OBJECTS)
	ar rcs $@ $^
endif

ifeq ($(BLITZ_USE_GPU), 1)
  objects: $(OBJECTS) $(NVCC_OBJECTS)
else
  objects: $(OBJECTS)
endif

$(OBJECTS): $(BUILD_DIR)/%.o : $(SRC_ROOT)/%.cc $(BUILD_DIR)/%.d 
	$(BLITZ_CC) $(DEPFLAGS) $(CXXFLAGS) $(INC) -o $@ -c $<
	$(POSTCOMPILE)

ifeq ($(BLITZ_USE_GPU), 1)
$(NVCC_OBJECTS): $(BUILD_DIR)/%.o : $(SRC_ROOT)/%.cu $(BUILD_DIR)/%.d 
	$(NVCC) $(NVCC_FLAGS) -Xcompiler "$(NVCC_XCOMPILE)" $(NVCC_INC) -M $< -o ${@:.o=.d} -odir $(@D)
	$(NVCC) $(NVCC_FLAGS) -Xcompiler "$(NVCC_XCOMPILE)" $(NVCC_INC) -c $< -o $@
endif

clean:
	-rm -rf $(BUILD_DIR) $(BIN_DIR) $(LIB_DIR)

#include dependency
$(AUTODEPS): ;
.PRECIOUS: $(AUTODEPS)

-include $(AUTODEPS)

#utils
print-% : ; $(info $* is $(flavor $*) variable set to [$($*)]) @true
