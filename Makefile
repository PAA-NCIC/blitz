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

#lib
LIB := $(LIB_DIR)/libblitz.so

#compilers
OPTIMIZE_OPTIONS := -O3
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

OPENMP_OPTIONS := -fopenmp
CXXFLAGS := -Wall -Wno-unused-parameter -Wunknown-pragmas -Wunused-but-set-variable -fPIC $(OPENMP_OPTIONS) $(OPTIMIZE_OPTIONS) 
INC := -Iinclude/

#dynamic libraries
ifeq ($(BLITZ_LIB_ONLY), 1)
	LDFLAGS := -Wl,--no-as-needed -lglog -lboost_thread -lboost_date_time -lboost_system
else
	LDFLAGS := -Wl,--no-as-needed -lyaml-cpp -lhdf5 -lglog -lboost_thread -lboost_date_time -lboost_system
endif

ifeq ($(BLITZ_USE_GPU), 1)
	NVCC := nvcc
	NVCC_INC := -Iinclude/
	NVCC_XCOMPILE := -O3 -Wall -fopenmp -fPIC -DBLITZ_USE_GPU -Wunknown-pragmas
	NVCC_FLAGS := -O3 $(CUDA_ARCH) --use_fast_math -ccbin $(BLITZ_CC)
	CXXFLAGS += -DBLITZ_USE_GPU
endif

#only use static libray for libxsmm
ifeq ($(BLITZ_USE_MIC), 1)
	XSMM_LIB := ./third-party/libxsmm
	CXXFLAGS += -DBLITZ_USE_MIC
	LDFLAGS += -lxsmm
	INC += -I$(XSMM_LIB)/include
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

#blitz different modes
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

CXXFLAGS += -DBLITZ_ALIGNMENT_SIZE=$(BLITZ_ALIGNMENT_SIZE)

#blas
ifeq ($(BLAS), mkl)
	#MKL
	LDFLAGS += -lmkl_intel_lp64 -lmkl_core -lmkl_gnu_thread -lpthread -ldl
	CXXFLAGS += -DUSE_MKL
	MKL_DIR := /opt/intel/mkl
	BLAS_INCLUDE := $(MKL_DIR)/include
	BLAS_LIB := $(MKL_DIR)/lib/intel64
	INC += -I$(BLAS_INCLUDE)
	LIBRARY_DIR += -L$(BLAS_LIB)
else ifeq ($(BLAS), openblas)
	LDFLAGS += -lopenblas
else ifeq ($(BLAS), atlas)
	#ATLAS
	LDFLAGS += -lcblas -latlas
endif

#variables
ifeq ($(BLITZ_LIB_ONLY), 0)
ifeq ($(BLITZ_USE_MIC), 0)
	SRCS := $(shell find $(SRC_ROOT) -maxdepth 4 -name "*.cc" ! -name $(PROJECT).cc ! -path "*backup*" ! -path "*samples*" ! -path "*xsmm*" ! -path "*mic*")
else
	SRCS := $(shell find $(SRC_ROOT) -maxdepth 4 -name "*.cc" ! -name $(PROJECT).cc ! -path "*backup*" ! -path "*samples*")
endif
else
ifeq ($(BLITZ_USE_MIC), 0)
	SRCS := $(shell find $(SRC_ROOT)/backends $(SRC_ROOT)/utils $(SRC_ROOT)/kernels -maxdepth 4 -name "*.cc" ! -name $(PROJECT).cc ! -path "*backup*" ! -path "*samples*" ! -path "*xsmm*" ! -path "*mic*")
else
	SRCS := $(shell find $(SRC_ROOT)/backends $(SRC_ROOT)/utils $(SRC_ROOT)/kernels -maxdepth 4 -name "*.cc" ! -name $(PROJECT).cc ! -path "*backup*" ! -path "*samples*")
endif
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

#dirs
ifeq ($(BLITZ_USE_GPU), 1)
	ALL_OBJECTS_DIR := $(sort $(OBJECTS_DIR) $(NVCC_OBJECTS_DIR))
else
	ALL_OBJECTS_DIR := $(OBJECTS_DIR)
endif

#rules
#mkdir first
ifeq ($(BLITZ_LIB_ONLY), 0)
all: dirs bins objects libs

dirs: $(BIN_DIR) $(ALL_OBJECTS_DIR) $(LIB_DIR)

$(BIN_DIR):
	mkdir -p $@

bins: $(BINS)
else
all: dirs objects libs

dirs: $(ALL_OBJECTS_DIR) $(LIB_DIR)
endif

$(ALL_OBJECTS_DIR):
	mkdir -p $@

$(LIB_DIR):
	mkdir -p $@

libs: $(LIB)

ifeq ($(BLITZ_LIB_ONLY), 0)
$(BINS): $(BIN_DIR)/% : $(LIB) $(SRC_ROOT)/%.cc 
	$(BLITZ_CC) $(CXXFLAGS) $(INC) $(LIBRARY_DIR) $^ -o $@ $(LIB) 
endif

ifeq ($(BLITZ_USE_GPU), 1)
$(LIB): $(OBJECTS) $(NVCC_OBJECTS)
	$(BLITZ_CC) -shared $^ -o $@ $(LDFLAGS) 

objects: $(OBJECTS) $(NVCC_OBJECTS)
else
$(LIB): $(OBJECTS)
	$(BLITZ_CC) -shared $^ -o $@ $(LDFLAGS) 

objects: $(OBJECTS)
endif

$(OBJECTS): $(BUILD_DIR)/%.o : $(SRC_ROOT)/%.cc $(BUILD_DIR)/%.d 
	$(BLITZ_CC) $(CXXFLAGS) $(INC) -o $@ -c $< $(DEPFLAGS)
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
