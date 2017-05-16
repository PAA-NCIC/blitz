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
ifeq ($(AVX), 512)
	ifeq ($(CC), icc)
	  OPTIMIZE_OPTIONS += -DBLITZ_AVX512 -DBLITZ_AVX_WIDTH=64 -xMIC-AVX512
  else
	  OPTIMIZE_OPTIONS += -DBLITZ_AVX512 -DBLITZ_AVX_WIDTH=64 -mavx512 -mfma
	endif
else ifeq ($(AVX), 3)
	ifeq ($(CC), icc)
	  OPTIMIZE_OPTIONS += -DBLITZ_AVX3 -DBLITZ_AVX_WIDTH=64 -xCORE-AVX512
  else
	  OPTIMIZE_OPTIONS += -DBLITZ_AVX3 -DBLITZ_AVX_WIDTH=64 -mavx512 -mfma
  endif 
else ifeq ($(AVX), 2)
	ifeq ($(CC), icc)
	  OPTIMIZE_OPTIONS += -DBLITZ_AVX2 -DBLITZ_AVX_WIDTH=32 -xCORE-AVX2
  else
	  OPTIMIZE_OPTIONS += -DBLITZ_AVX2 -DBLITZ_AVX_WIDTH=32 -mavx2 -mfma
	endif
else ifeq ($(AVX), 1)
	OPTIMIZE_OPTIONS += -DBLITZ_AVX -DBLITZ_AVX_WIDTH=32 -mavx
else ifeq ($(AVX), 0)
	OPTIMIZE_OPTIONS += -DBLITZ_SSE -DBLITZ_AVX_WIDTH=32 -msse
endif

OPENMP_OPTIONS := -fopenmp
CXXFLAGS := -Wall -Wno-unused-parameter -Wunknown-pragmas -Wunused-but-set-variable -fPIC $(OPENMP_OPTIONS) $(OPTIMIZE_OPTIONS) 
INC := -Iinclude/

#dynamic libraries
ifeq ($(LIB_ONLY), 1)
	LDFLAGS := -Wl,--no-as-needed -lglog -lboost_thread -lboost_date_time -lboost_system
else
	LDFLAGS := -Wl,--no-as-needed -lyaml-cpp -lhdf5_serial -lglog -lboost_thread -lboost_date_time -lboost_system
endif

ifeq ($(USE_GPU), 1)
	NVCC := nvcc
	NVCC_INC := -Iinclude/
	NVCC_XCOMPILE := -O3 -Wall -fopenmp -fPIC -DBLITZ_USE_GPU -Wunknown-pragmas
	NVCC_FLAGS := -O3 -arch $(CUDA_ARCH) --use_fast_math -ccbin $(CC)
	CXXFLAGS += -DBLITZ_USE_GPU
endif

#dependency
DEPDIR := $(BUILD_DIR)
$(shell mkdir -p $(DEPDIR) >/dev/null)
DEPFLAGS = -MT $@ -MMD -MP -MF $(DEPDIR)/$*.Td
POSTCOMPILE = mv -f $(DEPDIR)/$*.Td $(DEPDIR)/$*.d

ifeq ($(USE_GPU), 1)
	LDFLAGS += -lcudart -lcuda -lcublas -lcurand
endif

ifeq ($(BLITZ_USE_CUDNN), 1)
	LDFLAGS += -lcudnn
endif

#blitz different modes
ifeq ($(MODE), release)
	CXXFLAGS += -DBLITZ_RELEASE
	NVCC_XCOMPILE += -DBLITZ_RELEASE
else ifeq ($(MODE), performance)
	CXXFLAGS += -DBLITZ_PERFORMANCE -g
	NVCC_XCOMPILE += -DBLITZ_PERFORMANCE -g
else ifeq ($(MODE), develop)
	CXXFLAGS += -DBLITZ_DEVELOP -g 
	NVCC_XCOMPILE += -DBLITZ_DEVELOP -g 
endif

CXXFLAGS += -DBLITZ_ALIGNMENT_SIZE=$(ALIGNMENT_SIZE)

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
ifeq ($(LIB_ONLY), 0)
	SRCS := $(shell find $(SRC_ROOT) -maxdepth 4 -name "*.cc" ! -name $(PROJECT).cc ! -path "*backup*" ! -path "*samples*")
else
	SRCS := $(shell find $(SRC_ROOT)/backends $(SRC_ROOT)/utils $(SRC_ROOT)/kernels -maxdepth 4 -name "*.cc" ! -name $(PROJECT).cc ! -path "*backup*" ! -path "*samples*")
endif

OBJECTS := $(addprefix $(BUILD_DIR)/, $(patsubst %.cc, %.o, $(SRCS:$(SRC_ROOT)/%=%)))
OBJECTS_DIR := $(sort $(addprefix $(BUILD_DIR)/, $(dir $(SRCS:$(SRC_ROOT)/%=%))))

BINS := $(BIN_DIR)/$(PROJECT)

AUTODEPS:= $(patsubst %.o, %.d, $(OBJECTS)) $(patsubst %.o, %.d, $(NVCC_OBJECTS))

ifeq ($(USE_GPU), 1)
	NVCC_SRCS := $(shell find $(SRC_ROOT) -maxdepth 4 -name "*.cu" ! -name $(PROJECT).cc ! -path "*backup*" ! -path "*samples*")
	NVCC_OBJECTS := $(addprefix $(BUILD_DIR)/, $(patsubst %.cu, %.o, $(NVCC_SRCS:$(SRC_ROOT)/%=%)))
	NVCC_OBJECTS_DIR := $(sort $(addprefix $(BUILD_DIR)/, $(dir $(NVCC_SRCS:$(SRC_ROOT)/%=%))))
	AUTODEPS:= $(AUTODEPS) $(patsubst %.o, %.d, $(NVCC_OBJECTS))
endif

#dirs
ifeq ($(USE_GPU), 1)
	ALL_OBJECTS_DIR := $(sort $(OBJECTS_DIR) $(NVCC_OBJECTS_DIR))
else
	ALL_OBJECTS_DIR := $(OBJECTS_DIR)
endif

#rules
#mkdir first
ifeq ($(LIB_ONLY), 0)
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

ifeq ($(LIB_ONLY), 0)
$(BINS): $(BIN_DIR)/% : $(SRC_ROOT)/%.cc $(LIB)
	$(CC) $(CXXFLAGS) $(INC) $(LIBRARY_DIR) $< -o $@ $(LDFLAGS) -lblitz
endif

ifeq ($(USE_GPU), 1)
$(LIB): $(OBJECTS) $(NVCC_OBJECTS)
	$(CC) -shared $^ -o $@ $(LDFLAGS) 

objects: $(OBJECTS) $(NVCC_OBJECTS)
else
$(LIB): $(OBJECTS)
	$(CC) -shared $^ -o $@ $(LDFLAGS) 

objects: $(OBJECTS)
endif

$(OBJECTS): $(BUILD_DIR)/%.o : $(SRC_ROOT)/%.cc $(BUILD_DIR)/%.d 
	$(CC) $(CXXFLAGS) $(INC) -o $@ -c $< $(DEPFLAGS)
	$(POSTCOMPILE)

ifeq ($(USE_GPU), 1)
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
