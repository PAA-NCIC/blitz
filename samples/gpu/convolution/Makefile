#variables
.PHONY: clean bins

#project
BINS := convolution shuffle
CC := nvcc

bins: $(BINS)

#flags
LDFLAGS := -lglog -lblitz
CXXFLAGS := -Wall -Wno-unused-parameter -O3 -fPIC -fopenmp -mavx -DBLITZ_USE_GPU

$(BINS): % : %.cu
	$(CC) -Xcompiler "$(CXXFLAGS)" $(INC) -o $@ $^ $(LDFLAGS) 

clean:
	-rm -rf $(BINS)

#utils
print-% : ; $(info $* is $(flavor $*) variable set to [$($*)]) @true
