#variables
.PHONY: clean bins

#project
BINS := convolution
CC := g++

bins: $(BINS)

#flags
LDFLAGS := -Wl,--no-as-needed -lglog -lblitz -lcudnn
CXXFLAGS := -Wall -Wno-unused-parameter -O3 -fPIC -fopenmp -mavx -DBLITZ_USE_GPU

$(BINS): % : %.cc
	$(CC) $(CXXFLAGS) $(INC) -o $@ $^ $(LDFLAGS) 

clean:
	-rm -rf $(BINS)

#utils
print-% : ; $(info $* is $(flavor $*) variable set to [$($*)]) @true
