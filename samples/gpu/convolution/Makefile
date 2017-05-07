#variables
.PHONY: clean bins

#project
BINS := convolution shuffle
CC := g++

bins: $(BINS)

#flags
LDFLAGS := -Wl,--no-as-needed -lglog -lblitz -lcuda -lcudart
CXXFLAGS := -Wall -Wno-unused-parameter -O3 -fPIC -fopenmp -mavx -DBLITZ_USE_GPU

$(BINS): % : %.cc
	$(CC) $(CXXFLAGS) $(INC) -o $@ $^ $(LDFLAGS) 

clean:
	-rm -rf $(BINS)

#utils
print-% : ; $(info $* is $(flavor $*) variable set to [$($*)]) @true
