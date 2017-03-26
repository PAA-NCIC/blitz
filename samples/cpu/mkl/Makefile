#variables
.PHONY: clean bins

#project
BINS := convolution
CC := g++

bins: $(BINS)

#flags
LDFLAGS := -Wl,--no-as-needed -lglog -lmkl_intel_lp64 -lmkl_core -lmkl_gnu_thread -lpthread -ldl
CXXFLAGS := -Wall -Wno-unused-parameter -O3 -fPIC -fopenmp -mavx -g

$(BINS): % : %.cc
	$(CC) $(CXXFLAGS) $(INC) -o $@ $^ $(LDFLAGS) 

clean:
	-rm -rf $(BINS)

#utils
print-% : ; $(info $* is $(flavor $*) variable set to [$($*)]) @true
