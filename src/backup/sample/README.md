#API Samples
---

##1. How to use yaml-cpp?##

**Install**

https://github.com/jbeder/yaml-cpp

set *BUILD_SHARED_LIBS=ON*

**yaml-cpp configs in `~/.bashrc`**

        CPLUS_INCLUDE_PATH=/home/zkr/install/yaml-cpp/include:$CPLUS_INCLUDE_PATH
        LD_LIBRARY_PATH=/home/zkr/install/yaml-cpp/lib:$LD_LIBRARY_PATH
        LIBRARY_PATH=/home/zkr/install/yaml-cpp/lib:$LIBRARY_PATH

        export CPLUS_INCLUDE_PATH
        export LD_LIBRARY_PATH
        export LIBRARY_PATH

**Practical examples**

        cd blitz/src/samples
        make
        ./yaml
