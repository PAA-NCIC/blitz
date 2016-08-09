#include <iostream>

#include <unistd.h>
#include <cuda.h>
#include <cuda_runtime_api.h>

#include "backend/backends.h"

using namespace blitz;

int main() {
  const int iter = 200000;
  const unsigned int microseconds = 100;
  Shape shape(1);
  shape[0] = 100000;

  for (int i = 0; i < iter; ++i) {
    GPUTensor<float> t(shape);
    usleep(microseconds);
  }
  return 0;
}
