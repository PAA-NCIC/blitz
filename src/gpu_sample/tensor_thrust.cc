#include <iostream>

#include <cuda.h>
#include <cuda_runtime_api.h>

#include "backend/backends.h"

using namespace blitz;

int main() {
  Shape shape(1);
  shape[0] = 128000;
  GPUTensor<float> tmp(shape);
  float all = Backend<GPUTensor, float>::SumFunc(&tmp);
  std::cout << "sum " << all << std::endl;
  return 0;
}
