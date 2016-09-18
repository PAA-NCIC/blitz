#include <iostream>
#include "util/common.h"
#include "sys/time.h"
#include "backend/backends.h"

using namespace blitz;
const int ITER = 8 * 13 * 3;
//const int ITER = 1;

double gemm(const int dim_left, const int dim_right, const int dim_common)
{
  Shape left_shape(2);
  left_shape[0] = dim_left;
  left_shape[1] = dim_common;
  std::vector<shared_ptr<CPUTensor<float> > > left_vec;
  left_vec.resize(ITER + 1);

  for (int i = 0; i < ITER; ++i) {
    left_vec[i] = make_shared<CPUTensor<float> >(left_shape);
    Backend<CPUTensor, float>::NormalDistributionFunc(0, 1, left_vec[i].get());
  }

  Shape right_shape(2);
  right_shape[0] = dim_common;
  right_shape[1] = dim_right;
  CPUTensor<float> right(right_shape);
  Backend<CPUTensor, float>::NormalDistributionFunc(0, 1, &right);

  Shape output_shape(2);
  output_shape[0] = dim_left;
  output_shape[1] = dim_right;
  CPUTensor<float> output(output_shape);

  timeval t1, t2;
  double elapsed_time = 0.0f;
  gettimeofday(&t1, NULL);
  //std::chrono::time_point<std::chrono::system_clock> start, end;
  //std::chrono::duration<double> time = std::chrono::duration<double>::zero();
  //start = std::chrono::system_clock::now();

  for (int i = 0; i < ITER; ++i)
    Backend<CPUTensor, float>::MatrixDotFunc(left_vec[i].get(), &right, false, false, 1, 0, &output);

  //end = std::chrono::system_clock::now();
  //time = end - start;
  //return time.count();
  gettimeofday(&t2, NULL);
  elapsed_time = (t2.tv_sec - t1.tv_sec) * 1000.0; 
  elapsed_time += (t2.tv_usec - t1.tv_usec) / 1000.0;
  return elapsed_time / 1000.0;
}


int main() {
  std::cout << "start" << std::endl;
  int dim_left, dim_right, dim_common;
  long computations = 0;
  double interval;

  dim_left = 3;
  dim_right = 384;
  dim_common = 192;
  computations = 2 * dim_left * dim_right * dim_common;
  interval = gemm(dim_left, dim_right, dim_common);
  std::cout << "origin forward first conv: " << interval << std::endl;
  std::cout << "flops: " << computations * ITER / interval << std::endl;

  std::cout << "end" << std::endl;
  return 0;
}
  
