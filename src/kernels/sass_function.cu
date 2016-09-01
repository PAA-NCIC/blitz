#include "kernels/sass_function.h"

#include <cuda.h>

#include <string>

namespace blitz {

scoped_ptr<CudaLoadModule> CudaModule::instance_(0);
boost::once_flag CudaModule::flag_ = BOOST_ONCE_INIT;

template<>
void BlitzSassGemm(const bool transa, const bool transb,
  const int M, const int N, const int K,
  float* A, float* B, float* C, float alpha, float beta) {
  CUfunction function;
  int lda, ldb, ldc = N;

#ifdef BLITZ_PERFORMANCE  // only valid for a single thread
  float elapsed_time = 0.0f;
  CUevent event_start, event_stop;
  cuEventCreate(&event_start, CU_EVENT_BLOCKING_SYNC);
  cuEventCreate(&event_stop, CU_EVENT_BLOCKING_SYNC);
  cuEventRecord(event_start, NULL);
#endif

  // create kernel
  string kernel;
  if (transa == true && transb == false) {
    lda = M * 32;
    ldb = N * 32;
    if (M % 4 == 0 && N % 4 == 0) {
      kernel = "sgemm_tn_128x128_vec";
    } else {
      kernel = "sgemm_tn_128x128";
    }
  } else if (transa == false && transb == true) {
    lda = K;
    ldb = K;
    if (K % 16 == 0) {
      kernel = "sgemm_nt_128x128_vec";
    } else {
      kernel = "sgemm_nt_128x128";
    }
  } else if (transa == false && transb == false) {
    lda = K;
    ldb = N * 32;
    if (K % 16 == 0 && N % 4 == 0) {
      kernel = "sgemm_nn_128x128_vec";
    } else {
      kernel = "sgemm_nn_128x128";
    }
  } else {
    LOG(FATAL) << "Not support both matrice transport!";
  }

  // kernel call, asynrhonize
  function = CudaModule::GetFunction(kernel);

#ifdef BLITZ_PERFORMANCE
  cuEventRecord(event_stop, NULL);
  cuEventSynchronize(event_stop);
  cuEventElapsedTime(&elapsed_time, event_start, event_stop);
  LOG(INFO) << "Load kernel: " << kernel;
  LOG(INFO) << "Load kernel time: " << elapsed_time / 1000.0;
#endif  // BLITZ_PERFORMANCE

  void* params[] = {&A, &B, &C, &alpha, &beta, &lda, &ldb, &ldc,
    (void*)&M, (void*)&N, (void*)&K};

  // TODO(keren): multiple kernels
  int sizeA = 128, sizeB = 128;
  int gridA = M / sizeA + (M % sizeA != 0);
  int gridB = N / sizeB + (N % sizeB != 0);

  // TODO(keren): adjust number of threads
  int threads = 256;

#ifdef BLITZ_PERFORMANCE  // only valid for a single thread
  cuEventRecord(event_start, NULL);
#endif
  cuLaunchKernel(function, 1, gridA, gridB, threads, 1, 1, 0, 0, params, NULL);
#ifdef BLITZ_PERFORMANCE
  cuEventRecord(event_stop, NULL);
  cuEventSynchronize(event_stop);
  cuEventElapsedTime(&elapsed_time, event_start, event_stop);
  cuEventCreate(&event_stop, CU_EVENT_BLOCKING_SYNC);
  LOG(INFO) << "Compute time: " << elapsed_time / 1000.0;

  cuEventDestroy(event_start);
  cuEventDestroy(event_stop);
#endif  // BLITZ_PERFORMANCE
}

template<>
void BlitzSassGemm(const bool transa, const bool transb,
  const int M, const int N, const int K,
  double* A, double* B, double* C, double alpha, double beta) {
  LOG(FATAL) << "sass kernel dost not support double precision";
}

template<>
void BlitzSass2DConvolution(const int batch_size,
  const int input_channel, const int input_height,
  const int input_width, const int filter_height,
  const int filter_width, const int output_channel,
  const int output_height, const int output_width,
  const int stride_height, const int stride_width,
  float* input, float* output, float* filter,
  const string& phase) {
}

template<>
void BlitzSass2DConvolution(const int batch_size,
    const int input_channel, const int input_height,
  const int input_width, const int filter_height,
  const int filter_width, const int output_channel,
  const int output_height, const int output_width,
  const int stride_height, const int stride_width,
  double* input, double* output, double* filter,
  const string& phase) {
  LOG(FATAL) << "sass kernel dost not support double precision";
}

}  // namespace blitz

