#include "kernels/sass_function.h"

#include <cuda.h>

#include <string>

#include "utils/blitz_math_function.h"
#include "utils/blitz_gpu_function.h"
#include "utils/blitz_cpu_function.h"

namespace blitz {

scoped_ptr<CubinLoadModule> CubinModule::instance_(0);
boost::once_flag CubinModule::flag_ = BOOST_ONCE_INIT;

template<>
void BlitzSassGemm(
  const float* A,
  const float* B,
  float* C,
  bool transa, bool transb,
  float alpha, float beta,
  size_t M, size_t N, size_t K) {
  CUfunction function;
  size_t lda, ldb, ldc = N;

#ifdef BLITZ_PERFORMANCE
  float elapsed_time = 0.0f;
  CUevent event_start, event_stop;
  cuEventCreate(&event_start, CU_EVENT_BLOCKING_SYNC);
  cuEventCreate(&event_stop, CU_EVENT_BLOCKING_SYNC);
  cuEventRecord(event_start, NULL);
#endif  // BLITZ_PERFORMANCE
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
    if (K % 4 == 0) {
      kernel = "sgemm_nt_128x128_vec";
    } else {
      kernel = "sgemm_nt_128x128";
    }
  } else if (transa == false && transb == false) {
    lda = K;
    ldb = N * 32;
    if (K % 4 == 0 && N % 4 == 0) {
      kernel = "sgemm_nn_128x128_vec";
    } else {
      kernel = "sgemm_nn_128x128";
    }
  } else {
    LOG(FATAL) << "Not support both matrice transport!";
  }

  // kernel call, asynrhonize
  function = CubinModule::GetFunction(kernel);

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
  size_t sizeA = 128, sizeB = 128;
  size_t gridA = M / sizeA + (M % sizeA != 0);
  size_t gridB = N / sizeB + (N % sizeB != 0);
  // TODO(keren): adjust number of threads
  size_t threads = 256;

  // lanuch kernel
  cuLaunchKernel(function, 1, gridA, gridB, threads, 1, 1, 0, 0, params, NULL);
}

template<>
void BlitzSassGemm(
  const double* A,
  const double* B,
  double* C,
  bool transa, bool transb,
  double alpha, double beta,
  size_t M, size_t N, size_t K) {
  LOG(FATAL) << "sass kernel dost not support double precision";
}

template<>
void BlitzSassConvolution2D(
  float* I,
  float* O,
  float* F,
  size_t N,
  size_t C, size_t H, size_t W,
  size_t R, size_t S,
  size_t K, size_t P, size_t Q,
  size_t str_h, size_t str_w,
  size_t pad_h, size_t pad_w,
  const string& phase) {
  float alpha = 1.0f;
  size_t D = 1, M = 1, T = 1;
  size_t str_d = 1;
  size_t pad_d = 0;
  size_t WN, HW, DHW, HWN, DHWN;
  size_t RS, RST, KRST, CRST;
  size_t PQ, QN, MPQ, PQN, MPQN;
  size_t magic_HW, shift_HW;
  size_t magic_W, shift_W;
  size_t magic_RST, shift_RST;
  size_t magic_RS, shift_RS;
  size_t magic_S, shift_S;
  size_t magic_PQ, shift_PQ;
  size_t magic_Q, shift_Q;
  size_t magic_PQu, shift_PQu;
  size_t magic_Qu, shift_Qu;
  size_t magic_str_w, shift_str_w;
  size_t magic_str_h, shift_str_h;
  size_t magic_str_d, shift_str_d;
  size_t grid_P = 1;
  size_t grid_Q = 1;
  size_t grid_PQ = grid_P * grid_Q;
  size_t grid_PQM = grid_PQ * M;
  size_t CRST32, MPQN32;
  // input
  WN = W * N;
  HW = H * W;
  DHW = D * HW;
  HWN = H * WN;
  DHWN = HWN;
  // filter
  RS = R * S;
  RST = RS;
  KRST = K * RST;
  CRST = C * RST;
  // output
  QN = Q * N;
  PQ = P * Q;
  PQN = P * QN;
  MPQ = PQ;
  MPQN = PQN;
  // special bprop
  CRST32 = 32 * CRST;
  MPQN32 = 32 * MPQN;
  // magic numbers
  blitz_magic32(DHW, HW, magic_HW, shift_HW);
  blitz_magic32(HW, W, magic_W, shift_W);
  blitz_magic32(CRST, RST, magic_RST, shift_RST);
  blitz_magic32(RST + 32, RS, magic_RS, shift_RS);
  blitz_magic32(RS + 32, S, magic_S, shift_S);
  blitz_magic32(MPQ, PQ, magic_PQ, shift_PQ);
  blitz_magic32(PQ, Q, magic_Q, shift_Q);
  blitz_magic32(grid_PQM, grid_PQ, magic_PQu, shift_PQu);
  blitz_magic32(grid_PQ, grid_Q, magic_Qu, shift_Qu);
  blitz_magic32(W + S - pad_w - 2, str_w, magic_str_w, shift_str_w);
  blitz_magic32(H + R - pad_h - 2, str_h, magic_str_h, shift_str_h);
  blitz_magic32(D + T - pad_d - 2, str_d, magic_str_d, shift_str_d);
  // test param set up TODO(keren): erase
  float *test_param;
  // arguments
  size_t gridX, gridY, gridZ;
  CUresult result;
  CUfunction function;
  string kernel_name;
  if (phase == "forward") {
    void *args[37] = {
      &test_param, &O, &I, &F, &alpha,
      &N, &K, &D, &H, &W, &WN, &HWN, &DHWN,
      &C, &KRST, &RST,
      &RS, &magic_RS, &shift_RS,
      &S, &magic_S, &shift_S,
      &pad_d, &pad_h, &pad_w,
      &str_d, &str_h, &str_w,
      &Q, &PQ, &QN, &PQN, &MPQN,
      &magic_Q, &shift_Q,
      &magic_PQ, &shift_PQ};
		if (K <= 64 && N <= 64) {
			gridX = MPQ;
			gridY = K / 64 + (K % 64 != 0);
			gridZ = N / 64 + (N % 64 != 0);
			kernel_name = "sconv_fprop_K64_N64";
			// TODO(keren): tune kernels in future
			function = CubinModule::GetFunction(kernel_name);
			result = cuLaunchKernel(function, gridX, gridY, gridZ,
				64, 1, 1, 64 * 8 * 4 + RST * 4 * 2 + 8, 0, args, NULL);
		} else {
			gridX = MPQ;
			gridY = K / 128 + (K % 128 != 0);
			gridZ = N / 128 + (N % 128 != 0);
			kernel_name = "sconv_fprop_K128_N128";
			// TODO(keren): tune kernels in future
			function = CubinModule::GetFunction(kernel_name);
			result = cuLaunchKernel(function, gridX, gridY, gridZ,
				256, 1, 1, 128 * 8 * 4 + RST * 4 * 2 + 8, 0, args, NULL);
		}
		if (result != CUDA_SUCCESS) {
			LOG(FATAL) << "Launch kernel: " << kernel_name << " error!";
		}
  } else if (phase == "backward") {
    if (C % 64 == 0) {  // C64
      void *args[45] = {
        &test_param, &I, &O, &F, &alpha,
        &N, &C, &M, &P, &Q, &QN, &PQN, &MPQN,
        &K, &CRST, &RST,
        &RS, &magic_RS, &shift_RS,
        &S, &magic_S, &shift_S,
        &pad_d, &pad_h, &pad_w,
        &str_d, &str_h, &str_w,
        &W, &HW, &WN, &HWN, &DHWN,
        &magic_W, &shift_W,
        &magic_HW, &shift_HW,
        &R, &T,
        &magic_str_w, &shift_str_w,
        &magic_str_h, &shift_str_h,
        &magic_str_d, &shift_str_d};
      gridX = DHW;
      gridY = C / 64 + (C % 64 != 0);
      gridZ = N / 64 + (N % 64 != 0);
      kernel_name = "sconv_bprop_C64_N64";
      function = CubinModule::GetFunction(kernel_name);
      result = cuLaunchKernel(function, gridX, gridY, gridZ,
        64, 1, 1, 0, 0, args, NULL);
      if (result != CUDA_SUCCESS) {
        LOG(FATAL) << "Launch kernel: " << kernel_name << " error!";
      }
    } else {  // C1
      void *args[41] = {
        &test_param, &I, &O, &F, &alpha,
        &N, &K, &D, &H, &W, &WN, &HWN, &DHWN,
        &C, &CRST,
        &RST, &magic_RST, &shift_RST,
        &RS, &magic_RS, &shift_RS,
        &S, &magic_S, &shift_S,
        &pad_d, &pad_h, &pad_w,
        &str_d, &str_h, &str_w,
        &Q, &PQ, &QN, &PQN, &MPQN,
        &magic_Q, &shift_Q,
        &magic_PQ, &shift_PQ,
        &CRST32, &MPQN32};
      gridX = MPQ;
      gridY = CRST / 32 + (CRST % 32 != 0);
      gridZ = N / 64 + (N % 64 != 0);
      kernel_name = "sconv_bprop_C1_N64";
      function = CubinModule::GetFunction(kernel_name);
      result = cuLaunchKernel(function, gridX, gridY, gridZ,
        32, 1, 1, 0, 0, args, NULL);
      if (result != CUDA_SUCCESS) {
        LOG(FATAL) << "Launch kernel: " << kernel_name << " error!";
      }
    }
  } else if (phase == "update") {
    void *args[43] = {
      &test_param, &F, &I, &O, &alpha,
      &N, &K, &D, &H, &W, &WN, &HWN, &DHWN,
      &C, &CRST,
      &RST, &magic_RST, &shift_RST,
      &RS, &magic_RS, &shift_RS,
      &S, &magic_S, &shift_S,
      &pad_d, &pad_h, &pad_w,
      &str_d, &str_h, &str_w,
      &P, &Q, &PQ, &QN, &PQN, &MPQN,
      &magic_Qu, &shift_Qu,
      &magic_PQu, &shift_PQu,
      &grid_P, &grid_Q, &grid_PQ};
    gridX = grid_PQM;
    if ((K <= 64 && K % 128 != 0) || Q > 56) {
      gridY = CRST / 128 + (CRST % 128 != 0);
      gridZ = K / 64 + (K % 64 != 0);
      kernel_name = "sconv_update_C128_K64";
      function = CubinModule::GetFunction(kernel_name);
      result = cuLaunchKernel(function, gridX, gridY, gridZ,
        128, 1, 1, 0, 0, args, NULL);
    } else {
      gridY = CRST / 128 + (CRST % 128 != 0);
      gridZ = K / 128 + (K % 128 != 0);
      kernel_name = "sconv_update_C128_K128";
      function = CubinModule::GetFunction(kernel_name);
      result = cuLaunchKernel(function, gridX, gridY, gridZ,
        256, 1, 1, 0, 0, args, NULL);
    }
    if (result != CUDA_SUCCESS) {
      LOG(FATAL) << "Launch kernel: " << kernel_name << " error!";
    }
  }
}

template<>
void BlitzSassConvolution2D(
  double* I,
  double* O,
  double* F,
  size_t N,
  size_t C, size_t H, size_t W,
  size_t R, size_t S,
  size_t K, size_t P, size_t Q,
  size_t str_h, size_t str_w,
  size_t pad_h, size_t pad_w,
  const string& phase) { 
  LOG(FATAL) << "sass kernel dost not support double precision";
}

// shuffle
// K * C * T * R * S
// to
// K * T * R * S * C
template<typename DType>
__global__ void GPUFilterShuffle(
  const DType* input, DType* output,
  size_t TRSC, size_t TRS,
  size_t RSC, size_t SC,
  size_t C, size_t K,
  size_t RS, size_t magic_RS, size_t shift_RS,
  size_t S, size_t magic_S, size_t shift_S) {
  // C * K
  __shared__ DType tile[32][33];
  size_t tx  = threadIdx.x;
  size_t ty  = threadIdx.y;
  size_t bk  = blockIdx.x;
  size_t bc  = blockIdx.y;
  size_t trs = blockIdx.z;
  // t = trs % rs
  // r = rs % s
  // s = rs - r * s
  size_t t = magic_RS * trs;
  t >>= shift_RS;
  size_t rs = trs - t * RS;
  size_t r = magic_S * rs;
  r >>= shift_S;
  size_t s = rs - r * S;
  size_t k = bk * 32 + tx;
  size_t c = bc * 32 + ty;
  for (size_t i = 0; i < 32; i += 8) {
    size_t ci = c + i;
    if (ci < C && k < K)
      tile[ty + i][tx] = input[k * TRSC + ci * TRS + t * RS + r * S + s];
  }
  __syncthreads();
  k = bk * 32 + ty;
  c = bc * 32 + tx;
  for (size_t i = 0; i < 32; i += 8) {
    size_t ki = k + i;
    if (ki < K && c < C)
      output[ki * TRSC + t * RSC + r * SC + s * C + c] = tile[tx][ty + i];
  }
}

template<>
void BlitzFilter2DShuffle(
  const float* input,
  float* output,
  size_t K, size_t C,
  size_t R, size_t S) {
  size_t T = 1;
  size_t TRSC, RSC, SC;
  size_t RST, RS;
  size_t magic_RS, shift_RS;
  size_t magic_S, shift_S;
  // output
  SC = S * C;
  RSC = R * SC;
  TRSC = T * RSC;
  // filter
  RS = R * S; 
  RST = T * RS;
  blitz_magic32(RST + 32, RS, magic_RS, shift_RS);
  blitz_magic32(RS + 32, S, magic_S, shift_S);
  const size_t gridX = K / 32 + (K % 32 != 0);
  const size_t gridY = C / 32 + (C % 32 != 0);
  dim3 grid_dim(gridX, gridY, RST);
  dim3 block_dim(32, 8, 1);
  GPUFilterShuffle<<<grid_dim, block_dim>>>(
    input, output,
    TRSC, RST, RSC, SC,
    C, K,
    RS, magic_RS, shift_RS,
    S, magic_S, shift_S);
}

template<>
void BlitzFilter2DShuffle(
  const double* input,
  double* output,
  size_t K, size_t C,
  size_t R, size_t S) {
  LOG(FATAL) << "sass kernel dost not support double precision";
}

}  // namespace blitz

