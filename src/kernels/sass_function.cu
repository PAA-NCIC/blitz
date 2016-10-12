#include "kernels/sass_function.h"

#include <cuda.h>

#include <string>

#include "utils/blitz_math_function.h"

namespace blitz {

scoped_ptr<CubinLoadModule> CubinModule::instance_(0);
boost::once_flag CubinModule::flag_ = BOOST_ONCE_INIT;

template<>
void BlitzSassGemm(
  bool transa, bool transb,
  int M, int N, int K,
  const float* A,
  const float* B,
  float* C,
  float alpha,
  float beta) {
  CUfunction function;
  int lda, ldb, ldc = N;

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
  int sizeA = 128, sizeB = 128;
  int gridA = M / sizeA + (M % sizeA != 0);
  int gridB = N / sizeB + (N % sizeB != 0);
  // TODO(keren): adjust number of threads
  int threads = 256;

  // lanuch kernel
  cuLaunchKernel(function, 1, gridA, gridB, threads, 1, 1, 0, 0, params, NULL);
}

template<>
void BlitzSassGemm(
  bool transa, bool transb,
  int M, int N, int K,
  const double* A,
  const double* B,
  double* C,
  double alpha,
  double beta) {
  LOG(FATAL) << "sass kernel dost not support double precision";
}

template<>
void BlitzSassConvolution2D(
  const string& phase, 
  int N,
  int C, int H, int W,
  int R, int S,
  int K, int P, int Q,
  int str_h, int str_w,
  int pad_h, int pad_w,
  float* I,
  float* O,
  float* F) {
  float alpha = 1.0f;
  unsigned int D = 1, M = 1, T = 1;
  unsigned int str_d = 1;
  unsigned int pad_d = 0;
  unsigned int WN, HW, DHW, HWN, DHWN;
  unsigned int RS, RST, KRST, CRST;
  unsigned int PQ, QN, MPQ, PQN, MPQN;
  unsigned int magic_HW, shift_HW;
  unsigned int magic_W, shift_W;
  unsigned int magic_RST, shift_RST;
  unsigned int magic_RS, shift_RS;
  unsigned int magic_S, shift_S;
  unsigned int magic_PQ, shift_PQ;
  unsigned int magic_Q, shift_Q;
  unsigned int magic_PQu, shift_PQu;
  unsigned int magic_Qu, shift_Qu;
  unsigned int magic_str_w, shift_str_w;
  unsigned int magic_str_h, shift_str_h;
  unsigned int magic_str_d, shift_str_d;
  unsigned int grid_P = 1;
  unsigned int grid_Q = 1;
  unsigned int grid_PQ = grid_P * grid_Q;
  unsigned int grid_PQM = grid_PQ * M;
  unsigned int CRST32, MPQN32;
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
  // test param set up
  float *test_param;
#ifdef BLITZ_DEVELOP
  cudaError_t cuda_error;
  cuda_error = cudaMalloc((void**)&test_param, sizeof(float) * 1024);
  cudaMemset(test_param, 0, sizeof(float) * 1024);
#endif
  // arguments
  unsigned int gridX, gridY, gridZ;
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
    #ifdef BLITZ_DEVELOP
    LOG(INFO) << "N: " << N;
    LOG(INFO) << "K: " << K;
    LOG(INFO) << "D: " << D;
    LOG(INFO) << "H: " << H;
    LOG(INFO) << "W: " << W;
    LOG(INFO) << "WN: " << WN;
    LOG(INFO) << "HWN: " << HWN;
    LOG(INFO) << "C: " << C;
    LOG(INFO) << "KRST: " << KRST;
    LOG(INFO) << "RST: " << RST;
    LOG(INFO) << "magic_RS: " << magic_RS << ", shift_RS: " << shift_RS;
    LOG(INFO) << "magic_S: " << magic_S << ", shift_S: " << shift_S;
    LOG(INFO) << "pad_d: " << pad_d;
    LOG(INFO) << "pad_w: " << pad_w;
    LOG(INFO) << "pad_h: " << pad_h;
    LOG(INFO) << "str_d: " << str_d;
    LOG(INFO) << "str_w: " << str_w;
    LOG(INFO) << "str_h: " << str_h;
    LOG(INFO) << "Q: " << Q;
    LOG(INFO) << "PQ: " << PQ;
    LOG(INFO) << "QN: " << QN;
    LOG(INFO) << "PQN: " << PQN;
    LOG(INFO) << "MPQN: " << MPQN;
    LOG(INFO) << "magic_Q: " << magic_Q << ", shift_Q: " << shift_Q;
    LOG(INFO) << "magic_PQ: " << magic_PQ << ", shift_PQ: " << shift_PQ;
    #endif  // BLITZ_DEVELOP
    gridX = MPQ;
    gridY = K / 64 + (K % 64 != 0);
    gridZ = N / 64 + (N % 64 != 0);
    kernel_name = "sconv_fprop_K64_N64";
    // TODO(keren): tune kernels in future
    function = CubinModule::GetFunction(kernel_name);
    result = cuLaunchKernel(function, gridX, gridY, gridZ,
      64, 1, 1, 64 * 8 * 4 + RST * 4 * 2 + 8, 0, args, NULL);
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
      #ifdef BLITZ_DEVELOP
      LOG(INFO) << "N: " << N;
      LOG(INFO) << "C: " << C;
      LOG(INFO) << "M: " << M;
      LOG(INFO) << "P: " << P;
      LOG(INFO) << "Q: " << Q;
      LOG(INFO) << "QN: " << QN;
      LOG(INFO) << "PQN: " << PQN;
      LOG(INFO) << "MPQN: " << MPQN;
      LOG(INFO) << "K: " << K;
      LOG(INFO) << "CRST: " << CRST;
      LOG(INFO) << "RST: " << RST;
      LOG(INFO) << "RS: " << RS;
      LOG(INFO) << "magic_RS: " << magic_RS << ", shift_RS: " << shift_RS;
      LOG(INFO) << "S: " << S;
      LOG(INFO) << "magic_S: " << magic_S << ", shift_S: " << shift_S;
      LOG(INFO) << "pad_d: " << pad_d;
      LOG(INFO) << "pad_w: " << pad_w;
      LOG(INFO) << "pad_h: " << pad_h;
      LOG(INFO) << "str_d: " << str_d;
      LOG(INFO) << "str_w: " << str_w;
      LOG(INFO) << "str_h: " << str_h;
      LOG(INFO) << "W: " << W;
      LOG(INFO) << "HW: " << HW;
      LOG(INFO) << "WN: " << WN;
      LOG(INFO) << "HWN: " << HWN;
      LOG(INFO) << "DHWN: " << DHWN;
      LOG(INFO) << "magic_W: " << magic_W << ", shift_W: " << shift_W;
      LOG(INFO) << "magic_HW " << magic_HW << ", shift_HW: " << shift_HW;
      #endif  // BLITZ_DEVELOP
      gridX = DHW;
      gridY = C / 64 + (C % 64 != 0);
      gridZ = N / 64 + (N % 64 != 0);
      kernel_name = "sconv_bprop_C64_N64";
      function = CubinModule::GetFunction(kernel_name);
      //result = cuLaunchKernel(function, gridX, gridY, gridZ,
      //  64, 1, 1, 0, 0, args, NULL);
      //if (result != CUDA_SUCCESS) {
      //  LOG(FATAL) << "Launch kernel: " << kernel_name << " error!";
      //}
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
        &CRST32,
        &MPQN32};
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
    gridY = CRST / 128 + (CRST % 128 != 0);
    gridZ = K / 128 + (K % 128 != 0);
    kernel_name = "sconv_update_C128_K128";
    function = CubinModule::GetFunction(kernel_name);
    result = cuLaunchKernel(function, gridX, gridY, gridZ,
      256, 1, 1, 0, 0, args, NULL);
    if (result != CUDA_SUCCESS) {
      LOG(FATAL) << "Launch kernel: " << kernel_name << " error!";
    }
  }
}

template<>
void BlitzSassConvolution2D(
  const string& phase, 
  int N,
  int C, int H, int W,
  int R, int S,
  int K, int P, int Q,
  int str_h, int str_w,
  int pad_h, int pad_w,
  double* I,
  double* O,
  double* F) {
  LOG(FATAL) << "sass kernel dost not support double precision";
}

// shuffle
// K * C * T * R * S
// to
// K * T * R * S * C
template<typename DType>
__global__ void GPUFilterShuffle(
  const DType* input, DType* output,
  unsigned int TRSC, unsigned int TRS,
  unsigned int RSC, unsigned int SC,
  unsigned int C, unsigned int K,
  unsigned int RS, unsigned int magic_RS, unsigned int shift_RS,
  unsigned int S, unsigned int magic_S, unsigned int shift_S) {
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
  int K, int C,
  int R, int S,
  const float* input,
  float* output) {
  unsigned int T = 1;
  unsigned int TRSC, RSC, SC;
  unsigned int RST, RS;
  unsigned int magic_RS, shift_RS;
  unsigned int magic_S, shift_S;
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
  int K, int C,
  int R, int S,
  const double* input,
  double* output) {
}

}  // namespace blitz

