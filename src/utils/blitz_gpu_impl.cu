#include "utils/blitz_impl_function.h"

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cublas.h>
#include <curand_kernel.h>

#include "utils/blitz_gpu_function.h"
#include "backends/gpu_tensor.h"

namespace blitz {

namespace utils {

// small kernel
static __global__ void GPUUnpack1024Kernel(
  const float* I,
  float* U,
  size_t H, size_t W,
  size_t R, size_t S,
  size_t pad_h, size_t pad_w,
  size_t str_h, size_t str_w) {
  size_t p = threadIdx.x;
  size_t q = threadIdx.y;
  size_t c = blockIdx.x;
  size_t Q = blockDim.y;
  size_t C = gridDim.x;
  int H_offset = p * str_h - pad_h;
  int W_offset = q * str_w - pad_w;
  const float* I_slice = I + (c * H + H_offset) * W + W_offset;
  float* U_slice = U + (p * Q + q) * R * S * C + c * R * S;
  int h, w;
  for (size_t i = 0; i < R; ++i) {
    h = H_offset + i;
    if (h < 0 || h >= static_cast<int>(H)) {
      for (size_t j = 0; j < S; ++j) {
        *U_slice++ = 0;
      }
    } else {
      for (size_t j = 0; j < S; ++j) {
        w = W_offset + j;
        if (w < 0 || w >= static_cast<int>(W)) {
          *U_slice++ = 0;
        } else {
          *U_slice++ = I_slice[i * W + j];
        }
      }
    }
  }
}

// general kernel
static __global__ void GPUUnpackKernel(
  const float* I,
  float* U,
  size_t size,
  size_t C, size_t H, size_t W,
  size_t R, size_t S,
  size_t P, size_t Q,
  size_t pad_h, size_t pad_w,
  size_t str_h, size_t str_w) {
  BLITZ_CUDA_LOOP(index, size) {
    size_t k = index / Q;
    size_t p = k % P;
    size_t q = index % Q;
    size_t c = k / P;
    int H_offset = p * str_h - pad_h;
    int W_offset = q * str_w - pad_w;
    const float* I_slice = I + (c * H + H_offset) * W + W_offset;
    float* U_slice = U + (p * Q + q) * R * S * C + c * R * S;
    int h, w;
    for (size_t i = 0; i < R; ++i) {
      h = H_offset + i;
      if (h < 0 || h >= static_cast<int>(H)) {
        for (size_t j = 0; j < S; ++j) {
          *U_slice++ = 0;
        }
      } else {
        for (size_t j = 0; j < S; ++j) {
          w = W_offset + j;
          if (w < 0 || w >= static_cast<int>(W)) {
            *U_slice++ = 0;
          } else {
            *U_slice++ = I_slice[i * W + j];
          }
        }
      }
    }
  }
}

template<>
void UnpackImpl<GPUTensor, float, BLITZ_BUFFER_NCHW>(
  const float* I,
  float* U,
  size_t C, size_t H, size_t W,
  size_t R, size_t S,
  size_t P, size_t Q,
  size_t pad_h, size_t pad_w,
  size_t str_h, size_t str_w) {
  if (C <= 64 && P * Q <= 256) {
    dim3 thread_per_block(P, Q);
    GPUUnpack1024Kernel<<<C, thread_per_block>>>(
      I, U,
      H, W,
      R, S,
      pad_h, pad_w,
      str_h, str_w);
  } else {
    size_t size = C * P * Q;
    GPUUnpackKernel<<<BlitzGPUGetBlocks(size),
      BLITZ_NUM_GPU_THREADS>>>(
      I, U,
      size,
      C, H, W,
      R, S,
      P, Q,
      pad_h, pad_w,
      str_h, str_w);
  }
}

template<>
void UnpackImpl<GPUTensor, float, BLITZ_BUFFER_NHWC>(
  const float* I,
  float* U,
  size_t C, size_t H, size_t W,
  size_t R, size_t S,
  size_t P, size_t Q,
  size_t pad_h, size_t pad_w,
  size_t str_h, size_t str_w) {
  LOG(FATAL) << "GPU unpack NHWC not implemented!";
}

// small kernel
static __global__ void GPUPack1024Kernel(
  const float* U,
  float* I,
  size_t R, size_t S,
  size_t P, size_t Q,
  size_t pad_h, size_t pad_w,
  size_t str_h, size_t str_w) {
  size_t h = threadIdx.x;
  size_t w = threadIdx.y;
  size_t c = blockIdx.x;
  size_t H_pad = h + pad_h;
  size_t W_pad = w + pad_w;
  size_t H = blockDim.x;
  size_t W = blockDim.y;
  size_t C = gridDim.x;
  size_t U_w =  R * S * C;
  size_t U_hstart = H_pad < R ?  0 : (H_pad - R) / str_h + 1;
  size_t U_hend = min(H_pad / str_h + 1, P);
  size_t U_wstart = W_pad < S ?  0 : (W_pad - S) / str_w + 1;
  size_t U_wend = min(W_pad / str_w + 1, Q);
  const float *U_slice = U + S * R * c;
  float *I_slice = I + c * H * W + h * W + w;
  float sum = 0.0;
  size_t r, s;
  for (size_t i = U_hstart; i < U_hend; ++i) {
    for (size_t j = U_wstart; j < U_wend; ++j) {
      r = (H_pad - i * str_h);
      s = (W_pad - j * str_w);
      sum += U_slice[(i * Q + j) * U_w + r * S + s];
    }
  }
  *(I_slice) = sum;
}

// general kernel
static __global__ void GPUPackKernel(
  const float* U,
  float* I,
  size_t size,
  size_t C, size_t H, size_t W,
  size_t R, size_t S,
  size_t P, size_t Q,
  size_t pad_h, size_t pad_w,
  size_t str_h, size_t str_w) {
  BLITZ_CUDA_LOOP(index, size) {
    size_t C_hoffset = index / W;
    size_t h = C_hoffset % H;
    size_t w = index % W;
    size_t c = C_hoffset % C;
    size_t H_pad = h + pad_h;
    size_t W_pad = w + pad_w;
    size_t U_w =  R * S * C;
    size_t U_hstart = H_pad < R ?  0 : (H_pad - R) / str_h + 1;
    size_t U_hend = min(H_pad / str_h + 1, P);
    size_t U_wstart = W_pad < S ?  0 : (W_pad - S) / str_w + 1;
    size_t U_wend = min(W_pad / str_w + 1, Q);
    const float *U_slice = U + S * R * c;
    float* I_slice = I + c * H * W + h * W + w;
    float sum = 0.0;
    size_t r, s;
    for (size_t i = U_hstart; i < U_hend; ++i) {
      for (size_t j = U_wstart; j < U_wend; ++j) {
        r = (H_pad - i * str_h);
        s = (W_pad - j * str_w);
        sum += U_slice[(i * Q + j) * U_w + r * S + s];
      }
    }
    *(I_slice) = sum;
  }
}

template<>
void PackImpl<GPUTensor, float, BLITZ_BUFFER_NCHW>(
  const float* U,
  float* I,
  size_t C, size_t H, size_t W,
  size_t R, size_t S,
  size_t P, size_t Q,
  size_t pad_h, size_t pad_w,
  size_t str_h, size_t str_w) {
  if (C <= 64 && H * W <= 256) {
    dim3 thread_per_block(H, W);
    GPUPack1024Kernel<<<C, thread_per_block>>>(
      U, I,
      R, S,
      P, Q,
      pad_h, pad_w,
      str_h, str_w);
  } else {
    size_t size = C * H * W;
    GPUPackKernel<<<BlitzGPUGetBlocks(size),
      BLITZ_NUM_GPU_THREADS>>>(
      U, I,
      size,
      C, H, W,
      R, S,
      P, Q,
      pad_h, pad_w,
      str_h, str_w);
  }
}

template<>
void PackImpl<GPUTensor, float, BLITZ_BUFFER_NHWC>(
  const float* U,
  float* I,
  size_t C, size_t H, size_t W,
  size_t R, size_t S,
  size_t P, size_t Q,
  size_t pad_h, size_t pad_w,
  size_t str_h, size_t str_w) {
  LOG(FATAL) << "GPU pack NHWC not implemented!";
}

static __global__ void GPUMaxPoolingForward(
  const float* I,
  float* O,
  size_t* max_index, 
  size_t size,
  size_t C, size_t H, size_t W,
  size_t P, size_t Q,
  size_t R, size_t S,
  size_t str_h, size_t str_w) {
  BLITZ_CUDA_LOOP(index, size) {
    size_t q = index % Q;
    size_t p = (index / Q) % P;
    size_t c = (index / (Q * P)) % C;
    size_t n = index / (Q * P * C);
    size_t h_start = p * str_h;
    size_t w_start = q * str_w;
    size_t h_end = h_start + R;
    size_t w_end = w_start + S;
    size_t max_idx = h_start * W + w_start;
    const float* I_slice = I + (n * C + c) * H * W;
    for (size_t i = h_start; i < h_end; ++i) {
      for (size_t j = w_start; j < w_end; ++j) {
        if (I_slice[i * W + j] > I_slice[max_idx]) {
          max_idx = i * W + j;
        }
      }
    }
    O[index] = I_slice[max_idx];
    max_index[index] = max_idx;
  }
}

template<>
void MaxPoolingForwardImpl<GPUTensor, float, BLITZ_BUFFER_NCHW>(
  const float* I,
  float* O,
  size_t* max_index,
  size_t N,
  size_t C, size_t H, size_t W,
  size_t K, size_t P, size_t Q,
  size_t R, size_t S,
  size_t str_h, size_t str_w) {
  GPUMaxPoolingForward
    <<<BlitzGPUGetBlocks(N * K * P * Q), BLITZ_NUM_GPU_THREADS>>>(
    I, O, max_index,
    N * K * P * Q,
    C, H, W,
    P, Q,
    R, S,
    str_h, str_w);
}


static __global__ void GPUMaxPoolingBackward(
  const float* O,
  float* I,
  const size_t* max_index,
  size_t size,
  size_t C, size_t H, size_t W,
  size_t P, size_t Q) {
  BLITZ_CUDA_LOOP(i, size) {
    size_t c = (i / (Q * P)) % C;
    size_t n = i / (Q * P * C);
    float* I_slice = I + (n * C + c) * H * W;
    I_slice[max_index[i]] = O[i];
  }
}

template<>
void MaxPoolingBackwardImpl<GPUTensor, float, BLITZ_BUFFER_NCHW>(
  const float* O,
  float* I,
  const size_t* max_index,
  size_t N,
  size_t C, size_t H, size_t W,
  size_t K, size_t P, size_t Q) {
  GPUMaxPoolingBackward
    <<<BlitzGPUGetBlocks(N * K * P * Q), BLITZ_NUM_GPU_THREADS>>>(
    O, I, max_index,
    N * K * P * Q,
    C, H, W,
    P, Q);
}

template<>
void BlitzGemm<GPUTensor, float>(
  float* A, float* B, float* C,
  bool transa, bool transb,
  float alpha, float beta,
  size_t M, size_t N, size_t K) {
  cublasOperation_t TransA = transa ? CUBLAS_OP_T : CUBLAS_OP_N;
  size_t lda = transa ? M : K;
  cublasOperation_t TransB = transb ? CUBLAS_OP_T : CUBLAS_OP_N;
  size_t ldb = transb ? K : N;
  cublasSgemm_v2(CuBlasHandle::GetInstance(),
    TransB, TransA,
    N, M, K,
    &alpha,
    B, ldb,
    A, lda,
    &beta,
    C, N);
}

template<>
void BlitzGemm<GPUTensor, double>(
  double* A, double* B, double* C,
  bool transa, bool transb,
  double alpha, double beta,
  size_t M, size_t N, size_t K) {
  cublasOperation_t TransA = transa ? CUBLAS_OP_T : CUBLAS_OP_N;
  size_t lda = transa ? M : K;
  cublasOperation_t TransB = transb ? CUBLAS_OP_T : CUBLAS_OP_N;
  size_t ldb = transb ? K : N;
  cublasDgemm_v2(CuBlasHandle::GetInstance(),
    TransB, TransA,
    N, M, K,
    &alpha,
    B, ldb,
    A, lda,
    &beta,
    C, N);
}

}  // namespace utils

}  // namespace blitz
