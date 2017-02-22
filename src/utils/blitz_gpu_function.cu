#include "utils/blitz_gpu_function.h"

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cublas.h>
#include <curand_kernel.h>

namespace blitz {

boost::scoped_ptr<cublasHandle_t> CuBlasHandle::instance_(0);
boost::once_flag CuBlasHandle::flag_ = BOOST_ONCE_INIT;

template<>
void BlitzGPUTrans(float* input, float* output, size_t M, size_t N) {
  float alpha = 1.0;
  float beta = 0.0;
  cublasSgeam(CuBlasHandle::GetInstance(), CUBLAS_OP_T, CUBLAS_OP_N,
    M, N, &alpha, input, N, &beta, input, M, output, M);
}

template<>
void BlitzGPUTrans(double* input, double* output, size_t M, size_t N) {
  double alpha = 1.0;
  double beta = 0.0;
  cublasDgeam(CuBlasHandle::GetInstance(), CUBLAS_OP_T, CUBLAS_OP_N,
    M, N, &alpha, input, N, &beta, input, M, output, M);
}

template<>
float BlitzGPUASum(const float* data, size_t N) {
  float sum = 0.0;
  cublasSasum_v2(CuBlasHandle::GetInstance(), N, data, 1, &sum);
  return sum;
}

template<>
double BlitzGPUASum(const double* data, size_t N) {
  double sum = 0.0;
  cublasDasum_v2(CuBlasHandle::GetInstance(), N, data, 1, &sum);
  return sum;
}

template<>
void BlitzGenerateNormal(curandGenerator_t* gen, float* data,
  float loc, float scale, size_t size) {
  curandGenerateNormal(*gen, data, size, loc, scale);
}

template<>
void BlitzGenerateNormal(curandGenerator_t* gen, double* data,
  double loc, double scale, size_t size) {
  curandGenerateNormalDouble(*gen, data, size, loc, scale);
}

template<>
void BlitzGenerateUniform(curandGenerator_t* gen, float* data, size_t size) {
  curandGenerateUniform(*gen, data, size);
}

template<>
void BlitzGenerateUniform(curandGenerator_t* gen, double* data, size_t size) {
  curandGenerateUniformDouble(*gen, data, size);
}

template<>
__global__ void GPURectlinApply(
  const float* input, float* output,
  float compare_value, float slope,
  size_t size) {
  BLITZ_CUDA_LOOP(i, size) {
    float greater = input[i] > compare_value ? input[i] : compare_value;
    float less = input[i] <= compare_value ?
      slope * input[i] : slope * compare_value;
    output[i] = greater + less;
  }
}

template<>
__global__ void GPURectlinDerivative(
  const float* input, float* output,
  float compare_value, float slope,
  size_t size) {
  BLITZ_CUDA_LOOP(i, size) {
    float greater = input[i] > compare_value ? 1.0 : 0.0;
    float less = input[i] <= compare_value ? slope : 0.0;
    output[i] = (greater + less) * output[i];
  }
}

template<>
__global__ void GPUSoftmaxApply(
  const float* input, float* output,
  size_t batch_size, size_t dim) {
  BLITZ_CUDA_LOOP(i, batch_size) {
    float sum = 0; 
    for (size_t j = 0; j < dim; ++j) {
      size_t index = i * dim + j;
      output[index] = exp(input[index]);
      sum += output[index];
    }
    for (size_t j = 0; j < dim; ++j) {
      output[i * dim + j] /= sum;
    }
  }
}

template<>
__global__ void GPULogisticApply(const float* input, float* output, size_t size) {
  BLITZ_CUDA_LOOP(i, size) {
    output[i] = 1 / (exp(-input[i]) + 1);
  }
}

template<>
__global__ void GPUCrossEntropyBinaryApply(
  const float* input, const float* target,
  float* sum, size_t size) {
  BLITZ_CUDA_LOOP(i, size) {
    float safe_input = BlitzGPUSafeLog(input[i]);
    float safe_inverse_input = BlitzGPUSafeLog(1 - input[i]);
    sum[i] += -safe_input * target[i] - safe_inverse_input
      * (1 - target[i]);
  }
}

template<>
__global__ void GPUCrossEntropyMultiApply(
  const float* input, const float* target,
  float* sum, size_t size) {
  BLITZ_CUDA_LOOP(i, size) {
    sum[i] = -BlitzGPUSafeLog(input[i]) * target[i];
  }
}

template<>
__global__ void GPUBiasApply(
  const float* input, const float* bias, float* output,
  size_t batch_size, size_t dim) {
  BLITZ_CUDA_LOOP(i, batch_size) {
    for (size_t j = 0; j < dim; ++j) {
      output[i * dim + j] = input[i * dim + j] + bias[j];
    }
  }
}

template<>
__global__ void GPUBiasDerivative(const float* input, float* update,
  size_t dim, size_t batch_size) {
  BLITZ_CUDA_LOOP(i, dim) {
    for (size_t j = 0; j < batch_size; ++j) {
      update[i] += input[j * dim + i];
    }
  }
}

template<>
__global__ void GPUGradientdescent(
  float* weight, float* gradient, float* velocity,
  float momentum_coef, float learning_rate,
  float decay, size_t batch_size, size_t size) {
  BLITZ_CUDA_LOOP(i, size) {
    gradient[i] /= batch_size;
    velocity[i] = velocity[i] * momentum_coef - learning_rate *
      (gradient[i] + decay * weight[i]);
    weight[i] = weight[i] + velocity[i];
  }
}

template<>
__global__ void GPUMakeBinaryMask(float* output, float keep, size_t size) {
  BLITZ_CUDA_LOOP(i, size) {
    if (output[i] < keep) {
      output[i] = float(1);
    } else {
      output[i] = float(0);
    }
  }
}

template<>
__global__ void GPUUniformTransform(float* output, float low, float high,
  size_t size) {
  BLITZ_CUDA_LOOP(i, size) {
    output[i] = low + (high - low) * output[i];
  }
}

template<>
__global__ void GPUEvaluateClass(
  const float* output, const float* target, float* correct,
  size_t dim, size_t size) {
  BLITZ_CUDA_LOOP(i, size) {
    float max = output[i * dim];
    size_t max_index = 0;
    for (size_t j = 1; j < dim; ++j) {
      if (max < output[i * dim + j]) {
        max_index = j;
        max = output[i * dim + j];
      }
    }
    if (target[i * dim + max_index] == (float)1.0) {
      correct[i] = 1.0f;
    }
  }
}

}  // namespace blitz

