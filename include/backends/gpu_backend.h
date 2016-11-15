#ifndef INCLUDE_BACKENDS_GPU_BACKEND_H_
#define INCLUDE_BACKENDS_GPU_BACKEND_H_

#include <string>
#include <vector>

#include "backends/backend.h"
#include "backends/gpu_tensor.h"

namespace blitz {

template<typename DType>
class Backend<GPUTensor, DType> {
 public:
  static void RectlinApplyFunc(
    const GPUTensor<DType>* input, GPUTensor<DType>* output,
    DType slope);

  static void RectlinDerivativeFunc(
    const GPUTensor<DType>* input, GPUTensor<DType>* output,
    DType slope);

  static void LogisticApplyFunc(
    const GPUTensor<DType>* input, GPUTensor<DType>* output);

  static void LogisticDerivativeFunc(
    const GPUTensor<DType>* input, GPUTensor<DType>* output);

  static void SoftmaxApplyFunc(
    const GPUTensor<DType>* input, GPUTensor<DType>* output);

  static void SoftmaxDerivativeFunc(
    const GPUTensor<DType>* input, GPUTensor<DType>* output);

  static DType SquareMeanApplyFunc(
    const GPUTensor<DType>* input, const GPUTensor<DType>* target);

  static void SquareMeanDerivativeFunc(
    const GPUTensor<DType>* input, const GPUTensor<DType>* target,
    GPUTensor<DType>* output);

  static DType AbsMeanApplyFunc(
    const GPUTensor<DType>* input, const GPUTensor<DType>* target);

  static void AbsMeanDerivativeFunc(
    const GPUTensor<DType>* input, const GPUTensor<DType>* target,
    GPUTensor<DType>* output);

  static DType CrossEntropyBinaryApplyFunc(
    const GPUTensor<DType>* input, const GPUTensor<DType>* target);

  static void CrossEntropyBinaryDerivativeFunc(
    const GPUTensor<DType>* input, const GPUTensor<DType>* target,
    GPUTensor<DType>* output);

  static DType CrossEntropyMultiApplyFunc(
    const GPUTensor<DType>* input, const GPUTensor<DType>* target);

  static void CrossEntropyMultiDerivativeFunc(
    const GPUTensor<DType>* input, const GPUTensor<DType>* target,
    GPUTensor<DType>* output);

  static void BiasForwardFunc(
    const GPUTensor<DType>* input, const GPUTensor<DType>* bias,
    GPUTensor<DType>* output);

  static void BiasBackwardUpdateFunc(
    const GPUTensor<DType>* input, GPUTensor<DType>* update);

  static void BatchNormForwardFunc(
    const GPUTensor<DType>* input,
    const GPUTensor<DType>* gamma,
    const GPUTensor<DType>* beta,
    GPUTensor<DType>* input_var,
    GPUTensor<DType>* input_hat,
    GPUTensor<DType>* output,
    DType epsilon);

  static void BatchNormBackwardFunc(
    const GPUTensor<DType>* backward_input,
    const GPUTensor<DType>* forward_input_hat,
    const GPUTensor<DType>* forward_input_var,
    const GPUTensor<DType>* gamma,
    GPUTensor<DType>* gamma_update,
    GPUTensor<DType>* beta_update,
    GPUTensor<DType>* output,
    DType epsilon);

  static void GradientdescentFunc(
    GPUTensor<DType>* filter,
    GPUTensor<DType>* gradient,
    GPUTensor<DType>* velocity,
    DType momentum_coef,
    DType learning_rate,
    DType decay,
    size_t batch_size);

  static void MatrixMultiplyFunc(
    const GPUTensor<DType>* left,
    const GPUTensor<DType>* right,
    GPUTensor<DType>* output, 
    bool transa,
    bool transb,
    DType alpha,
    DType beta,
    BLITZ_ALGORITHM algorithm = BLITZ_BLAS_GEMM);

  static void Transpose2DFunc(
    const GPUTensor<DType>* input, GPUTensor<DType>* output);

  static void MaximumFunc(
    const GPUTensor<DType>* left, const GPUTensor<DType>* right,
    GPUTensor<DType>* output);

  static void MinusFunc(
    const GPUTensor<DType>* left, const GPUTensor<DType>* right,
    GPUTensor<DType>* output);

  static DType SumFunc(const GPUTensor<DType>* input);

  static void AddFunc(
    const GPUTensor<DType>* left, const GPUTensor<DType>* right,
    GPUTensor<DType>* output);

  static void MultiplyFunc(
    const GPUTensor<DType>* left, const GPUTensor<DType>* right,
    GPUTensor<DType>* output);

  static void MultiplyFunc(
    const GPUTensor<DType>* left, GPUTensor<DType>* output,
    DType right);

  static void Convolution2DForwardFunc(
    const GPUTensor<DType>* input,
    const GPUTensor<DType>* filter,
    GPUTensor<DType>* output,
    GPUTensor<DType>* workspace,
    size_t padding_height, size_t padding_width,
    size_t stride_height, size_t stride_width,
    BLITZ_ALGORITHM algorithm = BLITZ_CONVOLUTION_BLAS_GEMM);

  static void Convolution2DBackwardFunc(
    const GPUTensor<DType>* output,
    const GPUTensor<DType>* filter,
    GPUTensor<DType>* input,
    GPUTensor<DType>* workspace,
    size_t padding_height, size_t padding_width,
    size_t stride_height, size_t stride_width,
		BLITZ_ALGORITHM algorithm = BLITZ_CONVOLUTION_BLAS_GEMM);

  static void Convolution2DUpdateFunc(
    const GPUTensor<DType>* input,
    const GPUTensor<DType>* output,
    GPUTensor<DType>* update,
    GPUTensor<DType>* workspace,
    size_t padding_height, size_t padding_width,
    size_t stride_height, size_t stride_width,
		BLITZ_ALGORITHM algorithm = BLITZ_CONVOLUTION_BLAS_GEMM);

  static void MaxPooling2DForwardFunc(
    const GPUTensor<DType>* input,
    GPUTensor<DType>* output,
    GPUTensor<size_t>* max_index,
    size_t filter_height, size_t filter_width,
    size_t stride_height, size_t stride_width);

  static void MaxPooling2DBackwardFunc(
    const GPUTensor<DType>* output, 
    GPUTensor<DType>* input,
    const GPUTensor<size_t>* max_index,
    size_t filter_height, size_t filter_width,
    size_t stride_height, size_t stride_width);

  static void MakeBinaryMaskFunc(
    GPUTensor<DType>* output,
    DType low,
    DType high,
    DType keep);

  static void ConstantDistributionFunc(GPUTensor<DType>* output, DType val);

  static void NormalDistributionFunc(GPUTensor<DType>* output, DType loc, DType scale);

  static void UniformDistributionFunc(GPUTensor<DType>* output, DType low, DType high);

  static void HostCopyToFunc(const DType* source, DType* target, size_t size);

  static float EvaluateClassifyFunc(
    const GPUTensor<DType>* output, const GPUTensor<DType>* target);

  static float EvaluateRegressFunc(
    const GPUTensor<DType>* output, const GPUTensor<DType>* target);

  static void Unpack2DFunc(
    const DType* input,
    DType* unpack,
    size_t channel,
    size_t input_height,
    size_t input_width,
    size_t filter_height,
    size_t filter_width,
    size_t output_height,
    size_t output_width,
    size_t padding_height,
    size_t padding_width,
    size_t stride_height,
    size_t stride_width);

  static void Pack2DFunc(
    const DType* pack,
    DType* input,
    size_t channel,
    size_t input_height,
    size_t input_width,
    size_t filter_height,
    size_t filter_width,
    size_t output_height,
    size_t output_width,
    size_t padding_height,
    size_t padding_width,
    size_t stride_height,
    size_t stride_width);
};

}  // namespace blitz

#endif  // INCLUDE_BACKENDS_GPU_BACKEND_H_
