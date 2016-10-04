#ifndef INCLUDE_BACKEND_GPU_BACKEND_H_
#define INCLUDE_BACKEND_GPU_BACKEND_H_

#include <string>
#include <vector>

#include "backend/backend.h"
#include "backend/gpu_tensor.h"

namespace blitz {

template<typename DType>
class Backend<GPUTensor, DType> {
 public:
  static void RectlinApplyFunc(
    const GPUTensor<DType>* input,
    const DType slope,
    GPUTensor<DType>* output);

  static void RectlinDerivativeFunc(
    const GPUTensor<DType>* input,
    const DType slope,
    GPUTensor<DType>* output);

  static void SoftmaxApplyFunc(
    const GPUTensor<DType>* input,
    GPUTensor<DType>* output);

  static void SoftmaxDerivativeFunc(
    const GPUTensor<DType>* input,
    GPUTensor<DType>* output);

  static DType SquareMeanApplyFunc(
    const GPUTensor<DType>* input,
    const GPUTensor<DType>* target);

  static void SquareMeanDerivativeFunc(
    const GPUTensor<DType>* input, const GPUTensor<DType>* target,
    GPUTensor<DType>* output);

  static DType AbsMeanApplyFunc(
    const GPUTensor<DType>* input,
    const GPUTensor<DType>* target);

  static void AbsMeanDerivativeFunc(
    const GPUTensor<DType>* input, const GPUTensor<DType>* target,
    GPUTensor<DType>* output);

  static void LogisticApplyFunc(
    const GPUTensor<DType>* input,
    GPUTensor<DType>* output);

  static void LogisticDerivativeFunc(
    const GPUTensor<DType>* input,
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

  static void BiasBackwardUpdateFunc(const GPUTensor<DType>* input,
    GPUTensor<DType>* update);

  static void BatchNormForwardFunc(
    const GPUTensor<DType>* input, const GPUTensor<DType>* gamma,
    const GPUTensor<DType>* beta, const DType epsilon,
    GPUTensor<DType>* input_var, GPUTensor<DType>* input_hat,
    GPUTensor<DType>* output);

  static void BatchNormBackwardFunc(
    const GPUTensor<DType>* backward_input,
    const GPUTensor<DType>* forward_input_hat,
    const GPUTensor<DType>* forward_input_var,
    const GPUTensor<DType>* gamma, const DType epsilon,
    GPUTensor<DType>* gamma_update, GPUTensor<DType>* beta_update,
    GPUTensor<DType>* output);

  static void GradientdescentFunc(
    const DType momentum_coef, const DType learning_rate,
    const DType decay, size_t batch_size,
    GPUTensor<DType>* filter,
    GPUTensor<DType>* gradient,
    GPUTensor<DType>* velocity);

  static void MatrixDotFunc(
    const GPUTensor<DType>* left, const GPUTensor<DType>* right,
    const bool transa, const bool transb,
    const DType alpha, const DType beta,
    GPUTensor<DType>* output, const string& kernel = "blas");

  static void MaximumFunc(
    const GPUTensor<DType>* left, const GPUTensor<DType>* right,
    GPUTensor<DType>* output);

  static void MaximumFunc(
    const GPUTensor<DType>* left,
    const DType right,
    GPUTensor<DType>* output);

  static void MinusFunc(
    const GPUTensor<DType>* left, const GPUTensor<DType>* right,
    GPUTensor<DType>* output);

  static void MinusFunc(
    const GPUTensor<DType>* left,
    const DType right,
    GPUTensor<DType>* output);

  static DType SumFunc(
    const GPUTensor<DType>* input);

  static void AddFunc(
    const GPUTensor<DType>* left, const GPUTensor<DType>* right,
    GPUTensor<DType>* output);

  static void MultiplyFunc(
    const GPUTensor<DType>* left, const GPUTensor<DType>* right,
    GPUTensor<DType>* output);

  static void MultiplyFunc(
    const GPUTensor<DType>* left, const DType right,
    GPUTensor<DType>* output);

  static void Convolution2DForwardFunc(
    const GPUTensor<DType>* input, const GPUTensor<DType>* filter,
    size_t padding_height, size_t padding_width,
    size_t stride_height, size_t stride_width,
    GPUTensor<DType>* unpack, GPUTensor<DType>* output,
    const string& kernel = "blas");

  static void Convolution2DBackwardFunc(
    const GPUTensor<DType>* output, const GPUTensor<DType>* filter,
    size_t padding_height, size_t padding_width,
    size_t stride_height, size_t stride_width,
    GPUTensor<DType>* pack, GPUTensor<DType>* input,
    const string& kernel = "blas");

  static void Convolution2DUpdateFunc(
    const GPUTensor<DType>* input, const GPUTensor<DType>* output,
    size_t padding_height, size_t padding_width,
    size_t stride_height, size_t stride_width,
    GPUTensor<DType>* unpack, GPUTensor<DType>* update,
    const string& kernel = "blas");

  static void MaxPooling2DForwardFunc(
    const GPUTensor<DType>* input,
    size_t filter_height, size_t filter_width,
    size_t stride_height, size_t stride_width,
    GPUTensor<size_t>* max_index, GPUTensor<DType>* output);

  static void MaxPooling2DBackwardFunc(
    const GPUTensor<DType>* output, const GPUTensor<size_t>* max_index,
    size_t filter_height, size_t filter_width,
    size_t stride_height, size_t stride_width,
    GPUTensor<DType>* input);

  static void MakeBinaryMaskFunc(const DType low, const DType high,
    const DType keep, GPUTensor<DType>* output);

  static void ConstantDistributionFunc(
    const DType val, GPUTensor<DType>* output);

  static void NormalDistributionFunc(
    const DType loc, const DType scale,
    GPUTensor<DType>* output);

  static void UniformDistributionFunc(
    const DType low, const DType high,
    GPUTensor<DType>* output);

  static void HostCopyToFunc(const DType* source, const size_t size,
    DType* target);

  static float EvaluateClassifyFunc(
    const GPUTensor<DType>* output, const GPUTensor<DType>* target);

  static float EvaluateRegressFunc(
    const GPUTensor<DType>* output, const GPUTensor<DType>* target);

  static void Unpack2DFunc(
    const DType* input, size_t channel,
    size_t input_height, size_t input_width,
    size_t filter_height, size_t filter_width,
    size_t output_height, size_t output_width,
    size_t padding_height, size_t padding_width,
    size_t stride_height, size_t stride_width,
    DType* unpack);

  static void Pack2DFunc(
    const DType* pack, size_t channel,
    size_t input_height, size_t input_width,
    size_t filter_height, size_t filter_width,
    size_t output_height, size_t output_width,
    size_t padding_height, size_t padding_width,
    size_t stride_height, size_t stride_width,
    DType* input);
};

}  // namespace blitz

#endif  // INCLUDE_BACKEND_GPU_BACKEND_H_
