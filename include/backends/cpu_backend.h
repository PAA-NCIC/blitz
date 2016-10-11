#ifndef INCLUDE_BACKENDS_CPU_BACKEND_H_
#define INCLUDE_BACKENDS_CPU_BACKEND_H_

#include <vector>
#include <string>

#include "backends/backend.h"
#include "backends/cpu_tensor.h"

namespace blitz {

// default general CPU
template<typename DType>
class Backend<CPUTensor, DType> {
 public:
  static void RectlinApplyFunc(
    const CPUTensor<DType>* input,
    const DType slope,
    CPUTensor<DType>* output);

  static void RectlinDerivativeFunc(
    const CPUTensor<DType>* input,
    const DType slope,
    CPUTensor<DType>* output);

  static void LogisticApplyFunc(
    const CPUTensor<DType>* input,
    CPUTensor<DType>* output);

  static void LogisticDerivativeFunc(
    const CPUTensor<DType>* input,
    CPUTensor<DType>* output);

  static void SoftmaxApplyFunc(
    const CPUTensor<DType>* input,
    CPUTensor<DType>* output);

  static void SoftmaxDerivativeFunc(
    const CPUTensor<DType>* input,
    CPUTensor<DType>* output);

  static DType SquareMeanApplyFunc(
    const CPUTensor<DType>* input,
    const CPUTensor<DType>* target);

  static void SquareMeanDerivativeFunc(
    const CPUTensor<DType>* input, const CPUTensor<DType>* target,
    CPUTensor<DType>* output);

  static DType AbsMeanApplyFunc(
    const CPUTensor<DType>* input,
    const CPUTensor<DType>* target);

  static void AbsMeanDerivativeFunc(
    const CPUTensor<DType>* input, const CPUTensor<DType>* target,
    CPUTensor<DType>* output);

  static DType CrossEntropyBinaryApplyFunc(
    const CPUTensor<DType>* input,
    const CPUTensor<DType>* target);

  static void CrossEntropyBinaryDerivativeFunc(
    const CPUTensor<DType>* input, const CPUTensor<DType>* target,
    CPUTensor<DType>* output);

  static DType CrossEntropyMultiApplyFunc(
    const CPUTensor<DType>* input, const CPUTensor<DType>* target);

  static void CrossEntropyMultiDerivativeFunc(
    const CPUTensor<DType>* input, const CPUTensor<DType>* target,
    CPUTensor<DType>* output);

  static void BiasForwardFunc(
    const CPUTensor<DType>* input, const CPUTensor<DType>* bias,
    CPUTensor<DType>* output);

  static void BiasBackwardUpdateFunc(const CPUTensor<DType>* input,
    CPUTensor<DType>* update);

  static void BatchNormForwardFunc(
    const CPUTensor<DType>* input, const CPUTensor<DType>* gamma,
    const CPUTensor<DType>* beta, const DType epsilon,
    CPUTensor<DType>* input_var, CPUTensor<DType>* input_hat,
    CPUTensor<DType>* output);

  static void BatchNormBackwardFunc(
    const CPUTensor<DType>* backward_input,
    const CPUTensor<DType>* forward_input_hat,
    const CPUTensor<DType>* forward_input_var,
    const CPUTensor<DType>* gamma, const DType epsilon,
    CPUTensor<DType>* gamma_update, CPUTensor<DType>* beta_update,
    CPUTensor<DType>* output);

  static void GradientdescentFunc(
    const DType momentum_coef, const DType learning_rate,
    const DType decay, size_t batch_size,
    CPUTensor<DType>* filter,
    CPUTensor<DType>* gradient,
    CPUTensor<DType>* velocity);

  static void MatrixDotFunc(
    const CPUTensor<DType>* left, const CPUTensor<DType>* right,
    const bool transa, const bool transb,
    const DType alpha, const DType beta,
    CPUTensor<DType>* output, const string& kernel = "blas");

  static void Transpose2DFunc(
    const CPUTensor<DType>* input, CPUTensor<DType>* output);

  static void MaximumFunc(
    const CPUTensor<DType>* left, const CPUTensor<DType>* right,
    CPUTensor<DType>* output);

  static void MaximumFunc(
    const CPUTensor<DType>* left,
    const DType right,
    CPUTensor<DType>* output);

  static void MinusFunc(
    const CPUTensor<DType>* left, const CPUTensor<DType>* right,
    CPUTensor<DType>* output);

  static void MinusFunc(
    const CPUTensor<DType>* left,
    const DType right,
    CPUTensor<DType>* output);

  static DType SumFunc(const CPUTensor<DType>* input);

  static void AddFunc(
    const CPUTensor<DType>* left, const CPUTensor<DType>* right,
    CPUTensor<DType>* output);

  static void MultiplyFunc(
    const CPUTensor<DType>* left, const CPUTensor<DType>* right,
    CPUTensor<DType>* output);

  static void MultiplyFunc(
    const CPUTensor<DType>* left, const DType right,
    CPUTensor<DType>* output);

  static void Convolution2DForwardFunc(
    const CPUTensor<DType>* input, const CPUTensor<DType>* filter,
    size_t padding_height, size_t padding_width,
    size_t stride_height, size_t stride_width,
    CPUTensor<DType>* unpack, CPUTensor<DType>* output,
    const string& kernel = "blas");

  static void Convolution2DBackwardFunc(
    const CPUTensor<DType>* output, const CPUTensor<DType>* filter,
    size_t padding_height, size_t padding_width,
    size_t stride_height, size_t stride_width,
    CPUTensor<DType>* pack, CPUTensor<DType>* input,
    const string& kernel = "blas");

  static void Convolution2DUpdateFunc(
    const CPUTensor<DType>* input, const CPUTensor<DType>* output,
    size_t padding_height, size_t padding_width,
    size_t stride_height, size_t stride_width,
    CPUTensor<DType>* unpack, CPUTensor<DType>* update,
    const string& kernel = "blas");

  static void MaxPooling2DForwardFunc(
    const CPUTensor<DType>* input,
    size_t filter_height, size_t filter_width,
    size_t stride_width, size_t stride_height,
    CPUTensor<size_t>* max_index, CPUTensor<DType>* output);

  static void MaxPooling2DBackwardFunc(
    const CPUTensor<DType>* output, const CPUTensor<size_t>* max_index,
    size_t filter_height, size_t filter_width,
    size_t stride_height, size_t stride_width,
    CPUTensor<DType>* input);

  static void MakeBinaryMaskFunc(const DType low, const DType high,
    const DType keep, CPUTensor<DType>* output);

  static void ConstantDistributionFunc(
    const DType val, CPUTensor<DType>* output);

  static void NormalDistributionFunc(
    const DType loc, const DType scale,
    CPUTensor<DType>* output);

  static void UniformDistributionFunc(
    const DType low, const DType high,
    CPUTensor<DType>* output);

  static void HostCopyToFunc(const DType* source, const size_t size,
    DType* target);

  static float EvaluateClassifyFunc(
    const CPUTensor<DType>* output, const CPUTensor<DType>* target);

  static float EvaluateRegressFunc(
    const CPUTensor<DType>* output, const CPUTensor<DType>* target);

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

#endif  // INCLUDE_BACKENDS_CPU_BACKEND_H_
