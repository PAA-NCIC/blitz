#ifndef INCLUDE_BACKENDS_BACKEND_H_
#define INCLUDE_BACKENDS_BACKEND_H_

#include <vector>
#include <string>

#include "utils/common.h"

namespace blitz {

template<template <typename> class TensorType, typename DType>
class Backend {
 public:
  static void RectlinApplyFunc(
    const TensorType<DType>* input,
    const DType slope,
    TensorType<DType>* output);

  static void RectlinDerivativeFunc(
    const TensorType<DType>* input,
    const DType slope,
    TensorType<DType>* output);

  static void LogisticApplyFunc(
    const TensorType<DType>* input,
    TensorType<DType>* output);

  static void LogisticDerivativeFunc(
    const TensorType<DType>* input,
    TensorType<DType>* output);

  static void SoftmaxApplyFunc(
    const TensorType<DType>* input,
    TensorType<DType>* output);

  static void SoftmaxDerivativeFunc(
    const TensorType<DType>* input,
    TensorType<DType>* output);

  static DType SquareMeanApplyFunc(
    const TensorType<DType>* input,
    const TensorType<DType>* target);

  static void SquareMeanDerivativeFunc(
    const TensorType<DType>* input, const TensorType<DType>* target,
    TensorType<DType>* output);

  static DType AbsMeanApplyFunc(
    const TensorType<DType>* input,
    const TensorType<DType>* target);

  static void AbsMeanDerivativeFunc(
    const TensorType<DType>* input, const TensorType<DType>* target,
    TensorType<DType>* output);

  static DType CrossEntropyBinaryApplyFunc(
    const TensorType<DType>* input, const TensorType<DType>* target);

  static void CrossEntropyBinaryDerivativeFunc(
    const TensorType<DType>* input, const TensorType<DType>* target,
    TensorType<DType>* output);

  static DType CrossEntropyMultiApplyFunc(
    const TensorType<DType>* input, const TensorType<DType>* target);

  static void CrossEntropyMultiDerivativeFunc(
    const TensorType<DType>* input, const TensorType<DType>* target,
    TensorType<DType>* output);

  static void BiasForwardFunc(
    const TensorType<DType>* input, const TensorType<DType>* bias,
    TensorType<DType>* output);

  static void BiasBackwardUpdateFunc(const TensorType<DType>* input,
    TensorType<DType>* update);

  static void BatchNormForwardFunc(
    const TensorType<DType>* input, const TensorType<DType>* gamma,
    const TensorType<DType>* beta, const DType epsilon,
    TensorType<DType>* input_var, TensorType<DType>* input_hat,
    TensorType<DType>* output);

  static void BatchNormBackwardFunc(
    const TensorType<DType>* backward_input,
    const TensorType<DType>* forward_input_hat,
    const TensorType<DType>* forward_input_var,
    const TensorType<DType>* gamma, const DType epsilon,
    TensorType<DType>* gamma_update, TensorType<DType>* beta_update,
    TensorType<DType>* output);

  static void GradientdescentFunc(
    const DType momentum_coef, const DType learning_rate,
    const DType decay, size_t batch_size,
    TensorType<DType>* filter,
    TensorType<DType>* gradient,
    TensorType<DType>* velocity);

  static void MatrixDotFunc(
    const TensorType<DType>* left, const TensorType<DType>* right,
    const bool transa, const bool transb,
    const DType alpha, const DType beta,
    TensorType<DType>* output, const string& kernel = "blas");

  static void MaximumFunc(
    const TensorType<DType>* left, const TensorType<DType>* right,
    TensorType<DType>* output);

  static void MaximumFunc(
    const TensorType<DType>* left,
    const DType right,
    TensorType<DType>* output);

  static void MinusFunc(
    const TensorType<DType>* left, const TensorType<DType>* right,
    TensorType<DType>* output);

  static void MinusFunc(
    const TensorType<DType>* left,
    const DType right,
    TensorType<DType>* output);

  static DType SumFunc(
    const TensorType<DType>* input);

  static void AddFunc(
    const TensorType<DType>* left, const TensorType<DType>* right,
    TensorType<DType>* output);

  static void MultiplyFunc(
    const TensorType<DType>* left, const TensorType<DType>* right,
    TensorType<DType>* output);

  static void MultiplyFunc(
    const TensorType<DType>* left, const DType right,
    TensorType<DType>* output);

  static void Convolution2DForwardFunc(
    const TensorType<DType>* input, const TensorType<DType>* filter,
    size_t padding_height, size_t padding_width,
    size_t stride_height, size_t stride_width,
    TensorType<DType>* workspace,
    TensorType<DType>* output,
    const string& kernel = "blas");

  static void Convolution2DBackwardFunc(
    const TensorType<DType>* output, const TensorType<DType>* filter,
    size_t padding_height, size_t padding_width,
    size_t stride_height, size_t stride_width,
    TensorType<DType>* workspace,
    TensorType<DType>* input,
    const string& kernel = "blas");

  static void Convolution2DUpdateFunc(
    const TensorType<DType>* input, const TensorType<DType>* output,
    size_t padding_height, size_t padding_width,
    size_t stride_height, size_t stride_width,
    TensorType<DType>* workspace,
    TensorType<DType>* update,
    const string& kernel = "blas");

  static void MaxPooling2DForwardFunc(
    const TensorType<DType>* input,
    size_t filter_height, size_t filter_width,
    size_t stride_height, size_t stride_width,
    TensorType<size_t>* max_index, TensorType<DType>* output);

  static void MaxPooling2DBackwardFunc(
    const TensorType<DType>* output, const TensorType<size_t>* max_index,
    size_t filter_height, size_t filter_width,
    size_t stride_height, size_t stride_width,
    TensorType<DType>* input);

  static void MakeBinaryMaskFunc(const DType low, const DType high,
    const DType keep, TensorType<DType>* output);

  static void ConstantDistributionFunc(
    const DType val, TensorType<DType>* output);

  static void NormalDistributionFunc(
    const DType loc, const DType scale,
    TensorType<DType>* output);

  static void UniformDistributionFunc(
    const DType low, const DType high,
    TensorType<DType>* output);

  static void HostCopyToFunc(const DType* source, const size_t size,
    DType* target);

  static float EvaluateClassifyFunc(
    const TensorType<DType>* output, const TensorType<DType>* target);

  static float EvaluateRegressFunc(
    const TensorType<DType>* output, const TensorType<DType>* target);

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

  DISABLE_COPY_AND_ASSIGN(Backend);
};

}  // namespace blitz

#endif  // INCLUDE_BACKENDS_BACKEND_H_
