#ifndef INCLUDE_BACKENDS_BACKEND_H_
#define INCLUDE_BACKENDS_BACKEND_H_

#include <vector>
#include <string>

#include "utils/common.h"
#include "utils/blitz_algorithm_function.h"
#include "utils/blitz_shape_function.h"
#include "utils/blitz_impl_function.h"

namespace blitz {

template<template <typename> class TensorType, typename DType>
class Backend {
 public:
  static void RectlinApplyFunc(
    const TensorType<DType>* input, TensorType<DType>* output,
    DType slope);

  static void RectlinDerivativeFunc(
    const TensorType<DType>* input, TensorType<DType>* output,
    DType slope);

  static void LogisticApplyFunc(
    const TensorType<DType>* input, TensorType<DType>* output);

  static void LogisticDerivativeFunc(
    const TensorType<DType>* input, TensorType<DType>* output);

  static void SoftmaxApplyFunc(
    const TensorType<DType>* input, TensorType<DType>* output);

  static void SoftmaxDerivativeFunc(
    const TensorType<DType>* input, TensorType<DType>* output);

  static DType SquareMeanApplyFunc(
    const TensorType<DType>* input, const TensorType<DType>* target);

  static void SquareMeanDerivativeFunc(
    const TensorType<DType>* input, const TensorType<DType>* target,
    TensorType<DType>* output);

  static DType AbsMeanApplyFunc(
    const TensorType<DType>* input, const TensorType<DType>* target);

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

  static void BiasBackwardUpdateFunc(
    const TensorType<DType>* input, TensorType<DType>* update);

  static void BatchNormForwardFunc(
    const TensorType<DType>* input,
    const TensorType<DType>* gamma,
    const TensorType<DType>* beta,
    TensorType<DType>* input_var,
    TensorType<DType>* input_hat,
    TensorType<DType>* output,
    DType epsilon);

  static void BatchNormBackwardFunc(
    const TensorType<DType>* backward_input,
    const TensorType<DType>* forward_input_hat,
    const TensorType<DType>* forward_input_var,
    const TensorType<DType>* gamma,
    TensorType<DType>* gamma_update,
    TensorType<DType>* beta_update,
    TensorType<DType>* output,
    DType epsilon);

  static void GradientdescentFunc(
    TensorType<DType>* filter,
    TensorType<DType>* gradient,
    TensorType<DType>* velocity,
    DType momentum_coef,
    DType learning_rate,
    DType decay,
    size_t batch_size);

  static void MatrixMultiplyFunc(
    const TensorType<DType>* left,
    const TensorType<DType>* right,
    TensorType<DType>* output, 
    bool transa,
    bool transb,
    DType alpha,
    DType beta,
    BLITZ_ALGORITHM algorithm = BLITZ_BLAS_GEMM);

  static void Transpose2DFunc(
    const TensorType<DType>* input, TensorType<DType>* output);

  static void MaximumFunc(
    const TensorType<DType>* left, const TensorType<DType>* right,
    TensorType<DType>* output);

  static void MinusFunc(
    const TensorType<DType>* left, const TensorType<DType>* right,
    TensorType<DType>* output);

  static DType SumFunc(const TensorType<DType>* input);

  static void AddFunc(
    const TensorType<DType>* left, const TensorType<DType>* right,
    TensorType<DType>* output);

  static void MultiplyFunc(
    const TensorType<DType>* left, const TensorType<DType>* right,
    TensorType<DType>* output);

  static void MultiplyFunc(
    const TensorType<DType>* left, TensorType<DType>* output,
    DType right);

  static void Convolution2DForwardFunc(
    const TensorType<DType>* input,
    const TensorType<DType>* filter,
    TensorType<DType>* output,
    TensorType<DType>* workspace,
    size_t padding_height, size_t padding_width,
    size_t stride_height, size_t stride_width,
    BLITZ_ALGORITHM algorithm = BLITZ_CONVOLUTION_BLAS_GEMM);

  static void Convolution2DBackwardFunc(
    const TensorType<DType>* output,
    const TensorType<DType>* filter,
    TensorType<DType>* input,
    TensorType<DType>* workspace,
    size_t padding_height, size_t padding_width,
    size_t stride_height, size_t stride_width,
    BLITZ_ALGORITHM algorithm = BLITZ_CONVOLUTION_BLAS_GEMM);

  static void Convolution2DUpdateFunc(
    const TensorType<DType>* input,
    const TensorType<DType>* output,
    TensorType<DType>* update,
    TensorType<DType>* workspace,
    size_t padding_height, size_t padding_width,
    size_t stride_height, size_t stride_width,
    BLITZ_ALGORITHM algorithm = BLITZ_CONVOLUTION_BLAS_GEMM);

  static void MaxPooling2DForwardFunc(
    const TensorType<DType>* input,
    TensorType<DType>* output,
    TensorType<size_t>* max_index,
    size_t filter_height, size_t filter_width,
    size_t stride_height, size_t stride_width);

  static void MaxPooling2DBackwardFunc(
    const TensorType<DType>* output, 
    TensorType<DType>* input,
    const TensorType<size_t>* max_index,
    size_t filter_height, size_t filter_width,
    size_t stride_height, size_t stride_width);

  static void MakeBinaryMaskFunc(
    TensorType<DType>* output,
    DType low,
    DType high,
    DType keep);

  static void ConstantDistributionFunc(TensorType<DType>* output, DType val);

  static void NormalDistributionFunc(TensorType<DType>* output, DType loc, DType scale);

  static void UniformDistributionFunc(TensorType<DType>* output, DType low, DType high);

  static void HostCopyToFunc(const DType* source, DType* target, size_t size);

  static float EvaluateClassifyFunc(
    const TensorType<DType>* output, const TensorType<DType>* target);

  static float EvaluateRegressFunc(
    const TensorType<DType>* output, const TensorType<DType>* target);

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
    size_t stride_width,
    BLITZ_DATA_LAYOUT input_data_layout);

  static void Pack2DFunc(
    const DType* unpack,
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
    size_t stride_width,
    BLITZ_DATA_LAYOUT input_data_layout);

  DISABLE_COPY_AND_ASSIGN(Backend);
};

}  // namespace blitz

#endif  // INCLUDE_BACKENDS_BACKEND_H_
