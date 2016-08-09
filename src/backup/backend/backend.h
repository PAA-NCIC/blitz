#ifndef SRC_BACKEND_BACKEND_H_
#define SRC_BACKEND_BACKEND_H_

#include <vector>

#include "util/common.h"

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
    const DType decay, const int batch_size,
    TensorType<DType>* weight,
    TensorType<DType>* gradient,
    TensorType<DType>* velocity);

  static void MatrixDotFunc(
    const TensorType<DType>* left, const TensorType<DType>* right,
    const bool transa, const bool transb,
    const DType alpha, const DType beta,
    TensorType<DType>* output);

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
    const TensorType<DType>* input, const TensorType<DType>* weight,
    const int padding_height, const int padding_width,
    const int stride_height, const int stride_width,
    TensorType<DType>* unpack, TensorType<DType>* output);

  static void Convolution2DBackwardFunc(
    const TensorType<DType>* output, const TensorType<DType>* weight,
    const int padding_height, const int padding_width,
    const int stride_height, const int stride_width,
    TensorType<DType>* pack, TensorType<DType>* input);

  static void Convolution2DUpdateFunc(
    const TensorType<DType>* input, const TensorType<DType>* output,
    const int padding_height, const int padding_width,
    const int stride_height, const int stride_width,
    TensorType<DType>* unpack, TensorType<DType>* update);

  // batch parallel
  static void Convolution2DForwardFunc(
    const TensorType<DType>* input, const TensorType<DType>* weight,
    const int padding_height, const int padding_width,
    const int stride_height, const int stride_width,
    vector<shared_ptr<TensorType<DType> > >* unpack_batch,
    TensorType<DType>* output);

  static void Convolution2DBackwardFunc(
    const TensorType<DType>* output, const TensorType<DType>* weight,
    const int padding_height, const int padding_width,
    const int stride_height, const int stride_width,
    vector<shared_ptr<TensorType<DType> > >* pack_batch,
    TensorType<DType>* input);

  static void Convolution2DUpdateFunc(
    const TensorType<DType>* input, const TensorType<DType>* output,
    const int padding_height, const int padding_width,
    const int stride_height, const int stride_width,
    vector<shared_ptr<TensorType<DType> > >* unpack_batch,
    vector<shared_ptr<TensorType<DType> > >* update_batch,
    TensorType<DType>* update);

  // naive parallel
  static void Convolution2DForwardFunc(
    const TensorType<DType>* input, const TensorType<DType>* weight,
    const int stride_height, const int stride_width,
    TensorType<DType>* output);

  static void MaxPooling2DForwardFunc(
    const TensorType<DType>* input,
    const int filter_height, const int filter_width,
    const int stride_height, const int stride_width,
    TensorType<int>* max_index, TensorType<DType>* output);

  static void MaxPooling2DBackwardFunc(
    const TensorType<DType>* output, const TensorType<int>* max_index,
    const int filter_height, const int filter_width,
    const int stride_height, const int stride_width,
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

  static void CopyFunc(const DType* source, const size_t size,
    DType* target);

  static float EvaluateFunc(
    const TensorType<DType>* output, const TensorType<DType>* target);

 private:
  static void Unpack2DParallelFunc(
    const DType* input, const int channel,
    const int input_height, const int input_width,
    const int filter_height, const int filter_width,
    const int output_height, const int output_width,
    const int padding_height, const int padding_width,
    const int stride_height, const int stride_width,
    DType* unpack);

  static void Pack2DParallelFunc(
    const DType* pack, const int channel,
    const int input_height, const int input_width,
    const int filter_height, const int filter_width,
    const int output_height, const int output_width,
    const int padding_height, const int padding_width,
    const int stride_height, const int stride_width,
    DType* input);

  static void Unpack2DFunc(
    const DType* input, const int channel,
    const int input_height, const int input_width,
    const int filter_height, const int filter_width,
    const int output_height, const int output_width,
    const int padding_height, const int padding_width,
    const int stride_height, const int stride_width,
    DType* unpack);

  static void Pack2DFunc(
    const DType* pack, const int channel,
    const int input_height, const int input_width,
    const int filter_height, const int filter_width,
    const int output_height, const int output_width,
    const int padding_height, const int padding_width,
    const int stride_height, const int stride_width,
    DType* input);
};

}  // namespace blitz

#endif  // SRC_BACKEND_BACKEND_H_
