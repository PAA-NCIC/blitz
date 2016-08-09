#ifndef SRC_BACKEND_GPU_BACKEND_H_
#define SRC_BACKEND_GPU_BACKEND_H_

#include "backend/backend.h"
#include "backend/gpu_tensor.h"

namespace blitz {

template<typename DType>
class Backend<GPUTensor, DType> {
 public:
  static void RectlinApplyFunc(
    const GPUTensor<DType>* input,
    const DType slope,
    GPUTensor<DType>* output) {}

  static void RectlinDerivativeFunc(
    const GPUTensor<DType>* input,
    const DType slope,
    GPUTensor<DType>* output) {}

  static void SoftmaxApplyFunc(
    const GPUTensor<DType>* input,
    GPUTensor<DType>* output) {}

  static void SoftmaxDerivativeFunc(
    const GPUTensor<DType>* input,
    GPUTensor<DType>* output) {}

  static void LogisticApplyFunc(
    const GPUTensor<DType>* input,
    GPUTensor<DType>* output) {}

  static void LogisticDerivativeFunc(
    const GPUTensor<DType>* input,
    GPUTensor<DType>* output) {}

  static DType CrossEntropyBinaryApplyFunc(
    const GPUTensor<DType>* input, const GPUTensor<DType>* target) {
    return 0;
  }

  static void CrossEntropyBinaryDerivativeFunc(
    const GPUTensor<DType>* input, const GPUTensor<DType>* target,
    GPUTensor<DType>* output) {}

  static DType CrossEntropyMultiApplyFunc(
    const GPUTensor<DType>* input, const GPUTensor<DType>* target) {
    return 0;
  }

  static void CrossEntropyMultiDerivativeFunc(
    const GPUTensor<DType>* input, const GPUTensor<DType>* target,
    GPUTensor<DType>* output) {
  }

  static void BiasForwardFunc(
    const GPUTensor<DType>* input, const GPUTensor<DType>* bias,
    GPUTensor<DType>* output) {
  }

  static void BiasBackwardUpdateFunc(const GPUTensor<DType>* input,
    GPUTensor<DType>* update) {
  }

  static void BatchNormForwardFunc(
    const GPUTensor<DType>* input, const GPUTensor<DType>* gamma,
    const GPUTensor<DType>* beta, const DType epsilon,
    GPUTensor<DType>* input_var, GPUTensor<DType>* input_hat,
    GPUTensor<DType>* output) {}

  static void BatchNormBackwardFunc(
    const GPUTensor<DType>* backward_input,
    const GPUTensor<DType>* forward_input_hat,
    const GPUTensor<DType>* forward_input_var,
    const GPUTensor<DType>* gamma, const DType epsilon,
    GPUTensor<DType>* gamma_update, GPUTensor<DType>* beta_update,
    GPUTensor<DType>* output) {}

  static void GradientdescentFunc(
    const DType momentum_coef, const DType learning_rate,
    const DType decay, const int batch_size,
    GPUTensor<DType>* weight,
    GPUTensor<DType>* gradient,
    GPUTensor<DType>* velocity) {}

  static void MatrixDotFunc(
    const GPUTensor<DType>* left, const GPUTensor<DType>* right,
    const bool transa, const bool transb,
    const DType alpha, const DType beta,
    GPUTensor<DType>* output) {}

  static void MaximumFunc(
    const GPUTensor<DType>* left, const GPUTensor<DType>* right,
    GPUTensor<DType>* output) {}

  static void MaximumFunc(
    const GPUTensor<DType>* left,
    const DType right,
    GPUTensor<DType>* output) {}

  static void MinusFunc(
    const GPUTensor<DType>* left, const GPUTensor<DType>* right,
    GPUTensor<DType>* output) {}

  static void MinusFunc(
    const GPUTensor<DType>* left,
    const DType right,
    GPUTensor<DType>* output) {}

  static DType SumFunc(
    const GPUTensor<DType>* input) {
    return 0;
  }

  static void AddFunc(
    const GPUTensor<DType>* left, const GPUTensor<DType>* right,
    GPUTensor<DType>* output) {}

  static void MultiplyFunc(
    const GPUTensor<DType>* left, const GPUTensor<DType>* right,
    GPUTensor<DType>* output) {}

  static void MultiplyFunc(
    const GPUTensor<DType>* left, const DType right,
    GPUTensor<DType>* output) {}

  static void Convolution2DForwardFunc(
    const GPUTensor<DType>* input, const GPUTensor<DType>* filter,
    const int padding_height, const int padding_width,
    const int stride_height, const int stride_width,
    GPUTensor<DType>* unpack, GPUTensor<DType>* output) {}

  static void Convolution2DBackwardFunc(
    const GPUTensor<DType>* output, const GPUTensor<DType>* filter,
    const int padding_height, const int padding_width,
    const int stride_height, const int stride_width,
    GPUTensor<DType>* pack, GPUTensor<DType>* input) {}

  static void Convolution2DUpdateFunc(
    const GPUTensor<DType>* input, const GPUTensor<DType>* output,
    const int padding_height, const int padding_width,
    const int stride_height, const int stride_width,
    GPUTensor<DType>* unpack, GPUTensor<DType>* update) {}

  // batch parallel
  static void Convolution2DForwardFunc(
    const GPUTensor<DType>* input, const GPUTensor<DType>* filter,
    const int padding_height, const int padding_width,
    const int stride_height, const int stride_width,
    vector<shared_ptr<GPUTensor<DType> > >* unpack_batch,
    GPUTensor<DType>* output) {}

  static void Convolution2DBackwardFunc(
    const GPUTensor<DType>* output, const GPUTensor<DType>* filter,
    const int padding_height, const int padding_width,
    const int stride_height, const int stride_width,
    vector<shared_ptr<GPUTensor<DType> > >* pack_batch,
    GPUTensor<DType>* input) {}

  static void Convolution2DUpdateFunc(
    const GPUTensor<DType>* input, const GPUTensor<DType>* output,
    const int padding_height, const int padding_width,
    const int stride_height, const int stride_width,
    vector<shared_ptr<GPUTensor<DType> > >* unpack_batch,
    vector<shared_ptr<GPUTensor<DType> > >* update_batch,
    GPUTensor<DType>* update) {}

  // naive parallel
  static void Convolution2DForwardFunc(
    const GPUTensor<DType>* input, const GPUTensor<DType>* weight,
    const int stride_height, const int stride_width,
    GPUTensor<DType>* output) {}

  static void MaxPooling2DForwardFunc(
    const GPUTensor<DType>* input,
    const int filter_height, const int filter_width,
    const int stride_height, const int stride_width,
    GPUTensor<int>* max_index, GPUTensor<DType>* output) {}

  static void MaxPooling2DBackwardFunc(
    const GPUTensor<DType>* output, const GPUTensor<int>* max_index,
    const int filter_height, const int filter_width,
    const int stride_height, const int stride_width,
    GPUTensor<DType>* input) {}

  static void MakeBinaryMaskFunc(const DType low, const DType high,
    const DType keep, GPUTensor<DType>* output) {}

  static void ConstantDistributionFunc(
    const DType val, GPUTensor<DType>* output) {}

  static void NormalDistributionFunc(
    const DType loc, const DType scale,
    GPUTensor<DType>* output) {}

  static void UniformDistributionFunc(
    const DType low, const DType high,
    GPUTensor<DType>* output) {}

  static void CopyFunc(const DType* source, const size_t size,
    DType* target) {}

  static float EvaluateFunc(
    const GPUTensor<DType>* output, const GPUTensor<DType>* target) {
    return 0.0f;
  }

 private:
  static void Unpack2DParallelFunc(
    const DType* input, const int channel,
    const int input_height, const int input_width,
    const int filter_height, const int filter_width,
    const int output_height, const int output_width,
    const int padding_height, const int padding_width,
    const int stride_height, const int stride_width,
    DType* unpack) {}

  static void Pack2DParallelFunc(
    const DType* pack, const int channel,
    const int input_height, const int input_width,
    const int filter_height, const int filter_width,
    const int output_height, const int output_width,
    const int padding_height, const int padding_width,
    const int stride_height, const int stride_width,
    DType* input) {}

  static void Unpack2DFunc(
    const DType* input, const int channel,
    const int input_height, const int input_width,
    const int filter_height, const int filter_width,
    const int output_height, const int output_width,
    const int padding_height, const int padding_width,
    const int stride_height, const int stride_width,
    DType* unpack) {}

  static void Pack2DFunc(
    const DType* pack, const int channel,
    const int input_height, const int input_width,
    const int filter_height, const int filter_width,
    const int output_height, const int output_width,
    const int padding_height, const int padding_width,
    const int stride_height, const int stride_width,
    DType* input) {}
};

}  // namespace blitz

#endif  // SRC_BACKEND_GPU_BACKEND_H_
