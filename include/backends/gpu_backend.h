#ifndef INCLUDE_BACKENDS_GPU_BACKEND_H_
#define INCLUDE_BACKENDS_GPU_BACKEND_H_

#include "backends/backend.h"
#include "backends/gpu_tensor.h"
#include "utils/blitz_gpu_function.h"

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
    ConvolutionContext<GPUTensor, DType>* context);

  static void Convolution2DBackwardFunc(
    const GPUTensor<DType>* output,
    const GPUTensor<DType>* filter,
    GPUTensor<DType>* input,
    ConvolutionContext<GPUTensor, DType>* context);

  static void Convolution2DUpdateFunc(
    const GPUTensor<DType>* input,
    const GPUTensor<DType>* output,
    GPUTensor<DType>* update,
    ConvolutionContext<GPUTensor, DType>* context);

  static void MaxPooling2DForwardFunc(
    const GPUTensor<DType>* input,
    GPUTensor<DType>* output,
    GPUTensor<size_t>* max_index,
    size_t R, size_t S,
    size_t str_h, size_t str_w);

  static void MaxPooling2DBackwardFunc(
    const GPUTensor<DType>* output, 
    GPUTensor<DType>* input,
    const GPUTensor<size_t>* max_index);

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

  static void TransformCopyFunc(const GPUTensor<DType>* source, GPUTensor<DType>* dest);

  static void Unpack2DFunc(
    const GPUTensor<DType>* input,
    GPUTensor<DType>* unpack,
    size_t R, size_t S,
    size_t pad_h, size_t pad_w,
    size_t str_h, size_t str_w);

  static void Pack2DFunc(
    const GPUTensor<DType>* unpack,
    GPUTensor<DType>* input,
    size_t R, size_t S,
    size_t pad_h, size_t pad_w,
    size_t str_h, size_t str_w);

 private:
  //static void Unpack2DDispatch(
  //  const DType *input,
  //  DType *unpack,
  //  size_t C, size_t H, size_t W,
  //  size_t R, size_t S,
  //  size_t P, size_t Q,
  //  size_t pad_h, size_t pad_w,
  //  size_t str_h, size_t str_w,
  //  BLITZ_DATA_LAYOUT input_data_layout);

  //static void Pack2DDispatch(
  //  const DType *unpack,
  //  DType *input,
  //  size_t C, size_t H, size_t W,
  //  size_t R, size_t S,
  //  size_t P, size_t Q,
  //  size_t pad_h, size_t pad_w,
  //  size_t str_h, size_t str_w,
  //  BLITZ_DATA_LAYOUT input_data_layout);
};

}  // namespace blitz

#endif  // INCLUDE_BACKENDS_GPU_BACKEND_H_
