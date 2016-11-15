#ifndef INCLUDE_BACKENDS_MIC_BACKEND_H_
#define INCLUDE_BACKENDS_MIC_BACKEND_H_

#include <vector>
#include <string>

#include "backends/backend.h"
#include "backends/mic_tensor.h"

namespace blitz {

template<typename DType>
class Backend<MICTensor, DType> {
 public:
  static void RectlinApplyFunc(
    const MICTensor<DType>* input, MICTensor<DType>* output,
    DType slope);

  static void RectlinDerivativeFunc(
    const MICTensor<DType>* input, MICTensor<DType>* output,
    DType slope);

  static void LogisticApplyFunc(
    const MICTensor<DType>* input, MICTensor<DType>* output);

  static void LogisticDerivativeFunc(
    const MICTensor<DType>* input, MICTensor<DType>* output);

  static void SoftmaxApplyFunc(
    const MICTensor<DType>* input, MICTensor<DType>* output);

  static void SoftmaxDerivativeFunc(
    const MICTensor<DType>* input, MICTensor<DType>* output);

  static DType SquareMeanApplyFunc(
    const MICTensor<DType>* input, const MICTensor<DType>* target);

  static void SquareMeanDerivativeFunc(
    const MICTensor<DType>* input, const MICTensor<DType>* target,
    MICTensor<DType>* output);

  static DType AbsMeanApplyFunc(
    const MICTensor<DType>* input, const MICTensor<DType>* target);

  static void AbsMeanDerivativeFunc(
    const MICTensor<DType>* input, const MICTensor<DType>* target,
    MICTensor<DType>* output);

  static DType CrossEntropyBinaryApplyFunc(
    const MICTensor<DType>* input, const MICTensor<DType>* target);

  static void CrossEntropyBinaryDerivativeFunc(
    const MICTensor<DType>* input, const MICTensor<DType>* target,
    MICTensor<DType>* output);

  static DType CrossEntropyMultiApplyFunc(
    const MICTensor<DType>* input, const MICTensor<DType>* target);

  static void CrossEntropyMultiDerivativeFunc(
    const MICTensor<DType>* input, const MICTensor<DType>* target,
    MICTensor<DType>* output);

  static void BiasForwardFunc(
    const MICTensor<DType>* input, const MICTensor<DType>* bias,
    MICTensor<DType>* output);

  static void BiasBackwardUpdateFunc(
    const MICTensor<DType>* input, MICTensor<DType>* update);

  static void BatchNormForwardFunc(
    const MICTensor<DType>* input,
    const MICTensor<DType>* gamma,
    const MICTensor<DType>* beta,
    MICTensor<DType>* input_var,
    MICTensor<DType>* input_hat,
    MICTensor<DType>* output,
    DType epsilon);

  static void BatchNormBackwardFunc(
    const MICTensor<DType>* backward_input,
    const MICTensor<DType>* forward_input_hat,
    const MICTensor<DType>* forward_input_var,
    const MICTensor<DType>* gamma,
    MICTensor<DType>* gamma_update,
    MICTensor<DType>* beta_update,
    MICTensor<DType>* output,
    DType epsilon);

  static void GradientdescentFunc(
    MICTensor<DType>* filter,
    MICTensor<DType>* gradient,
    MICTensor<DType>* velocity,
    DType momentum_coef,
    DType learning_rate,
    DType decay,
    size_t batch_size);

  static void MatrixMultiplyFunc(
    const MICTensor<DType>* left,
    const MICTensor<DType>* right,
    MICTensor<DType>* output, 
    bool transa,
    bool transb,
    DType alpha,
    DType beta,
    BLITZ_ALGORITHM algorithm = BLITZ_BLAS_GEMM);

  static void Transpose2DFunc(
    const MICTensor<DType>* input, MICTensor<DType>* output);

  static void MaximumFunc(
    const MICTensor<DType>* left, const MICTensor<DType>* right,
    MICTensor<DType>* output);

  static void MinusFunc(
    const MICTensor<DType>* left, const MICTensor<DType>* right,
    MICTensor<DType>* output);

  static DType SumFunc(const MICTensor<DType>* input);

  static void AddFunc(
    const MICTensor<DType>* left, const MICTensor<DType>* right,
    MICTensor<DType>* output);

  static void MultiplyFunc(
    const MICTensor<DType>* left, const MICTensor<DType>* right,
    MICTensor<DType>* output);

  static void MultiplyFunc(
    const MICTensor<DType>* left, MICTensor<DType>* output,
    DType right);

  static void Convolution2DForwardFunc(
    const MICTensor<DType>* input,
    const MICTensor<DType>* filter,
    MICTensor<DType>* output,
    MICTensor<DType>* workspace,
    size_t padding_height, size_t padding_width,
    size_t stride_height, size_t stride_width,
    BLITZ_ALGORITHM algorithm = BLITZ_CONVOLUTION_BLAS_GEMM);

  static void Convolution2DBackwardFunc(
    const MICTensor<DType>* output,
    const MICTensor<DType>* filter,
    MICTensor<DType>* input,
    MICTensor<DType>* workspace,
    size_t padding_height, size_t padding_width,
    size_t stride_height, size_t stride_width,
		BLITZ_ALGORITHM algorithm = BLITZ_CONVOLUTION_BLAS_GEMM);

  static void Convolution2DUpdateFunc(
    const MICTensor<DType>* input,
    const MICTensor<DType>* output,
    MICTensor<DType>* update,
    MICTensor<DType>* workspace,
    size_t padding_height, size_t padding_width,
    size_t stride_height, size_t stride_width,
		BLITZ_ALGORITHM algorithm = BLITZ_CONVOLUTION_BLAS_GEMM);

  static void MaxPooling2DForwardFunc(
    const MICTensor<DType>* input,
    MICTensor<DType>* output,
    MICTensor<size_t>* max_index,
    size_t filter_height, size_t filter_width,
    size_t stride_height, size_t stride_width);

  static void MaxPooling2DBackwardFunc(
    const MICTensor<DType>* output, 
    MICTensor<DType>* input,
    const MICTensor<size_t>* max_index,
    size_t filter_height, size_t filter_width,
    size_t stride_height, size_t stride_width);

  static void MakeBinaryMaskFunc(
    MICTensor<DType>* output,
    DType low,
    DType high,
    DType keep);

  static void ConstantDistributionFunc(MICTensor<DType>* output, DType val);

  static void NormalDistributionFunc(MICTensor<DType>* output, DType loc, DType scale);

  static void UniformDistributionFunc(MICTensor<DType>* output, DType low, DType high);

  static void HostCopyToFunc(const DType* source, DType* target, size_t size);

  static float EvaluateClassifyFunc(
    const MICTensor<DType>* output, const MICTensor<DType>* target);

  static float EvaluateRegressFunc(
    const MICTensor<DType>* output, const MICTensor<DType>* target);

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

#endif  // INCLUDE_BACKENDS_MIC_BACKEND_H_
