#ifndef INCLUDE_BACKENDS_CPU_BACKEND_H_
#define INCLUDE_BACKENDS_CPU_BACKEND_H_

#include <vector>
#include <string>

#include "backends/backend.h"
#include "backends/cpu_tensor.h"

namespace blitz {

template<typename DType>
class Backend<CPUTensor, DType> {
 public:
  static void RectlinApplyFunc(
    const CPUTensor<DType>* input, CPUTensor<DType>* output,
    DType slope);

  static void RectlinDerivativeFunc(
    const CPUTensor<DType>* input, CPUTensor<DType>* output,
    DType slope);

  static void LogisticApplyFunc(
    const CPUTensor<DType>* input, CPUTensor<DType>* output);

  static void LogisticDerivativeFunc(
    const CPUTensor<DType>* input, CPUTensor<DType>* output);

  static void SoftmaxApplyFunc(
    const CPUTensor<DType>* input, CPUTensor<DType>* output);

  static void SoftmaxDerivativeFunc(
    const CPUTensor<DType>* input, CPUTensor<DType>* output);

  static DType SquareMeanApplyFunc(
    const CPUTensor<DType>* input, const CPUTensor<DType>* target);

  static void SquareMeanDerivativeFunc(
    const CPUTensor<DType>* input, const CPUTensor<DType>* target,
    CPUTensor<DType>* output);

  static DType AbsMeanApplyFunc(
    const CPUTensor<DType>* input, const CPUTensor<DType>* target);

  static void AbsMeanDerivativeFunc(
    const CPUTensor<DType>* input, const CPUTensor<DType>* target,
    CPUTensor<DType>* output);

  static DType CrossEntropyBinaryApplyFunc(
    const CPUTensor<DType>* input, const CPUTensor<DType>* target);

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

  static void BiasBackwardUpdateFunc(
    const CPUTensor<DType>* input, CPUTensor<DType>* update);

  static void BatchNormForwardFunc(
    const CPUTensor<DType>* input,
    const CPUTensor<DType>* gamma,
    const CPUTensor<DType>* beta,
    CPUTensor<DType>* input_var,
    CPUTensor<DType>* input_hat,
    CPUTensor<DType>* output,
    DType epsilon);

  static void BatchNormBackwardFunc(
    const CPUTensor<DType>* backward_input,
    const CPUTensor<DType>* forward_input_hat,
    const CPUTensor<DType>* forward_input_var,
    const CPUTensor<DType>* gamma,
    CPUTensor<DType>* gamma_update,
    CPUTensor<DType>* beta_update,
    CPUTensor<DType>* output,
    DType epsilon);

  static void GradientdescentFunc(
    CPUTensor<DType>* filter,
    CPUTensor<DType>* gradient,
    CPUTensor<DType>* velocity,
    DType momentum_coef,
    DType learning_rate,
    DType decay,
    size_t batch_size);

  static void MatrixMultiplyFunc(
    const CPUTensor<DType>* left,
    const CPUTensor<DType>* right,
    CPUTensor<DType>* output, 
    bool transa,
    bool transb,
    DType alpha,
    DType beta,
    BLITZ_ALGORITHM algorithm = BLITZ_BLAS_GEMM);

  static void Transpose2DFunc(
    const CPUTensor<DType>* input, CPUTensor<DType>* output);

  static void MaximumFunc(
    const CPUTensor<DType>* left, const CPUTensor<DType>* right,
    CPUTensor<DType>* output);

  static void MinusFunc(
    const CPUTensor<DType>* left, const CPUTensor<DType>* right,
    CPUTensor<DType>* output);

  static DType SumFunc(const CPUTensor<DType>* input);

  static void AddFunc(
    const CPUTensor<DType>* left, const CPUTensor<DType>* right,
    CPUTensor<DType>* output);

  static void MultiplyFunc(
    const CPUTensor<DType>* left, const CPUTensor<DType>* right,
    CPUTensor<DType>* output);

  static void MultiplyFunc(
    const CPUTensor<DType>* left, CPUTensor<DType>* output,
    DType right);

  static void Convolution2DForwardFunc(
    const CPUTensor<DType>* input,
    const CPUTensor<DType>* filter,
    CPUTensor<DType>* output,
    CPUTensor<DType>* workspace,
    size_t padding_height, size_t padding_width,
    size_t stride_height, size_t stride_width,
    BLITZ_ALGORITHM algorithm = BLITZ_CONVOLUTION_BLAS_GEMM);

  static void Convolution2DBackwardFunc(
    const CPUTensor<DType>* output,
    const CPUTensor<DType>* filter,
    CPUTensor<DType>* input,
    CPUTensor<DType>* workspace,
    size_t padding_height, size_t padding_width,
    size_t stride_height, size_t stride_width,
    BLITZ_ALGORITHM algorithm = BLITZ_CONVOLUTION_BLAS_GEMM);

  static void Convolution2DUpdateFunc(
    const CPUTensor<DType>* input,
    const CPUTensor<DType>* output,
    CPUTensor<DType>* update,
    CPUTensor<DType>* workspace,
    size_t padding_height, size_t padding_width,
    size_t stride_height, size_t stride_width,
    BLITZ_ALGORITHM algorithm = BLITZ_CONVOLUTION_BLAS_GEMM);

  static void MaxPooling2DForwardFunc(
    const CPUTensor<DType>* input,
    CPUTensor<DType>* output,
    CPUTensor<size_t>* max_index,
    size_t filter_height, size_t filter_width,
    size_t stride_height, size_t stride_width);

  static void MaxPooling2DBackwardFunc(
    const CPUTensor<DType>* output, 
    CPUTensor<DType>* input,
    const CPUTensor<size_t>* max_index,
    size_t filter_height, size_t filter_width,
    size_t stride_height, size_t stride_width);

  static void MakeBinaryMaskFunc(
    CPUTensor<DType>* output,
    DType low,
    DType high,
    DType keep);

  static void ConstantDistributionFunc(CPUTensor<DType>* output, DType val);

  static void NormalDistributionFunc(CPUTensor<DType>* output, DType loc, DType scale);

  static void UniformDistributionFunc(CPUTensor<DType>* output, DType low, DType high);

  static void HostCopyToFunc(const DType* source, DType* target, size_t size);

  static float EvaluateClassifyFunc(
    const CPUTensor<DType>* output, const CPUTensor<DType>* target);

  static float EvaluateRegressFunc(
    const CPUTensor<DType>* output, const CPUTensor<DType>* target);

  static BLITZ_DATA_LAYOUT Unpack2DFunc(
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
		BLITZ_DATA_LAYOUT input_data_layout = BLITZ_PACK_CRSPQ);

  static BLITZ_DATA_LAYOUT Pack2DFunc(
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
    size_t stride_width,
		BLITZ_DATA_LAYOUT pack_data_layout = BLITZ_PACK_CRSPQ);

 private:
	static void ConvolutionForwardGEMMDispatch(
		DType* unpack,
		DType* output,
		DType* filter,
		size_t K, size_t PQ, size_t CRS,
		BLITZ_DATA_LAYOUT unpack_data_layout,
		BLITZ_DATA_LAYOUT output_data_layout,
		BLITZ_DATA_LAYOUT filter_data_layout);
};

}  // namespace blitz

#endif  // INCLUDE_BACKENDS_CPU_BACKEND_H_
