#ifndef SRC_BACKEND_GPU_BACKEND_COMMON_INL_H_
#define SRC_BACKEND_GPU_BACKEND_COMMON_INL_H_

template<typename DType>
__global__ void GPURectlinApply(const DType* input,
  DType* output, const int size, const DType compare_value,
  const DType slope) {
  BLITZ_CUDA_LOOP(i, size) {
    DType greater = input[i] > compare_value ? input[i] : compare_value;
    DType less = input[i] <= compare_value ?
      slope * input[i] : slope * compare_value;
    output[i] = greater + less;
  }
}

template<typename DType>
void Backend<GPUTensor, DType>::RectlinApplyFunc(
  const GPUTensor<DType>* input, const DType slope,
  GPUTensor<DType>* output) {
  CHECK_EQ(input->size(), output->size());

  DType compara_value = static_cast<DType>(0);
  GPURectlinApply<DType><<<BlitzGPUGetBlocks(input->size()),
    BLITZ_NUM_GPU_THREADS>>>(input->data(), output->data(),
    input->size(), compara_value, slope);
}

template<typename DType>
__global__ void GPURectlinDerivative(const DType* input,
  DType* output, const int size, const DType compare_value,
  const DType slope) {
  BLITZ_CUDA_LOOP(i, size) {
    DType greater = input[i] > compare_value ? 1.0 : 0.0;
    DType less = input[i] <= compare_value ? slope : 0.0;
    output[i] = (greater + less) * output[i];
  }
}

template<typename DType>
void Backend<GPUTensor, DType>::RectlinDerivativeFunc(
  const GPUTensor<DType>* input,
  const DType slope,
  GPUTensor<DType>* output) {
  CHECK_EQ(input->size(), output->size());

  DType compara_value = static_cast<DType>(0);
  GPURectlinDerivative<DType><<<BlitzGPUGetBlocks(input->size()),
    BLITZ_NUM_GPU_THREADS>>>(input->data(), output->data(),
    input->size(), compara_value, slope);
}

template<typename DType>
void Backend<GPUTensor, DType>::SoftmaxApplyFunc(
  const GPUTensor<DType>* input,
  GPUTensor<DType>* output) {}

template<typename DType>
void Backend<GPUTensor, DType>::SoftmaxDerivativeFunc(
  const GPUTensor<DType>* input,
  GPUTensor<DType>* output) {}

template<typename DType>
DType Backend<GPUTensor, DType>::SquareMeanApplyFunc(
  const GPUTensor<DType>* input,
  const GPUTensor<DType>* target) {
  return 0;
}

template<typename DType>
void Backend<GPUTensor, DType>::SquareMeanDerivativeFunc(
  const GPUTensor<DType>* input, const GPUTensor<DType>* target,
  GPUTensor<DType>* output) {}

template<typename DType>
DType Backend<GPUTensor, DType>::AbsMeanApplyFunc(
  const GPUTensor<DType>* input,
  const GPUTensor<DType>* target) {
  return 0;
}

template<typename DType>
void Backend<GPUTensor, DType>::AbsMeanDerivativeFunc(
  const GPUTensor<DType>* input, const GPUTensor<DType>* target,
  GPUTensor<DType>* output) {}

template<typename DType>
__global__ void GPULogisticApply(const DType* input,
  DType* output, const int size) {
  BLITZ_CUDA_LOOP(i, size) {
    output[i] = 1 / (exp(-input[i]) + 1);
  }
}

template<typename DType>
void Backend<GPUTensor, DType>::LogisticApplyFunc(
  const GPUTensor<DType>* input,
  GPUTensor<DType>* output) {
  CHECK_EQ(input->size(), output->size());

  GPULogisticApply<DType><<<BlitzGPUGetBlocks(input->size()),
    BLITZ_NUM_GPU_THREADS>>>(input->data(), output->data(),
    input->size());
}

template<typename DType>
void Backend<GPUTensor, DType>::LogisticDerivativeFunc(
  const GPUTensor<DType>* input,
  GPUTensor<DType>* output) {
  CHECK_EQ(input->size(), output->size());
  // TODO(keren) not short cut version
}

template<typename DType>
__global__ void GPUCrossEntropyBinary(const DType* input,
  const DType* target, const int size, DType* sum) {
  BLITZ_CUDA_LOOP(i, size) {
    DType safe_input = BlitzGPUSafeLog(input[i]);
    DType safe_inverse_input = BlitzGPUSafeLog(1 - input[i]);
    sum[i] += -safe_input * target[i] - safe_inverse_input
      * (1 - target[i]);
  }
}

template<typename DType>
DType Backend<GPUTensor, DType>::CrossEntropyBinaryApplyFunc(
  const GPUTensor<DType>* input, const GPUTensor<DType>* target) {
  GPUTensor<DType> sum(input->shape());
  GPUCrossEntropyBinary<DType><<<BlitzGPUGetBlocks(input->size()),
    BLITZ_NUM_GPU_THREADS>>>(input->data(), target->data(),
    input->size(), sum.data());
  thrust::device_ptr<DType> sptr = thrust::device_pointer_cast(sum.data());
  DType loss = thrust::reduce(sptr, sptr + sum.size());
  return loss / input->shape()[0];
}

template<typename DType>
void Backend<GPUTensor, DType>::CrossEntropyBinaryDerivativeFunc(
  const GPUTensor<DType>* input, const GPUTensor<DType>* target,
  GPUTensor<DType>* output) {
  MinusFunc(input, target, output);
}

template<typename DType>
DType Backend<GPUTensor, DType>::CrossEntropyMultiApplyFunc(
  const GPUTensor<DType>* input, const GPUTensor<DType>* target) {
  CHECK_EQ(input->size(), target->size());
  const Shape& input_shape = input->shape();
  const Shape& target_shape = target->shape();
  // softmax loss
  return 0;
}

template<typename DType>
void Backend<GPUTensor, DType>::CrossEntropyMultiDerivativeFunc(
  const GPUTensor<DType>* input, const GPUTensor<DType>* target,
  GPUTensor<DType>* output) {
}

template<typename DType>
void Backend<GPUTensor, DType>::BiasForwardFunc(
  const GPUTensor<DType>* input, const GPUTensor<DType>* bias,
  GPUTensor<DType>* output) {
}

template<typename DType>
void Backend<GPUTensor, DType>::BiasBackwardUpdateFunc(
  const GPUTensor<DType>* input, GPUTensor<DType>* update) {
}

template<typename DType>
void Backend<GPUTensor, DType>::BatchNormForwardFunc(
  const GPUTensor<DType>* input, const GPUTensor<DType>* gamma,
  const GPUTensor<DType>* beta, const DType epsilon,
  GPUTensor<DType>* input_var, GPUTensor<DType>* input_hat,
  GPUTensor<DType>* output) {}

template<typename DType>
void Backend<GPUTensor, DType>::BatchNormBackwardFunc(
  const GPUTensor<DType>* backward_input,
  const GPUTensor<DType>* forward_input_hat,
  const GPUTensor<DType>* forward_input_var,
  const GPUTensor<DType>* gamma, const DType epsilon,
  GPUTensor<DType>* gamma_update, GPUTensor<DType>* beta_update,
  GPUTensor<DType>* output) {}

template<typename DType>
__global__ void GPUGradientdescent(
  const DType momentum_coef, const DType learning_rate,
  const DType decay, const int batch_size,
  DType* weight, DType* gradient, DType* velocity,
  const int size) {
  BLITZ_CUDA_LOOP(i, size) {
    gradient[i] /= batch_size;
    velocity[i] = velocity[i] * momentum_coef - learning_rate *
      gradient[i] + decay * weight[i];
    weight[i] = weight[i] + velocity[i];
  }
}

template<typename DType>
void Backend<GPUTensor, DType>::GradientdescentFunc(
  const DType momentum_coef, const DType learning_rate,
  const DType decay, const int batch_size,
  GPUTensor<DType>* weight,
  GPUTensor<DType>* gradient,
  GPUTensor<DType>* velocity) {
  CHECK_EQ(weight->size(), gradient->size());
  CHECK_EQ(gradient->size(), velocity->size());
  GPUGradientdescent<DType><<<BlitzGPUGetBlocks(gradient->size()),
    BLITZ_NUM_GPU_THREADS>>>(momentum_coef, learning_rate,
    decay, batch_size, weight->data(), gradient->data(),
    velocity->data(), gradient->size());
}

template<typename DType>
void Backend<GPUTensor, DType>::MatrixDotFunc(
  const GPUTensor<DType>* left, const GPUTensor<DType>* right,
  const bool transa, const bool transb,
  const DType alpha, const DType beta,
  GPUTensor<DType>* output, const string& kernel) {
  bool gpu_transa = left->row_major()? transa : !transa;
  bool gpu_transb = right->row_major()? transb : !transb;
  int dim_left = gpu_transa ? left->size() / (left->shape())[0] :
    (left->shape())[0];
  int dim_right = gpu_transb ? (right->shape())[0] :
    right->size() / (right->shape())[0];
  int dim_common_left = gpu_transa ? (left->shape())[0] :
    left->size() / (left->shape())[0];
  int dim_common_right = gpu_transb ? right->size() / (right->shape())[0] :
    (right->shape())[0];
  CHECK_EQ(dim_common_left, dim_common_right);
#ifdef BLITZ_DEVELOP
  LOG(INFO) << "dim left: " << dim_left;
  LOG(INFO) << "dim common: " << dim_common_left;
  LOG(INFO) << "dim right: " << dim_right;
#endif
  if (kernel == "blas") {
    BlitzGPUGemm(gpu_transa, gpu_transb, dim_left, dim_right, dim_common_left,
      const_cast<GPUTensor<DType>*>(left)->data(),
      const_cast<GPUTensor<DType>*>(right)->data(),
      output->data(), alpha, beta);
  } else if (kernel == "asm") {
    BlitzSassGemm(gpu_transa, gpu_transb, dim_left, dim_right, dim_common_left,
      const_cast<GPUTensor<DType>*>(left)->data(),
      const_cast<GPUTensor<DType>*>(right)->data(),
      output->data(), alpha, beta);
  }
}

template<typename DType>
void Backend<GPUTensor, DType>::MaximumFunc(
  const GPUTensor<DType>* left, const GPUTensor<DType>* right,
  GPUTensor<DType>* output) {}

template<typename DType>
void Backend<GPUTensor, DType>::MaximumFunc(
  const GPUTensor<DType>* left,
  const DType right,
  GPUTensor<DType>* output) {}

template<typename DType>
__global__ void GPUMinus(const DType* left,
  const DType* right, const int size, DType* output) {
  BLITZ_CUDA_LOOP(index, size) {
    output[index] = left[index] - right[index];
  }
}

template<typename DType>
void Backend<GPUTensor, DType>::MinusFunc(
  const GPUTensor<DType>* left, const GPUTensor<DType>* right,
  GPUTensor<DType>* output) {
  CHECK_EQ(left->size(), right->size());
  CHECK_EQ(right->size(), output->size());
  GPUMinus<DType><<<BlitzGPUGetBlocks(left->size()),
    BLITZ_NUM_GPU_THREADS>>>(left->data(), right->data(),
    left->size(), output->data());
}

template<typename DType>
void Backend<GPUTensor, DType>::MinusFunc(
  const GPUTensor<DType>* left,
  const DType right,
  GPUTensor<DType>* output) {}

template<typename DType>
DType Backend<GPUTensor, DType>::SumFunc(
  const GPUTensor<DType>* input) {
  return 0;
}

template<typename DType>
void Backend<GPUTensor, DType>::AddFunc(
  const GPUTensor<DType>* left, const GPUTensor<DType>* right,
  GPUTensor<DType>* output) {}

template<typename DType>
void Backend<GPUTensor, DType>::MultiplyFunc(
  const GPUTensor<DType>* left, const GPUTensor<DType>* right,
  GPUTensor<DType>* output) {}

template<typename DType>
void Backend<GPUTensor, DType>::MultiplyFunc(
  const GPUTensor<DType>* left, const DType right,
  GPUTensor<DType>* output) {}

template<typename DType>
void Backend<GPUTensor, DType>::MakeBinaryMaskFunc(
  const DType low, const DType high,
  const DType keep, GPUTensor<DType>* output) {}

template<typename DType>
void Backend<GPUTensor, DType>::ConstantDistributionFunc(
  const DType val, GPUTensor<DType>* output) {
  output->Fill(val);
}

template<typename DType>
void Backend<GPUTensor, DType>::NormalDistributionFunc(
  const DType loc, const DType scale,
  GPUTensor<DType>* output) {
  static unsigned int seed = 0;
  curandGenerator_t gen;
  curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
  curandSetPseudoRandomGeneratorSeed(gen, (++seed)+time(NULL));
  BlitzGenerateNormal(&gen, output->data(), output->size(),
    loc, scale);
}

template<typename DType>
__global__ void GPUUniformTransform(DType* output, DType low,
  DType high, const int size) {
  BLITZ_CUDA_LOOP(i, size) {
    output[i] = low + (high - low) * output[i];
  }
}

template<typename DType>
void Backend<GPUTensor, DType>::UniformDistributionFunc(
  const DType low, const DType high,
  GPUTensor<DType>* output) {
  static unsigned int seed = 0;
  curandGenerator_t gen;
  curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
  curandSetPseudoRandomGeneratorSeed(gen, (++seed)+time(NULL));
  BlitzGenerateUniform(&gen, output->data(), output->size());
  GPUUniformTransform<DType><<<BlitzGPUGetBlocks(output->size()),
    BLITZ_NUM_GPU_THREADS>>>(output->data(), low, high, output->size());
}

template<typename DType>
void Backend<GPUTensor, DType>::HostCopyToFunc(
  const DType* source, const size_t size, DType* target) {
  cudaMemcpy(target, source, size * sizeof(DType), cudaMemcpyHostToDevice);
}

template<typename DType>
__global__ void GPUEvaluate(const DType* output, const DType* target,
  const int dim, const int size, DType* correct) {
  BLITZ_CUDA_LOOP(i, size) {
    DType max = output[i * dim];
    size_t max_index = 0;
    for (size_t j = 1; j < dim; ++j) {
      if (max < output[i * dim + j]) {
        max_index = j;
        max = output[i * dim + j];
      }
    }

    if (target[i * dim + max_index] == (DType)1.0) {
      correct[i] = 1.0f;
    }
  }
}

template<typename DType>
float Backend<GPUTensor, DType>::EvaluateRegressFunc(
  const GPUTensor<DType>* output, const GPUTensor<DType>* target) {
  return 0;
}

template<typename DType>
float Backend<GPUTensor, DType>::EvaluateClassifyFunc(
  const GPUTensor<DType>* output, const GPUTensor<DType>* target) {
  int batch_size = output->shape()[0];
  int dim = output->size() / batch_size;
  Shape shape(1);
  shape[0] = batch_size;
  GPUTensor<DType> correct(shape);
  GPUEvaluate<DType><<<BlitzGPUGetBlocks(batch_size),
    BLITZ_NUM_GPU_THREADS>>>(output->data(), target->data(),
    dim, batch_size, correct.data());
  thrust::device_ptr<DType> rptr =
    thrust::device_pointer_cast(correct.data());
  return thrust::reduce(rptr, rptr + correct.size()) / batch_size;
}

#endif  // SRC_BACKEND_GPU_BACKEND_COMMON_INL_H_
