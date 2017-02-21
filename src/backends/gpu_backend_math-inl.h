#ifndef SRC_BACKENDS_GPU_BACKEND_COMMON_INL_H_
#define SRC_BACKENDS_GPU_BACKEND_COMMON_INL_H_

static void RectlinApplyFunc(
  const GPUTensor<DType>* input, GPUTensor<DType>* output,
  DType slope) {
  CHECK_EQ(input->size(), output->size());
  GPURectlinApply<DType><<<BlitzGPUGetBlocks(input->size()),
    BLITZ_NUM_GPU_THREADS>>>(input->data(), output->data(),
    0, slope, input->size());
}

static void RectlinDerivativeFunc(
  const GPUTensor<DType>* input, GPUTensor<DType>* output,
  DType slope) {
  CHECK_EQ(input->size(), output->size());
  GPURectlinDerivative<DType><<<BlitzGPUGetBlocks(input->size()),
    BLITZ_NUM_GPU_THREADS>>>(input->data(), output->data(),
    0, slope, input->size());
}

static void SoftmaxApplyFunc(
  const GPUTensor<DType>* input, GPUTensor<DType>* output) {
  CHECK_EQ(input->size(), output->size());
  size_t batch_size = input->shape()[0];
  size_t dim = input->size() / batch_size;
  GPUSoftmaxApply<DType><<<BlitzGPUGetBlocks(batch_size),
    BLITZ_NUM_GPU_THREADS>>>(input->data(), output->data(),
    batch_size, dim);
}

static void SoftmaxDerivativeFunc(
  const GPUTensor<DType>* input, GPUTensor<DType>* output) {
  // TODO(keren) not short cut version
}

static DType SquareMeanApplyFunc(
  const GPUTensor<DType>* input, const GPUTensor<DType>* target) {
  return 0;
}

static void SquareMeanDerivativeFunc(
  const GPUTensor<DType>* input, const GPUTensor<DType>* target,
  GPUTensor<DType>* output) {}

static DType AbsMeanApplyFunc(
  const GPUTensor<DType>* input, const GPUTensor<DType>* target) {
  return 0;
}

static void AbsMeanDerivativeFunc(
  const GPUTensor<DType>* input, const GPUTensor<DType>* target,
  GPUTensor<DType>* output) {}

static void LogisticApplyFunc(
  const GPUTensor<DType>* input, GPUTensor<DType>* output) {
  CHECK_EQ(input->size(), output->size());
  GPULogisticApply<DType><<<BlitzGPUGetBlocks(input->size()),
    BLITZ_NUM_GPU_THREADS>>>(input->data(), output->data(),
    input->size());
}

static void LogisticDerivativeFunc(
  const GPUTensor<DType>* input, GPUTensor<DType>* output) {
  CHECK_EQ(input->size(), output->size());
  // TODO(keren) not short cut version
}

static DType CrossEntropyBinaryApplyFunc(
  const GPUTensor<DType>* input, const GPUTensor<DType>* target) {
  GPUTensor<DType> sum(input->shape());
  GPUCrossEntropyBinaryApply<DType><<<BlitzGPUGetBlocks(input->size()),
    BLITZ_NUM_GPU_THREADS>>>(input->data(), target->data(),
    sum.data(), input->size());
  thrust::device_ptr<DType> sptr = thrust::device_pointer_cast(sum.data());
  DType loss = thrust::reduce(sptr, sptr + sum.size());
  return loss / input->shape()[0];
}

static void CrossEntropyBinaryDerivativeFunc(
  const GPUTensor<DType>* input, const GPUTensor<DType>* target,
  GPUTensor<DType>* output) {
  MinusFunc(input, target, output);
}

static DType CrossEntropyMultiApplyFunc(
  const GPUTensor<DType>* input, const GPUTensor<DType>* target) {
  CHECK_EQ(input->size(), target->size());
  const Shape& input_shape = input->shape();
  const Shape& target_shape = target->shape();
  GPUTensor<DType> sum(input->shape());
  GPUCrossEntropyMultiApply<DType><<<BlitzGPUGetBlocks(input->size()),
    BLITZ_NUM_GPU_THREADS>>>(input->data(), target->data(),
    sum.data(), input->size());
  thrust::device_ptr<DType> sptr = thrust::device_pointer_cast(sum.data());
  DType loss = thrust::reduce(sptr, sptr + sum.size());
  return loss / input->shape()[0];
}

static void CrossEntropyMultiDerivativeFunc(
  const GPUTensor<DType>* input, const GPUTensor<DType>* target,
  GPUTensor<DType>* output) {
  MinusFunc(input, target, output);
}

static void BiasForwardFunc(
  const GPUTensor<DType>* input, const GPUTensor<DType>* bias,
  GPUTensor<DType>* output) {
  CHECK_EQ(input->size(), output->size());
  size_t batch_size = input->shape()[0];
  size_t dim = input->size() / batch_size;
  GPUBiasApply<DType><<<BlitzGPUGetBlocks(batch_size),
    BLITZ_NUM_GPU_THREADS>>>(input->data(), bias->data(), output->data(),
    batch_size, dim);
}

static void BiasBackwardUpdateFunc(
  const GPUTensor<DType>* input, GPUTensor<DType>* update) {
  size_t batch_size = input->shape()[0];
  size_t dim = input->size() / batch_size;
  GPUBiasDerivative<DType><<<BlitzGPUGetBlocks(dim),
    BLITZ_NUM_GPU_THREADS>>>(input->data(), update->data(),
    batch_size, dim);
}

static void BatchNormForwardFunc(
  const GPUTensor<DType>* input,
  const GPUTensor<DType>* gamma,
  const GPUTensor<DType>* beta,
  GPUTensor<DType>* input_var,
  GPUTensor<DType>* input_hat,
  GPUTensor<DType>* output,
  DType epsilon) {}

static void BatchNormBackwardFunc(
  const GPUTensor<DType>* backward_input,
  const GPUTensor<DType>* forward_input_hat,
  const GPUTensor<DType>* forward_input_var,
  const GPUTensor<DType>* gamma,
  GPUTensor<DType>* gamma_update,
  GPUTensor<DType>* beta_update,
  GPUTensor<DType>* output,
  DType epsilon) {}


static void GradientdescentFunc(
  GPUTensor<DType>* weight,
  GPUTensor<DType>* gradient,
  GPUTensor<DType>* velocity,
  DType momentum_coef,
  DType learning_rate,
  DType decay,
  size_t batch_size) {
  CHECK_EQ(weight->size(), gradient->size());
  CHECK_EQ(gradient->size(), velocity->size());
  GPUGradientdescent<DType><<<BlitzGPUGetBlocks(gradient->size()),
    BLITZ_NUM_GPU_THREADS>>>(
    weight->data(), gradient->data(), velocity->data(),
    momentum_coef, learning_rate, decay,
    batch_size, gradient->size());
}

static void MatrixMultiplyFunc(
  const GPUTensor<DType>* left,
  const GPUTensor<DType>* right,
  GPUTensor<DType>* output,
  bool transa,
  bool transb,
  DType alpha,
  DType beta,
  BLITZ_ALGORITHM algorithm) {
  bool gpu_transa = left->row_major()? transa : !transa;
  bool gpu_transb = right->row_major()? transb : !transb;
  size_t dim_left = gpu_transa ? left->size() / (left->shape())[0] : (left->shape())[0];
  size_t dim_right = gpu_transb ? (right->shape())[0] : right->size() / (right->shape())[0];
  size_t dim_common_left = gpu_transa ? (left->shape())[0] : left->size() / (left->shape())[0];
  size_t dim_common_right = gpu_transb ? right->size() / (right->shape())[0] : (right->shape())[0];
  CHECK_EQ(dim_common_left, dim_common_right);
  #ifdef BLITZ_PERFORMANCE
  float elapsed_time = 0.0f;
  CUevent event_start;
  CUevent event_stop;
  BLITZ_GPU_TIMER_START(elapsed_time, event_start, event_stop);
  #endif  // BLITZ_PERFORMANCE
  switch (algorithm) {
    case BLITZ_BLAS_GEMM:
      BlitzGemm<GPUTensor, DType>(
        const_cast<GPUTensor<DType>*>(left)->data(),
        const_cast<GPUTensor<DType>*>(right)->data(),
        output->data(),
        gpu_transa, gpu_transb,
        alpha, beta,
        dim_left, dim_right, dim_common_left);
      break;
    case BLITZ_SASS_GEMM:
      BlitzSassGemm(
        const_cast<GPUTensor<DType>*>(left)->data(),
        const_cast<GPUTensor<DType>*>(right)->data(),
        output->data(),
        gpu_transa, gpu_transb,
        alpha, beta,
        dim_left, dim_right, dim_common_left);
      break;
    default:
      LOG(FATAL) << "Unsupported algorithm type: " << algorithm;
      break;
  }
  #ifdef BLITZ_PERFORMANCE
  double computations = static_cast<double>(2 * dim_right) * static_cast<double>(dim_left) * static_cast<double>(dim_common_left);
  BLITZ_GPU_TIMER_END(elapsed_time, event_start, event_stop);
  BLITZ_GPU_TIMER_INFO(computations, elapsed_time);
  #endif  // BLITZ_PERFORMANCE
}

static void Transpose2DFunc(
  const GPUTensor<DType>* input, GPUTensor<DType>* output) {
  size_t dim_left = input->shape()[0];
  size_t dim_right = input->shape()[1];
  CHECK_EQ(dim_left, output->shape()[1]);
  CHECK_EQ(dim_right, output->shape()[0]);
  BlitzGPUTrans(const_cast<DType*>(input->data()),
    output->data(), dim_left, dim_right); 
}

static void MaximumFunc(
  const GPUTensor<DType>* left, const GPUTensor<DType>* right,
  GPUTensor<DType>* output) {}

static void MinusFunc(
  const GPUTensor<DType>* left, const GPUTensor<DType>* right,
  GPUTensor<DType>* output) {
  CHECK_EQ(left->size(), right->size());
  CHECK_EQ(right->size(), output->size());
  thrust::device_ptr<DType> lptr = thrust::device_pointer_cast(
    const_cast<DType*>(left->data()));
  thrust::device_ptr<DType> rptr = thrust::device_pointer_cast(
    const_cast<DType*>(right->data()));
  thrust::device_ptr<DType> optr = thrust::device_pointer_cast(
    output->data());
  thrust::transform(lptr, lptr + left->size(),
    rptr, optr, thrust::minus<DType>()); 
}

static DType SumFunc(
  const GPUTensor<DType>* input) {
  thrust::device_ptr<DType> ptr = thrust::device_pointer_cast(
    const_cast<DType*>(input->data()));
  return thrust::reduce(ptr, ptr + input->size());
}

static void AddFunc(
  const GPUTensor<DType>* left, const GPUTensor<DType>* right,
  GPUTensor<DType>* output) {}

static void MultiplyFunc(
  const GPUTensor<DType>* left, const GPUTensor<DType>* right,
  GPUTensor<DType>* output) {
  CHECK_EQ(left->size(), right->size());
  CHECK_EQ(right->size(), output->size());
  thrust::device_ptr<DType> lptr = thrust::device_pointer_cast(
    const_cast<DType*>(left->data()));
  thrust::device_ptr<DType> rptr = thrust::device_pointer_cast(
    const_cast<DType*>(right->data()));
  thrust::device_ptr<DType> optr = thrust::device_pointer_cast(
    output->data());
  thrust::transform(lptr, lptr + left->size(),
    rptr, optr, thrust::multiplies<DType>()); 
}

static void MultiplyFunc(
  const GPUTensor<DType>* left, GPUTensor<DType>* output,
  DType right) {
  GPUTensor<DType> right_vector(left->shape());
  right_vector.Fill(right);
  MultiplyFunc(left, &right_vector, output);
}

static void MakeBinaryMaskFunc(
  GPUTensor<DType>* output,
  DType low,
  DType high,
  DType keep) {
  UniformDistributionFunc(output, low, high);
  GPUMakeBinaryMask<DType><<<BlitzGPUGetBlocks(output->size()),
    BLITZ_NUM_GPU_THREADS>>>(output->data(), keep, output->size());
}

static void ConstantDistributionFunc(
  GPUTensor<DType>* output, DType val) {
  output->Fill(val);
}

static void NormalDistributionFunc(
  GPUTensor<DType>* output, DType loc, DType scale) {
  static unsigned int seed = 0;
  curandGenerator_t gen;
  curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
  curandSetPseudoRandomGeneratorSeed(gen, (++seed)+time(NULL));
  BlitzGenerateNormal(&gen, output->data(), loc, scale, output->size());
}

static void UniformDistributionFunc(
  GPUTensor<DType>* output, DType low, DType high) {
  static unsigned int seed = 0;
  curandGenerator_t gen;
  curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
  curandSetPseudoRandomGeneratorSeed(gen, (++seed)+time(NULL));
  BlitzGenerateUniform(&gen, output->data(), output->size());
  GPUUniformTransform<DType><<<BlitzGPUGetBlocks(output->size()),
    BLITZ_NUM_GPU_THREADS>>>(output->data(), low, high, output->size());
}

static void HostCopyToFunc(
  const DType* source, DType* target, size_t size) {
  cudaMemcpy(target, source, size * sizeof(DType), cudaMemcpyHostToDevice);
}

static float EvaluateRegressFunc(
  const GPUTensor<DType>* output, const GPUTensor<DType>* target) {
  return 0;
}

static float EvaluateClassifyFunc(
  const GPUTensor<DType>* output, const GPUTensor<DType>* target) {
  size_t batch_size = output->shape()[0];
  size_t dim = output->size() / batch_size;
  Shape shape(1);
  shape[0] = batch_size;
  GPUTensor<DType> correct(shape);
  GPUEvaluateClass<DType><<<BlitzGPUGetBlocks(batch_size),
    BLITZ_NUM_GPU_THREADS>>>(
    output->data(), target->data(), correct.data(),
    dim, batch_size);
  thrust::device_ptr<DType> rptr =
    thrust::device_pointer_cast(correct.data());
  return thrust::reduce(rptr, rptr + correct.size()) / batch_size;
}

#endif  // SRC_BACKENDS_GPU_BACKEND_COMMON_INL_H_
