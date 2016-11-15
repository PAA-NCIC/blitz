#ifndef SRC_BACKENDS_GPU_BACKEND_COMMON_INL_H_
#define SRC_BACKENDS_GPU_BACKEND_COMMON_INL_H_

template<typename DType>
__global__ void GPURectlinApply(
  const DType* input, DType* output,
  DType compare_value, DType slope,
  size_t size) {
  BLITZ_CUDA_LOOP(i, size) {
    DType greater = input[i] > compare_value ? input[i] : compare_value;
    DType less = input[i] <= compare_value ?
      slope * input[i] : slope * compare_value;
    output[i] = greater + less;
  }
}

template<typename DType>
void Backend<GPUTensor, DType>::RectlinApplyFunc(
  const GPUTensor<DType>* input, GPUTensor<DType>* output,
  DType slope) {
  CHECK_EQ(input->size(), output->size());

  DType compara_value = static_cast<DType>(0);
  GPURectlinApply<DType><<<BlitzGPUGetBlocks(input->size()),
    BLITZ_NUM_GPU_THREADS>>>(input->data(), output->data(),
    compara_value, slope, input->size());
}

template<typename DType>
__global__ void GPURectlinDerivative(
  const DType* input, DType* output,
  DType compare_value, DType slope,
  size_t size) {
  BLITZ_CUDA_LOOP(i, size) {
    DType greater = input[i] > compare_value ? 1.0 : 0.0;
    DType less = input[i] <= compare_value ? slope : 0.0;
    output[i] = (greater + less) * output[i];
  }
}

template<typename DType>
void Backend<GPUTensor, DType>::RectlinDerivativeFunc(
  const GPUTensor<DType>* input, GPUTensor<DType>* output,
  DType slope) {
  CHECK_EQ(input->size(), output->size());
  DType compara_value = static_cast<DType>(0);
  GPURectlinDerivative<DType><<<BlitzGPUGetBlocks(input->size()),
    BLITZ_NUM_GPU_THREADS>>>(input->data(), output->data(),
    compara_value, slope, input->size());
}

template<typename DType>
__global__ void GPUSoftmaxApply(
  const DType* input, DType* output,
  size_t num_sample, size_t dim) {
  BLITZ_CUDA_LOOP(i, num_sample) {
    DType sum = 0; 
    for (size_t j = 0; j < dim; ++j) {
      size_t index = i * dim + j;
      output[index] = exp(input[index]);
      sum += output[index];
    }
    for (size_t j = 0; j < dim; ++j) {
      output[i * dim + j] /= sum;
    }
  }
}

template<typename DType>
void Backend<GPUTensor, DType>::SoftmaxApplyFunc(
  const GPUTensor<DType>* input, GPUTensor<DType>* output) {
  CHECK_EQ(input->size(), output->size());
  size_t num_sample = input->shape()[0];
  size_t dim = input->size() / num_sample;
  GPUSoftmaxApply<DType><<<BlitzGPUGetBlocks(num_sample),
    BLITZ_NUM_GPU_THREADS>>>(input->data(), output->data(),
    num_sample, dim);
}

template<typename DType>
void Backend<GPUTensor, DType>::SoftmaxDerivativeFunc(
  const GPUTensor<DType>* input, GPUTensor<DType>* output) {
  // TODO(keren) not short cut version
}

template<typename DType>
DType Backend<GPUTensor, DType>::SquareMeanApplyFunc(
  const GPUTensor<DType>* input, const GPUTensor<DType>* target) {
  return 0;
}

template<typename DType>
void Backend<GPUTensor, DType>::SquareMeanDerivativeFunc(
  const GPUTensor<DType>* input, const GPUTensor<DType>* target,
  GPUTensor<DType>* output) {}

template<typename DType>
DType Backend<GPUTensor, DType>::AbsMeanApplyFunc(
  const GPUTensor<DType>* input, const GPUTensor<DType>* target) {
  return 0;
}

template<typename DType>
void Backend<GPUTensor, DType>::AbsMeanDerivativeFunc(
  const GPUTensor<DType>* input, const GPUTensor<DType>* target,
  GPUTensor<DType>* output) {}

template<typename DType>
__global__ void GPULogisticApply(const DType* input, DType* output, size_t size) {
  BLITZ_CUDA_LOOP(i, size) {
    output[i] = 1 / (exp(-input[i]) + 1);
  }
}

template<typename DType>
void Backend<GPUTensor, DType>::LogisticApplyFunc(
  const GPUTensor<DType>* input, GPUTensor<DType>* output) {
  CHECK_EQ(input->size(), output->size());
  GPULogisticApply<DType><<<BlitzGPUGetBlocks(input->size()),
    BLITZ_NUM_GPU_THREADS>>>(input->data(), output->data(),
    input->size());
}

template<typename DType>
void Backend<GPUTensor, DType>::LogisticDerivativeFunc(
  const GPUTensor<DType>* input, GPUTensor<DType>* output) {
  CHECK_EQ(input->size(), output->size());
  // TODO(keren) not short cut version
}

template<typename DType>
__global__ void GPUCrossEntropyBinaryApply(
  const DType* input, const DType* target,
  DType* sum, size_t size) {
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
  GPUCrossEntropyBinaryApply<DType><<<BlitzGPUGetBlocks(input->size()),
    BLITZ_NUM_GPU_THREADS>>>(input->data(), target->data(),
    sum.data(), input->size());
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
__global__ void GPUCrossEntropyMultiApply(
  const DType* input, const DType* target,
  DType* sum, size_t size) {
  BLITZ_CUDA_LOOP(i, size) {
    sum[i] = -BlitzGPUSafeLog(input[i]) * target[i];
  }
}

template<typename DType>
DType Backend<GPUTensor, DType>::CrossEntropyMultiApplyFunc(
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

template<typename DType>
void Backend<GPUTensor, DType>::CrossEntropyMultiDerivativeFunc(
  const GPUTensor<DType>* input, const GPUTensor<DType>* target,
  GPUTensor<DType>* output) {
  MinusFunc(input, target, output);
}

template<typename DType>
__global__ void GPUBiasForward(
  const DType* input, const DType* bias, DType* output,
  size_t num_sample, size_t dim) {
  BLITZ_CUDA_LOOP(i, num_sample) {
    for (size_t j = 0; j < dim; ++j) {
      output[i * dim + j] = input[i * dim + j] + bias[j];
    }
  }
}

template<typename DType>
void Backend<GPUTensor, DType>::BiasForwardFunc(
  const GPUTensor<DType>* input, const GPUTensor<DType>* bias,
  GPUTensor<DType>* output) {
  CHECK_EQ(input->size(), output->size());
  size_t num_sample = input->shape()[0];
  size_t dim = input->size() / num_sample;
  GPUBiasForward<DType><<<BlitzGPUGetBlocks(num_sample),
    BLITZ_NUM_GPU_THREADS>>>(input->data(), bias->data(), output->data(),
    num_sample, dim);
}

template<typename DType>
__global__ void GPUBiasBackwardUpdate(const DType* input, DType* update,
  size_t dim, size_t num_sample) {
  BLITZ_CUDA_LOOP(i, dim) {
    for (size_t j = 0; j < num_sample; ++j) {
      update[i] += input[j * dim + i];
    }
  }
}

template<typename DType>
void Backend<GPUTensor, DType>::BiasBackwardUpdateFunc(
  const GPUTensor<DType>* input, GPUTensor<DType>* update) {
  size_t num_sample = input->shape()[0];
  size_t dim = input->size() / num_sample;
  GPUBiasBackwardUpdate<DType><<<BlitzGPUGetBlocks(dim),
    BLITZ_NUM_GPU_THREADS>>>(input->data(), update->data(),
    num_sample, dim);
}

template<typename DType>
void Backend<GPUTensor, DType>::BatchNormForwardFunc(
  const GPUTensor<DType>* input,
  const GPUTensor<DType>* gamma,
  const GPUTensor<DType>* beta,
  GPUTensor<DType>* input_var,
  GPUTensor<DType>* input_hat,
  GPUTensor<DType>* output,
  DType epsilon) {}

template<typename DType>
void Backend<GPUTensor, DType>::BatchNormBackwardFunc(
  const GPUTensor<DType>* backward_input,
  const GPUTensor<DType>* forward_input_hat,
  const GPUTensor<DType>* forward_input_var,
  const GPUTensor<DType>* gamma,
  GPUTensor<DType>* gamma_update,
  GPUTensor<DType>* beta_update,
  GPUTensor<DType>* output,
  DType epsilon) {}

template<typename DType>
__global__ void GPUGradientdescent(
  DType* weight, DType* gradient, DType* velocity,
  DType momentum_coef, DType learning_rate,
  DType decay, size_t batch_size, size_t size) {
  BLITZ_CUDA_LOOP(i, size) {
    gradient[i] /= batch_size;
    velocity[i] = velocity[i] * momentum_coef - learning_rate *
      (gradient[i] + decay * weight[i]);
    weight[i] = weight[i] + velocity[i];
  }
}

template<typename DType>
void Backend<GPUTensor, DType>::GradientdescentFunc(
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

template<typename DType>
void Backend<GPUTensor, DType>::MatrixMultiplyFunc(
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
  size_t dim_left = gpu_transa ? left->size() / (left->shape())[0] :
    (left->shape())[0];
  size_t dim_right = gpu_transb ? (right->shape())[0] :
    right->size() / (right->shape())[0];
  size_t dim_common_left = gpu_transa ? (left->shape())[0] :
    left->size() / (left->shape())[0];
  size_t dim_common_right = gpu_transb ? right->size() / (right->shape())[0] :
    (right->shape())[0];
  CHECK_EQ(dim_common_left, dim_common_right);
  CHECK_NE(dim_left, 0);
  CHECK_NE(dim_common_right, 0);
  CHECK_NE(dim_right, 0);
#ifdef BLITZ_PERFORMANCE
  float elapsed_time = 0.0f;
  CUevent event_start, event_stop;
  cuEventCreate(&event_start, CU_EVENT_BLOCKING_SYNC);
  cuEventCreate(&event_stop, CU_EVENT_BLOCKING_SYNC);
  cuEventRecord(event_start, NULL);
#endif  // BLITZ_PERFORMANCE
	switch (algorithm) {
		case BLITZ_BLAS_GEMM:
			BlitzGPUGemm(
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
  cuEventRecord(event_stop, NULL);
  cuEventSynchronize(event_stop);
  cuEventElapsedTime(&elapsed_time, event_start, event_stop);
  cuEventCreate(&event_stop, CU_EVENT_BLOCKING_SYNC);
  double computations = 2 * static_cast<double>(dim_right) *
    static_cast<double>(dim_left) * static_cast<double>(dim_common_left);
  LOG(INFO) << "GEMM time: " << elapsed_time / 1000.0;
  LOG(INFO) << "GEMM computations: " << computations;
  LOG(INFO) << "GEMM gflops: " << computations / (elapsed_time * 1e6);
  cuEventDestroy(event_start);
  cuEventDestroy(event_stop);
#endif  // BLITZ_PERFORMANCE
}

template<typename DType>
void Backend<GPUTensor, DType>::Transpose2DFunc(
  const GPUTensor<DType>* input, GPUTensor<DType>* output) {
  size_t dim_left = input->shape()[0];
  size_t dim_right = input->shape()[1];
  CHECK_EQ(dim_left, output->shape()[1]);
  CHECK_EQ(dim_right, output->shape()[0]);
  BlitzGPUTrans(const_cast<DType*>(input->data()),
    output->data(), dim_left, dim_right); 
}

template<typename DType>
void Backend<GPUTensor, DType>::MaximumFunc(
  const GPUTensor<DType>* left, const GPUTensor<DType>* right,
  GPUTensor<DType>* output) {}

template<typename DType>
void Backend<GPUTensor, DType>::MinusFunc(
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

template<typename DType>
DType Backend<GPUTensor, DType>::SumFunc(
  const GPUTensor<DType>* input) {
  thrust::device_ptr<DType> ptr = thrust::device_pointer_cast(
    const_cast<DType*>(input->data()));
  return thrust::reduce(ptr, ptr + input->size());
}

template<typename DType>
void Backend<GPUTensor, DType>::AddFunc(
  const GPUTensor<DType>* left, const GPUTensor<DType>* right,
  GPUTensor<DType>* output) {}

template<typename DType>
void Backend<GPUTensor, DType>::MultiplyFunc(
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

template<typename DType>
void Backend<GPUTensor, DType>::MultiplyFunc(
  const GPUTensor<DType>* left, GPUTensor<DType>* output,
  DType right) {
  GPUTensor<DType> right_vector(left->shape());
  right_vector.Fill(right);
  MultiplyFunc(left, &right_vector, output);
}

template<typename DType>
__global__ void GPUMakeBinaryMask(DType* output, DType keep, size_t size) {
  BLITZ_CUDA_LOOP(i, size) {
    if (output[i] < keep) {
      output[i] = DType(1);
    } else {
      output[i] = DType(0);
    }
  }
}

template<typename DType>
void Backend<GPUTensor, DType>::MakeBinaryMaskFunc(
  GPUTensor<DType>* output,
  DType low,
  DType high,
  DType keep) {
  UniformDistributionFunc(output, low, high);
  GPUMakeBinaryMask<DType><<<BlitzGPUGetBlocks(output->size()),
    BLITZ_NUM_GPU_THREADS>>>(output->data(), keep, output->size());
}

template<typename DType>
void Backend<GPUTensor, DType>::ConstantDistributionFunc(
  GPUTensor<DType>* output, DType val) {
  output->Fill(val);
}

template<typename DType>
void Backend<GPUTensor, DType>::NormalDistributionFunc(
  GPUTensor<DType>* output, DType loc, DType scale) {
  static unsigned int seed = 0;
  curandGenerator_t gen;
  curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
  curandSetPseudoRandomGeneratorSeed(gen, (++seed)+time(NULL));
  BlitzGenerateNormal(&gen, output->data(), loc, scale, output->size());
}

template<typename DType>
__global__ void GPUUniformTransform(DType* output, DType low, DType high,
  size_t size) {
  BLITZ_CUDA_LOOP(i, size) {
    output[i] = low + (high - low) * output[i];
  }
}

template<typename DType>
void Backend<GPUTensor, DType>::UniformDistributionFunc(
  GPUTensor<DType>* output, DType low, DType high) {
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
  const DType* source, DType* target, size_t size) {
  cudaMemcpy(target, source, size * sizeof(DType), cudaMemcpyHostToDevice);
}

template<typename DType>
__global__ void GPUEvaluateClass(
  const DType* output, const DType* target, DType* correct,
  size_t dim, size_t size) {
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
