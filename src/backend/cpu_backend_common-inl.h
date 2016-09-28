#ifndef SRC_BACKEND_CPU_BACKEND_COMMON_INL_H_
#define SRC_BACKEND_CPU_BACKEND_COMMON_INL_H_

template<typename DType>
void Backend<CPUTensor, DType>::RectlinApplyFunc(
  const CPUTensor<DType>* input,
  const DType slope,
  CPUTensor<DType>* output) {
  CHECK_EQ(input->size(), output->size());

  DType compare_value = static_cast<DType>(0);
#ifdef BLITZ_AVX
  size_t avx_width = BLITZ_AVX_WIDTH / sizeof(DType);
  size_t remain = input->size() % avx_width;
  BlitzAVXReg<DType> compare_value_reg, input_reg, output_reg,
    slope_reg, left_reg, right_reg;
  BlitzAVXBroadcast<DType>(&slope, &slope_reg);
  BlitzAVXBroadcast<DType>(&compare_value, &compare_value_reg);
  for (size_t i = 0; i < remain; ++i) {
    (*output)[i] = std::max((*input)[i], compare_value) +
      slope * std::min((*input)[i], compare_value);
  }
  #pragma omp parallel for private(input_reg, output_reg, left_reg, \
    right_reg)
  for (size_t i = remain; i < input->size(); i += avx_width) {
    BlitzAVXLoad<DType>(input->data() + i, &input_reg);
    BlitzAVXMax<DType>(&input_reg, &compare_value_reg, &left_reg);
    BlitzAVXMin<DType>(&input_reg, &compare_value_reg, &right_reg);
    output_reg.v = left_reg.v + slope_reg.v * right_reg.v;
    BlitzAVXStore<DType>(output->data() + i, &output_reg);
  }

#else
  #pragma omp parallel for
  for (size_t i = 0; i < input->size(); ++i) {
    (*output)[i] = std::max((*input)[i], compare_value) +
      slope * std::min((*input)[i], compare_value);
  }
#endif
}

template<typename DType>
void Backend<CPUTensor, DType>::RectlinDerivativeFunc(
  const CPUTensor<DType>* input,
  const DType slope,
  CPUTensor<DType>* output) {
  CHECK_EQ(input->size(), output->size());

  DType compare_value = static_cast<DType>(0);
  #pragma omp parallel for
  for (size_t i = 0; i < input->size(); ++i) {
    DType greater = (*input)[i] > compare_value ? 1.0 : 0.0;
    DType less = (*input)[i] <= compare_value ? slope : 0.0;
    (*output)[i] = (greater + less) * (*output)[i];
  }
}

template<typename DType>
void Backend<CPUTensor, DType>::LogisticApplyFunc(
  const CPUTensor<DType>* input,
  CPUTensor<DType>* output) {
  CHECK_EQ(input->size(), output->size());

  #pragma omp parallel for
  for (size_t i = 0; i < input->size(); ++i) {
    (*output)[i] = 1 / (exp(-(*input)[i]) + 1);
  }
}

template<typename DType>
void Backend<CPUTensor, DType>::LogisticDerivativeFunc(
  const CPUTensor<DType>* input,
  CPUTensor<DType>* output) {
  CHECK_EQ(input->size(), output->size());
  // TODO(keren) not short cut version
}

template<typename DType>
void Backend<CPUTensor, DType>::SoftmaxApplyFunc(
  const CPUTensor<DType>* input,
  CPUTensor<DType>* output) {
  CHECK_EQ(input->size(), output->size());
  size_t num_sample = input->shape()[0];
  size_t dim = input->size() / num_sample;

  #pragma omp parallel for
  for (size_t i = 0; i < num_sample; ++i) {
    DType sum = 0;
    for (size_t j = 0; j < dim; ++j) {
      size_t index = i * dim + j;
      (*output)[index] = exp((*input)[index]);
      sum += (*output)[index];
    }

    for (size_t j = 0; j < dim; ++j) {
      (*output)[i * dim + j] /= sum;
    }
  }
}

template<typename DType>
void Backend<CPUTensor, DType>::SoftmaxDerivativeFunc(
  const CPUTensor<DType>* input,
  CPUTensor<DType>* output) {
  CHECK_EQ(input->size(), output->size());
  // TODO(keren) not short cut version
}

template<typename DType>
DType Backend<CPUTensor, DType>::CrossEntropyBinaryApplyFunc(
  const CPUTensor<DType>* input,
  const CPUTensor<DType>* target) {
  CHECK_EQ(input->size(), target->size());
  DType private_output = 0;
  DType output = 0;
  #pragma omp parallel firstprivate(private_output)
  {
    #pragma omp for
    for (size_t i = 0; i < input->size(); ++i) {
      private_output += -BlitzCPUSafeLog((*input)[i]) * (*target)[i] -
      BlitzCPUSafeLog(1 - (*input)[i]) * (1 - (*target)[i]);
    }

    #pragma omp atomic
    output += private_output;
  }

  output /= (input->shape())[0];
  return output;
}

template<typename DType>
DType Backend<CPUTensor, DType>::SquareMeanApplyFunc(
  const CPUTensor<DType>* input, const CPUTensor<DType>* target) {
  CHECK_EQ(input->size(), target->size());
  DType sum_square = 0;
  DType sum = 0;

  size_t batch_size = input->shape()[0];
  size_t dim = input->size() / batch_size;
  #pragma omp parallel firstprivate(sum_square)
  {
    #pragma omp for
    for (size_t i = 0; i < batch_size; ++i) {
      for (size_t j = 0; j < dim; ++j) {
        sum_square += pow((*input)[i * dim + j] -
          (*target)[i * dim + j], 2);
      }
    }

    #pragma omp atomic
    sum += sum_square;
  }
  return sum / (2 * batch_size);
}

template<typename DType>
void Backend<CPUTensor, DType>::SquareMeanDerivativeFunc(
  const CPUTensor<DType>* input, const CPUTensor<DType>* target,
  CPUTensor<DType>* output) {
  CHECK_EQ(input->size(), target->size());
  CHECK_EQ(output->size(), target->size());
  MinusFunc(input, target, output);
}

template<typename DType>
DType Backend<CPUTensor, DType>::AbsMeanApplyFunc(
  const CPUTensor<DType>* input,
  const CPUTensor<DType>* target) {
  CHECK_EQ(input->size(), target->size());
  DType sum_abs = 0;
  DType sum = 0;

  size_t batch_size = input->shape()[0];
  size_t dim = input->size() / batch_size;
  #pragma omp parallel firstprivate(sum_abs)
  {
    #pragma omp for
    for (size_t i = 0; i < batch_size; ++i) {
      for (size_t j = 0; j < dim; ++j) {
        sum_abs += fabs((*input)[i * dim + j] -
          (*target)[i * dim + j]);
      }
    }

    #pragma omp atomic
    sum += sum_abs;
  }
  return sum / batch_size;
}

template<typename DType>
void Backend<CPUTensor, DType>::AbsMeanDerivativeFunc(
  const CPUTensor<DType>* input, const CPUTensor<DType>* target,
  CPUTensor<DType>* output) {
  CHECK_EQ(input->size(), target->size());
  CHECK_EQ(output->size(), target->size());
  size_t batch_size = input->shape()[0];
  size_t dim = input->size() / batch_size;
  #pragma omp parallel for
  for (size_t i = 0; i < batch_size; ++i) {
    for (size_t j = 0; j < dim; ++j) {
      if ((*input)[i * dim + j] > (*target)[i * dim + j]) {
        (*output)[i * dim + j] = 1;
      } else if ((*input)[i * dim + j] < (*target)[i * dim + j]) {
        (*output)[i * dim + j] = -1;
      } else {
        (*output)[i * dim + j] = 0;
      }
    }
  }
}

template<typename DType>
void Backend<CPUTensor, DType>::CrossEntropyBinaryDerivativeFunc(
  const CPUTensor<DType>* input, const CPUTensor<DType>* target,
  CPUTensor<DType>* output) {
  MinusFunc(input, target, output);
}

template<typename DType>
DType Backend<CPUTensor, DType>::CrossEntropyMultiApplyFunc(
  const CPUTensor<DType>* input, const CPUTensor<DType>* target) {
  CHECK_EQ(input->size(), target->size());
  DType private_output = 0;
  DType output = 0;

  #pragma omp parallel firstprivate(private_output)
  {
    #pragma omp for
    for (size_t i = 0; i < input->size(); ++i) {
      private_output += BlitzCPUSafeLog((*input)[i]) * (*target)[i];
    }

    #pragma omp atomic
    output += -private_output;
  }

  output /= (input->shape())[0];
  return output;
}

template<typename DType>
void Backend<CPUTensor, DType>::CrossEntropyMultiDerivativeFunc(
  const CPUTensor<DType>* input, const CPUTensor<DType>* target,
  CPUTensor<DType>* output) {
  MinusFunc(input, target, output);
}

template<typename DType>
void Backend<CPUTensor, DType>::BiasForwardFunc(
  const CPUTensor<DType>* input, const CPUTensor<DType>* bias,
  CPUTensor<DType>* output) {
  CHECK_EQ(input->size(), output->size());
  size_t num_sample = input->shape()[0];
  size_t dim = input->size() / num_sample;

  #pragma omp parallel for
  for (size_t i = 0; i < num_sample; ++i) {
    for (size_t j = 0; j < dim; ++j) {
      (*output)[i * dim + j] = (*input)[i * dim + j] + (*bias)[j];
    }
  }
}

template<typename DType>
void Backend<CPUTensor, DType>::BiasBackwardUpdateFunc(
  const CPUTensor<DType>* input, CPUTensor<DType>* update) {
  size_t num_sample = input->shape()[0];
  size_t dim = input->size() / num_sample;
  #pragma omp parallel for
  for (size_t i = 0; i < dim; ++i) {
    for (size_t j = 0; j < num_sample; ++j) {
      (*update)[i] += (*input)[j * dim + i];
    }
  }
}

template<typename DType>
void Backend<CPUTensor, DType>::BatchNormForwardFunc(
  const CPUTensor<DType>* input, const CPUTensor<DType>* gamma,
  const CPUTensor<DType>* beta, const DType epsilon,
  CPUTensor<DType>* input_var, CPUTensor<DType>* input_hat,
  CPUTensor<DType>* output) {
  size_t num_sample = input->shape()[0];
  size_t dim = input->size() / num_sample;
  input_var->Fill(0);
  input_hat->Fill(0);

  #pragma omp parallel for
  for (size_t i = 0; i < dim; ++i) {
    DType mean = 0.0;
    for (size_t j = 0; j < num_sample; ++j) {
      mean += (*input)[j * dim + i];
    }
    mean /= num_sample;

    DType var;
    for (size_t j = 0; j < num_sample; ++j) {
      var = (*input)[j * dim + i] - mean;
      (*input_var)[i] += var * var;
    }
    (*input_var)[i] /= num_sample;

    DType divider = sqrt((*input_var)[i] + epsilon);

    size_t index = 0;
    for (size_t j = 0; j < num_sample; ++j) {
      index = j * dim + i;
      (*input_hat)[index] = ((*input)[index] -
        mean) / divider;
      (*output)[index] = (*gamma)[i] * (*input_hat)[index] +
        (*beta)[i];
    }
  }
}

template<typename DType>
void Backend<CPUTensor, DType>::BatchNormBackwardFunc(
  const CPUTensor<DType>* backward_input,
  const CPUTensor<DType>* forward_input_hat,
  const CPUTensor<DType>* forward_input_var,
  const CPUTensor<DType>* gamma, const DType epsilon,
  CPUTensor<DType>* gamma_update, CPUTensor<DType>* beta_update,
  CPUTensor<DType>* output) {
  size_t num_sample = backward_input->shape()[0];
  size_t dim = backward_input->size() / num_sample;

  #pragma omp parallel for
  for (size_t i = 0; i < dim; ++i) {
    for (size_t j = 0; j < num_sample; ++j) {
      const size_t index = j * dim + i;
      (*gamma_update)[i] += (*forward_input_hat)[index] *
        (*backward_input)[index];
      (*beta_update)[i] += (*backward_input)[index];
    }

    DType xhat;
    for (size_t j = 0; j < num_sample; ++j) {
      const size_t index = j * dim + i;
      xhat = ((*forward_input_hat)[index] * (*gamma_update)[i] +
        (*beta_update)[i]) / num_sample;
      (*output)[index] = (*gamma)[i] * ((*backward_input)[index] -
        xhat) / sqrt((*forward_input_var)[i] + epsilon);
    }
  }
}

template<typename DType>
void Backend<CPUTensor, DType>::GradientdescentFunc(
  const DType momentum_coef, const DType learning_rate,
  const DType decay, size_t batch_size,
  CPUTensor<DType>* weight,
  CPUTensor<DType>* gradient,
  CPUTensor<DType>* velocity) {
  CHECK_EQ(weight->size(), gradient->size());
  CHECK_EQ(gradient->size(), velocity->size());
#ifdef BLITZ_DEVELOP
  LOG(INFO) << "weight size: " << weight->size();
  LOG(INFO) << "momentum_coef: " << momentum_coef;
  LOG(INFO) << "learning_rate: " << learning_rate;
  LOG(INFO) << "decay: " << decay;
  LOG(INFO) << "batch_size: " << batch_size;
#endif

#ifdef BLITZ_AVX
  size_t avx_width = BLITZ_AVX_WIDTH / sizeof(DType);
  size_t remain = velocity->size() % avx_width;
  const DType batch_size_float = static_cast<DType>(batch_size);
  BlitzAVXReg<DType> gradient_reg, batch_size_reg, velocity_reg,
    momentum_coef_reg, learning_rate_reg, decay_reg, weight_reg;
  BlitzAVXBroadcast<DType>(&batch_size_float, &batch_size_reg);
  BlitzAVXBroadcast<DType>(&momentum_coef, &momentum_coef_reg);
  BlitzAVXBroadcast<DType>(&learning_rate, &learning_rate_reg);
  BlitzAVXBroadcast<DType>(&decay, &decay_reg);
  for (size_t i = 0; i < remain; ++i) {
    (*gradient)[i] /= batch_size;
    (*velocity)[i] = (*velocity)[i] * momentum_coef - learning_rate *
      ((*gradient)[i] + decay * (*weight)[i]);
    (*weight)[i] = (*weight)[i] + (*velocity)[i];
  }
  #pragma omp parallel for private(gradient_reg, velocity_reg, weight_reg)
  for (size_t i = remain; i < velocity->size(); i += avx_width) {
    BlitzAVXLoad<DType>(gradient->data() + i, &gradient_reg);
    BlitzAVXLoad<DType>(velocity->data() + i, &velocity_reg);
    BlitzAVXLoad<DType>(weight->data() + i, &weight_reg);
    gradient_reg.v = gradient_reg.v / batch_size_reg.v;
    velocity_reg.v = velocity_reg.v * momentum_coef_reg.v -
      learning_rate_reg.v * (gradient_reg.v + decay_reg.v *
      weight_reg.v);
    weight_reg.v = weight_reg.v + velocity_reg.v;
    BlitzAVXStore<DType>(gradient->data() + i, &gradient_reg);
    BlitzAVXStore<DType>(velocity->data() + i, &velocity_reg);
    BlitzAVXStore<DType>(weight->data() + i, &weight_reg);
  }
#else
  #pragma omp parallel for
  for (size_t i = 0; i < velocity->size(); ++i) {
    (*gradient)[i] /= batch_size;
    (*velocity)[i] = (*velocity)[i] * momentum_coef - learning_rate *
      ((*gradient)[i] + decay * (*weight)[i]);
    (*weight)[i] = (*weight)[i] + (*velocity)[i];
  }
#endif
}

template<typename DType>
void Backend<CPUTensor, DType>::MatrixDotFunc(
  const CPUTensor<DType>* left, const CPUTensor<DType>* right,
  const bool transa, const bool transb,
  const DType alpha, const DType beta,
  CPUTensor<DType>* output, const string& kernel) {
  size_t dim_left = transa ? left->size() / (left->shape())[0] :
    (left->shape())[0];
  size_t dim_right = transb ? (right->shape())[0] :
    right->size() / (right->shape())[0];
  size_t dim_common_left = transa ? (left->shape())[0] :
    left->size() / (left->shape())[0];
  size_t dim_common_right = transb ? right->size() / (right->shape())[0] :
    (right->shape())[0];
  CHECK_EQ(dim_common_left, dim_common_right);
#ifdef BLITZ_DEVELOP
  LOG(INFO) << "dim left: " << dim_left;
  LOG(INFO) << "dim common: " << dim_common_left;
  LOG(INFO) << "dim right: " << dim_right;
#endif
  BlitzCPUGemm(transa, transb, dim_left, dim_right, dim_common_left,
    const_cast<CPUTensor<DType>*>(left)->data(),
    const_cast<CPUTensor<DType>*>(right)->data(),
    output->data(), alpha, beta);
}

template<typename DType>
void Backend<CPUTensor, DType>::MaximumFunc(
  const CPUTensor<DType>* left, const CPUTensor<DType>* right,
  CPUTensor<DType>* output) {
  CHECK_EQ(left->size(), right->size());
  CHECK_EQ(right->size(), output->size());
  #pragma omp parallel for
  for (size_t i = 0; i < left->size(); ++i) {
    (*output)[i] = std::max((*left)[i], (*right)[i]);
  }
}

template<typename DType>
void Backend<CPUTensor, DType>::MaximumFunc(
  const CPUTensor<DType>* left,
  const DType right,
  CPUTensor<DType>* output) {
  CHECK_EQ(left->size(), output->size());
  #pragma omp parallel for
  for (size_t i = 0; i < left->size(); ++i) {
    (*output)[i] = std::max((*left)[i], right);
  }
}

template<typename DType>
void Backend<CPUTensor, DType>::MinusFunc(
  const CPUTensor<DType>* left, const CPUTensor<DType>* right,
  CPUTensor<DType>* output) {
  CHECK_EQ(left->size(), right->size());
  CHECK_EQ(right->size(), output->size());

  #pragma omp parallel for
  for (size_t i = 0; i < left->size(); ++i) {
    (*output)[i] = (*left)[i] - (*right)[i];
  }
}

template<typename DType>
void Backend<CPUTensor, DType>::MinusFunc(
  const CPUTensor<DType>* left,
  const DType right,
  CPUTensor<DType>* output) {
  CHECK_EQ(left->size(), output->size());
  #pragma omp parallel for
  for (size_t i = 0; i < left->size(); ++i) {
    (*output)[i] = (*left)[i] - right;
  }
}

template<typename DType>
DType Backend<CPUTensor, DType>::SumFunc(
  const CPUTensor<DType>* input) {
  DType output = 0;
  for (size_t i = 0; i < input->size(); ++i) {
    output += (*input)[i];
  }
  return output;
}

template<typename DType>
void Backend<CPUTensor, DType>::AddFunc(
  const CPUTensor<DType>* left, const CPUTensor<DType>* right,
  CPUTensor<DType>* output) {
  CHECK_EQ(left->size(), right->size());
  CHECK_EQ(right->size(), output->size());
  #pragma omp parallel for
  for (size_t i = 0; i < left->size(); ++i) {
    (*output)[i] = (*left)[i] + (*right)[i];
  }
}

template<typename DType>
void Backend<CPUTensor, DType>::MultiplyFunc(
  const CPUTensor<DType>* left, const CPUTensor<DType>* right,
  CPUTensor<DType>* output) {
  CHECK_EQ(left->size(), right->size());
  CHECK_EQ(right->size(), output->size());
  #pragma omp parallel for
  for (size_t i = 0; i < left->size(); ++i) {
    (*output)[i] = (*left)[i] * (*right)[i];
  }
}

template<typename DType>
void Backend<CPUTensor, DType>::MultiplyFunc(
  const CPUTensor<DType>* left, const DType right,
  CPUTensor<DType>* output) {
  CHECK_EQ(left->size(), output->size());
  #pragma omp parallel for
  for (size_t i = 0; i < left->size(); ++i) {
    (*output)[i] = (*left)[i] * right;
  }
}

template<typename DType>
void Backend<CPUTensor, DType>::MakeBinaryMaskFunc(
  const DType low, const DType high,
  const DType keep, CPUTensor<DType>* output) {
  Backend<CPUTensor, DType>::UniformDistributionFunc(low, high, output);

  #pragma omp parallel for
  for (size_t i = 0; i < output->size(); ++i) {
    if ((*output)[i] < keep) {
      (*output)[i] = DType(1);
    } else {
      (*output)[i] = DType(0);
    }
  }
}

template<typename DType>
void Backend<CPUTensor, DType>::ConstantDistributionFunc(
  const DType val, CPUTensor<DType>* output) {
  output->Fill(val);
}

template<typename DType>
void Backend<CPUTensor, DType>::NormalDistributionFunc(
  const DType loc, const DType scale,
  CPUTensor<DType>* output) {
  // TODO(keren) synchronized seed
  static unsigned int seed = 0;
  boost::mt19937 rng((++seed) + time(NULL));
  boost::normal_distribution<DType> nd(loc, scale);  // convert to DType
  boost::variate_generator<boost::mt19937&,
    boost::normal_distribution<DType> > var_nor(rng, nd);
  var_nor.distribution().reset();

  size_t size = output->size();
  for (size_t i = 0; i < size; ++i) {
    (*output)[i] = var_nor();
  }
}

template<typename DType>
void Backend<CPUTensor, DType>::UniformDistributionFunc(
  const DType low, const DType high,
  CPUTensor<DType>* output) {
  // TODO(keren) synchronized seed
  static unsigned int seed = 0;
  boost::mt19937 rng((++seed) + time(NULL));
  boost::uniform_real<DType> ur(low, high);  // convert to DType
  boost::variate_generator<boost::mt19937&,
    boost::uniform_real<DType> > var_uni(rng, ur);
  var_uni.distribution().reset();

  size_t size = output->size();
  for (size_t i = 0; i < size; ++i) {
    (*output)[i] = var_uni();
  }
}

template<typename DType>
void Backend<CPUTensor, DType>::HostCopyToFunc(
  const DType* source, const size_t size, DType* target) {
  BlitzCPUCopy(source, size, target);
}

template<typename DType>
float Backend<CPUTensor, DType>::EvaluateClassifyFunc(
  const CPUTensor<DType>* output, const CPUTensor<DType>* target) {
  size_t num_sample = output->shape()[0];
  size_t dim = output->size() / num_sample;

  float correct = 0.0f;
  #pragma omp parallel for
  for (size_t i = 0; i < num_sample; ++i) {
    DType max = (*output)[i * dim];
    size_t max_index = 0;
    for (size_t j = 1; j < dim; ++j) {
      if (max < (*output)[i * dim + j]) {
        max_index = j;
        max = (*output)[i * dim + j];
      }
    }

    if ((*target)[i * dim + max_index] ==
      static_cast<DType>(1)) {
      #pragma omp atomic
      correct += 1.0f;
    }
  }

  return correct / num_sample;
}

template<typename DType>
float Backend<CPUTensor, DType>::EvaluateRegressFunc(
  const CPUTensor<DType>* output, const CPUTensor<DType>* target) {
  size_t num_sample = output->shape()[0];
  size_t dim = output->size() / num_sample;

  DType result = 0.0;
  DType correct = 0.0;
  #pragma omp parallel for firstprivate(correct)
  for (size_t i = 0; i < num_sample; ++i) {
    DType correct = 0.0;
    for (size_t j = 0; j < dim; ++j) {
      correct += fabs((*output)[i] - (*target)[i]);
    }

    #pragma omp atomic
    result += correct;
  }

  return result / num_sample;
}

#endif  // SRC_BACKEND_CPU_BACKEND_COMMON_INL_H_
