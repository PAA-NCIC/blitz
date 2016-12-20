#ifndef SRC_BACKENDS_CPU_BACKEND_COMMON_INL_H_
#define SRC_BACKENDS_CPU_BACKEND_COMMON_INL_H_

template<typename DType>
void Backend<MICTensor, DType>::RectlinApplyFunc(
  const MICTensor<DType>* input, MICTensor<DType>* output,
  DType slope) {
  CHECK_EQ(input->size(), output->size());

  DType compare_value = static_cast<DType>(0);
  #pragma omp parallel for
  for (size_t i = 0; i < input->size(); ++i) {
    (*output)[i] = std::max((*input)[i], compare_value) +
      slope * std::min((*input)[i], compare_value);
  }
}

template<typename DType>
void Backend<MICTensor, DType>::RectlinDerivativeFunc(
  const MICTensor<DType>* input, MICTensor<DType>* output,
  DType slope) {
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
void Backend<MICTensor, DType>::LogisticApplyFunc(
  const MICTensor<DType>* input, MICTensor<DType>* output) {
  CHECK_EQ(input->size(), output->size());

  #pragma omp parallel for
  for (size_t i = 0; i < input->size(); ++i) {
    (*output)[i] = 1 / (exp(-(*input)[i]) + 1);
  }
}

template<typename DType>
void Backend<MICTensor, DType>::LogisticDerivativeFunc(
  const MICTensor<DType>* input, MICTensor<DType>* output) {
  CHECK_EQ(input->size(), output->size());
  // TODO(keren) not short cut version
}

template<typename DType>
void Backend<MICTensor, DType>::SoftmaxApplyFunc(
  const MICTensor<DType>* input, MICTensor<DType>* output) {
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
void Backend<MICTensor, DType>::SoftmaxDerivativeFunc(
  const MICTensor<DType>* input, MICTensor<DType>* output) {
  CHECK_EQ(input->size(), output->size());
  // TODO(keren) not short cut version
}

template<typename DType>
DType Backend<MICTensor, DType>::CrossEntropyBinaryApplyFunc(
  const MICTensor<DType>* input, const MICTensor<DType>* target) {
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
DType Backend<MICTensor, DType>::SquareMeanApplyFunc(
  const MICTensor<DType>* input, const MICTensor<DType>* target) {
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
void Backend<MICTensor, DType>::SquareMeanDerivativeFunc(
  const MICTensor<DType>* input, const MICTensor<DType>* target,
  MICTensor<DType>* output) {
  CHECK_EQ(input->size(), target->size());
  CHECK_EQ(output->size(), target->size());
  MinusFunc(input, target, output);
}

template<typename DType>
DType Backend<MICTensor, DType>::AbsMeanApplyFunc(
  const MICTensor<DType>* input, const MICTensor<DType>* target) {
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
void Backend<MICTensor, DType>::AbsMeanDerivativeFunc(
  const MICTensor<DType>* input, const MICTensor<DType>* target,
  MICTensor<DType>* output) {
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
void Backend<MICTensor, DType>::CrossEntropyBinaryDerivativeFunc(
  const MICTensor<DType>* input, const MICTensor<DType>* target,
  MICTensor<DType>* output) {
  MinusFunc(input, target, output);
}

template<typename DType>
DType Backend<MICTensor, DType>::CrossEntropyMultiApplyFunc(
  const MICTensor<DType>* input, const MICTensor<DType>* target) {
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
void Backend<MICTensor, DType>::CrossEntropyMultiDerivativeFunc(
  const MICTensor<DType>* input, const MICTensor<DType>* target,
  MICTensor<DType>* output) {
  MinusFunc(input, target, output);
}

template<typename DType>
void Backend<MICTensor, DType>::BiasForwardFunc(
  const MICTensor<DType>* input, const MICTensor<DType>* bias,
  MICTensor<DType>* output) {
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
void Backend<MICTensor, DType>::BiasBackwardUpdateFunc(
  const MICTensor<DType>* input, MICTensor<DType>* update) {
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
void Backend<MICTensor, DType>::BatchNormForwardFunc(
  const MICTensor<DType>* input,
  const MICTensor<DType>* gamma,
  const MICTensor<DType>* beta,
  MICTensor<DType>* input_var,
  MICTensor<DType>* input_hat,
  MICTensor<DType>* output,
  DType epsilon) {
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
void Backend<MICTensor, DType>::BatchNormBackwardFunc(
  const MICTensor<DType>* backward_input,
  const MICTensor<DType>* forward_input_hat,
  const MICTensor<DType>* forward_input_var,
  const MICTensor<DType>* gamma,
  MICTensor<DType>* gamma_update,
  MICTensor<DType>* beta_update,
  MICTensor<DType>* output,
  DType epsilon) {
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
void Backend<MICTensor, DType>::GradientdescentFunc(
  MICTensor<DType>* weight,
  MICTensor<DType>* gradient,
  MICTensor<DType>* velocity,
  DType momentum_coef,
  DType learning_rate,
  DType decay,
  size_t batch_size) {
  CHECK_EQ(weight->size(), gradient->size());
  CHECK_EQ(gradient->size(), velocity->size());
#ifdef BLITZ_DEVELOP
  LOG(INFO) << "weight size: " << weight->size();
  LOG(INFO) << "momentum_coef: " << momentum_coef;
  LOG(INFO) << "learning_rate: " << learning_rate;
  LOG(INFO) << "decay: " << decay;
  LOG(INFO) << "batch_size: " << batch_size;
#endif

  #pragma omp parallel for
  for (size_t i = 0; i < velocity->size(); ++i) {
    (*gradient)[i] /= batch_size;
    (*velocity)[i] = (*velocity)[i] * momentum_coef - learning_rate *
      ((*gradient)[i] + decay * (*weight)[i]);
    (*weight)[i] = (*weight)[i] + (*velocity)[i];
  }
}

template<typename DType>
void Backend<MICTensor, DType>::MatrixMultiplyFunc(
  const MICTensor<DType>* left,
  const MICTensor<DType>* right,
  MICTensor<DType>* output, 
  bool transa,
  bool transb,
  DType alpha,
  DType beta,
  BLITZ_ALGORITHM algorithm ) {
  size_t dim_left = transa ? left->size() / (left->shape())[0] :
    (left->shape())[0];
  size_t dim_right = transb ? (right->shape())[0] :
    right->size() / (right->shape())[0];
  size_t dim_common_left = transa ? (left->shape())[0] :
    left->size() / (left->shape())[0];
  size_t dim_common_right = transb ? right->size() / (right->shape())[0] :
    (right->shape())[0];
  CHECK_EQ(dim_common_left, dim_common_right);
  BlitzCPUGemm(
    const_cast<MICTensor<DType>*>(left)->data(),
    const_cast<MICTensor<DType>*>(right)->data(),
    output->data(), 
    transa, transb, 
    alpha, beta,
    dim_left, dim_right, dim_common_left);
}

template<typename DType>
void Backend<MICTensor, DType>::Transpose2DFunc(
  const MICTensor<DType>* input, MICTensor<DType>* output) {
  size_t dim_left = input->shape()[0];
  size_t dim_right = input->shape()[1];
  CHECK_EQ(dim_left, output->shape()[1]);
  CHECK_EQ(dim_right, output->shape()[0]);
  #pragma omp parallel for collapse(2)
  for (size_t i = 0; i < dim_left; ++i) {
    for (size_t j = 0; j < dim_right; ++j) {
      (*output)[j * dim_left + i] = (*input)[i * dim_right + j];
    }
  }
}

template<typename DType>
void Backend<MICTensor, DType>::MaximumFunc(
  const MICTensor<DType>* left, const MICTensor<DType>* right,
  MICTensor<DType>* output) {
  CHECK_EQ(left->size(), right->size());
  CHECK_EQ(right->size(), output->size());
  #pragma omp parallel for
  for (size_t i = 0; i < left->size(); ++i) {
    (*output)[i] = std::max((*left)[i], (*right)[i]);
  }
}

template<typename DType>
void Backend<MICTensor, DType>::MinusFunc(
  const MICTensor<DType>* left, const MICTensor<DType>* right,
  MICTensor<DType>* output) {
  CHECK_EQ(left->size(), right->size());
  CHECK_EQ(right->size(), output->size());

  #pragma omp parallel for
  for (size_t i = 0; i < left->size(); ++i) {
    (*output)[i] = (*left)[i] - (*right)[i];
  }
}

template<typename DType>
DType Backend<MICTensor, DType>::SumFunc(
  const MICTensor<DType>* input) {
  DType output = 0;
  for (size_t i = 0; i < input->size(); ++i) {
    output += (*input)[i];
  }
  return output;
}

template<typename DType>
void Backend<MICTensor, DType>::AddFunc(
  const MICTensor<DType>* left, const MICTensor<DType>* right,
  MICTensor<DType>* output) {
  CHECK_EQ(left->size(), right->size());
  CHECK_EQ(right->size(), output->size());
  #pragma omp parallel for
  for (size_t i = 0; i < left->size(); ++i) {
    (*output)[i] = (*left)[i] + (*right)[i];
  }
}

template<typename DType>
void Backend<MICTensor, DType>::MultiplyFunc(
  const MICTensor<DType>* left, const MICTensor<DType>* right,
  MICTensor<DType>* output) {
  CHECK_EQ(left->size(), right->size());
  CHECK_EQ(right->size(), output->size());
  #pragma omp parallel for
  for (size_t i = 0; i < left->size(); ++i) {
    (*output)[i] = (*left)[i] * (*right)[i];
  }
}


template<typename DType>
void Backend<MICTensor, DType>::MultiplyFunc(
  const MICTensor<DType>* left, MICTensor<DType>* output,
  DType right) {
  CHECK_EQ(left->size(), output->size());
  #pragma omp parallel for
  for (size_t i = 0; i < left->size(); ++i) {
    (*output)[i] = (*left)[i] * right;
  }
}

template<typename DType>
void Backend<MICTensor, DType>::MakeBinaryMaskFunc(
  MICTensor<DType>* output,
  DType low,
  DType high,
  DType keep) { 
  UniformDistributionFunc(output, low, high);

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
void Backend<MICTensor, DType>::ConstantDistributionFunc(
  MICTensor<DType>* output, DType val) {
  output->Fill(val);
}

template<typename DType>
void Backend<MICTensor, DType>::NormalDistributionFunc(
  MICTensor<DType>* output, DType loc, DType scale) {
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
void Backend<MICTensor, DType>::UniformDistributionFunc(
  MICTensor<DType>* output, DType low, DType high) {
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
void Backend<MICTensor, DType>::HostCopyToFunc(
  const DType* source, DType* target, size_t size) {
  BlitzCPUCopy(source, target, size);
}

template<typename DType>
float Backend<MICTensor, DType>::EvaluateClassifyFunc(
  const MICTensor<DType>* output, const MICTensor<DType>* target) {
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
float Backend<MICTensor, DType>::EvaluateRegressFunc(
  const MICTensor<DType>* output, const MICTensor<DType>* target) {
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

#endif  // SRC_BACKENDS_CPU_BACKEND_COMMON_INL_H_
