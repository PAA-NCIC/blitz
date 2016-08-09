#ifndef SRC_BACKEND_CPU_BACKEND_H_
#define SRC_BACKEND_CPU_BACKEND_H_

#include "backend/backend.h"

#include <chrono>
#include <cmath>
#include <boost/random.hpp>
#include <boost/random/normal_distribution.hpp>
#include <boost/random/uniform_real.hpp>
#include <omp.h>

#include "util/blitz_function.h"
#include "backend/cpu_tensor.h"

namespace blitz {

// default general CPU
template<typename DType>
class Backend<CPUTensor, DType> {
 public:
  static void RectlinApplyFunc(
    const CPUTensor<DType>* input,
    const DType slope,
    CPUTensor<DType>* output) {
    CHECK_EQ(input->size(), output->size());

    DType compara_value = static_cast<DType>(0);
    #pragma omp parallel for num_threads(8)
    for (size_t i = 0; i < input->size(); ++i) {
      (*output)[i] = std::max((*input)[i], compara_value) +
        slope * std::min((*input)[i], compara_value);
    }
  }

  static void RectlinDerivativeFunc(
    const CPUTensor<DType>* input,
    const DType slope,
    CPUTensor<DType>* output) {
    CHECK_EQ(input->size(), output->size());

    DType compara_value = static_cast<DType>(0);
    #pragma omp parallel for num_threads(8)
    for (size_t i = 0; i < input->size(); ++i) {
      DType greater = (*input)[i] > compara_value ? 1.0 : 0.0;
      DType less = (*input)[i] <= compara_value ? slope : 0.0;
      (*output)[i] = (greater + less) * (*output)[i];
    }
  }

  static void LogisticApplyFunc(
    const CPUTensor<DType>* input,
    CPUTensor<DType>* output) {
    CHECK_EQ(input->size(), output->size());

    #pragma omp parallel for
    for (size_t i = 0; i < input->size(); ++i) {
      (*output)[i] = 1 / (exp(-(*input)[i]) + 1);
    }
  }

  static void LogisticDerivativeFunc(
    const CPUTensor<DType>* input,
    CPUTensor<DType>* output) {
    CHECK_EQ(input->size(), output->size());
    // TODO(keren) not short cut version
  }

  static void SoftmaxApplyFunc(
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

  static void SoftmaxDerivativeFunc(
    const CPUTensor<DType>* input,
    CPUTensor<DType>* output) {
    CHECK_EQ(input->size(), output->size());
    // TODO(keren) not short cut version
  }

  static DType CrossEntropyBinaryApplyFunc(
    const CPUTensor<DType>* input,
    const CPUTensor<DType>* target) {
    CHECK_EQ(input->size(), target->size());
    DType private_output = 0;
    DType output = 0;
    #pragma omp parallel firstprivate(private_output)
    {
      #pragma omp for
      for (size_t i = 0; i < input->size(); ++i) {
        private_output += -log((*input)[i]) * (*target)[i] -
        log(1 - (*input)[i]) * (1 - (*target)[i]);
      }

      #pragma omp atomic
      output += private_output;
    }

    output /= (input->shape())[0];
    return output;
  }

  static void CrossEntropyBinaryDerivativeFunc(
    const CPUTensor<DType>* input, const CPUTensor<DType>* target,
    CPUTensor<DType>* output) {
    MinusFunc(input, target, output);
  }

  static DType CrossEntropyMultiApplyFunc(
    const CPUTensor<DType>* input, const CPUTensor<DType>* target) {
    CHECK_EQ(input->size(), target->size());
    DType private_output = 0;
    DType output = 0;

    #pragma omp parallel firstprivate(private_output)
    {
      #pragma omp for
      for (size_t i = 0; i < input->size(); ++i) {
        private_output += log((*input)[i]) * (*target)[i];
      }

      #pragma omp atomic
      output += -private_output;
    }

    output /= (input->shape())[0];
    return output;
  }

  static void CrossEntropyMultiDerivativeFunc(
    const CPUTensor<DType>* input, const CPUTensor<DType>* target,
    CPUTensor<DType>* output) {
    MinusFunc(input, target, output);
  }

  static void BiasForwardFunc(
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

  static void BiasBackwardUpdateFunc(const CPUTensor<DType>* input,
    CPUTensor<DType>* update) {
    size_t num_sample = input->shape()[0];
    size_t dim = input->size() / num_sample;

    update->Fill(0);

    #pragma omp parallel for
    for (size_t i = 0; i < dim; ++i) {
      for (size_t j = 0; j < num_sample; ++j) {
        (*update)[i] += (*input)[j * dim + i];
      }
    }
  }

  static void BatchNormForwardFunc(
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

  static void BatchNormBackwardFunc(
    const CPUTensor<DType>* backward_input,
    const CPUTensor<DType>* forward_input_hat,
    const CPUTensor<DType>* forward_input_var,
    const CPUTensor<DType>* gamma, const DType epsilon,
    CPUTensor<DType>* gamma_update, CPUTensor<DType>* beta_update,
    CPUTensor<DType>* output) {
    size_t num_sample = backward_input->shape()[0];
    size_t dim = backward_input->size() / num_sample;
    gamma_update->Fill(0);
    beta_update->Fill(0);

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

  static void GradientdescentFunc(
    const DType momentum_coef, const DType learning_rate,
    const DType decay, const int batch_size,
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

    #pragma omp parallel for
    for (size_t i = 0; i < velocity->size(); ++i) {
      (*gradient)[i] /= batch_size;
      (*velocity)[i] = (*velocity)[i] * momentum_coef - learning_rate *
        ((*gradient)[i] + decay * (*weight)[i]);
      (*weight)[i] = (*weight)[i] + (*velocity)[i];
    }
  }

  static void MatrixDotFunc(
    const CPUTensor<DType>* left, const CPUTensor<DType>* right,
    const bool transa, const bool transb,
    const DType alpha, const DType beta,
    CPUTensor<DType>* output) {
    int dim_left = transa ? left->size() / (left->shape())[0] :
      (left->shape())[0];
    int dim_right = transb ? (right->shape())[0] :
      right->size() / (right->shape())[0];
    int dim_common_left = transa ? (left->shape())[0] :
      left->size() / (left->shape())[0];
    int dim_common_right = transb ? right->size() / (right->shape())[0] :
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

  static void MaximumFunc(
    const CPUTensor<DType>* left, const CPUTensor<DType>* right,
    CPUTensor<DType>* output) {
    CHECK_EQ(left->size(), right->size());
    CHECK_EQ(right->size(), output->size());
    #pragma omp parallel for
    for (size_t i = 0; i < left->size(); ++i) {
      (*output)[i] = std::max((*left)[i], (*right)[i]);
    }
  }

  static void MaximumFunc(
    const CPUTensor<DType>* left,
    const DType right,
    CPUTensor<DType>* output) {
    CHECK_EQ(left->size(), output->size());
    #pragma omp parallel for
    for (size_t i = 0; i < left->size(); ++i) {
      (*output)[i] = std::max((*left)[i], right);
    }
  }

  static void MinusFunc(
    const CPUTensor<DType>* left, const CPUTensor<DType>* right,
    CPUTensor<DType>* output) {
    CHECK_EQ(left->size(), right->size());
    CHECK_EQ(right->size(), output->size());
    #pragma omp parallel for
    for (size_t i = 0; i < left->size(); ++i) {
      (*output)[i] = (*left)[i] - (*right)[i];
    }
  }

  static void MinusFunc(
    const CPUTensor<DType>* left,
    const DType right,
    CPUTensor<DType>* output) {
    CHECK_EQ(left->size(), output->size());
    #pragma omp parallel for
    for (size_t i = 0; i < left->size(); ++i) {
      (*output)[i] = (*left)[i] - right;
    }
  }

  static DType SumFunc(const CPUTensor<DType>* input) {
    DType output = 0;
    for (size_t i = 0; i < input->size(); ++i) {
      output += (*input)[i];
    }
    return output;
  }

  static void AddFunc(
    const CPUTensor<DType>* left, const CPUTensor<DType>* right,
    CPUTensor<DType>* output) {
    CHECK_EQ(left->size(), right->size());
    CHECK_EQ(right->size(), output->size());
    #pragma omp parallel for
    for (size_t i = 0; i < left->size(); ++i) {
      (*output)[i] = (*left)[i] + (*right)[i];
    }
  }

  static void MultiplyFunc(
    const CPUTensor<DType>* left, const CPUTensor<DType>* right,
    CPUTensor<DType>* output) {
    CHECK_EQ(left->size(), right->size());
    CHECK_EQ(right->size(), output->size());
    #pragma omp parallel for
    for (size_t i = 0; i < left->size(); ++i) {
      (*output)[i] = (*left)[i] * (*right)[i];
    }
  }

  static void MultiplyFunc(
    const CPUTensor<DType>* left, const DType right,
    CPUTensor<DType>* output) {
    CHECK_EQ(left->size(), output->size());
    #pragma omp parallel for
    for (size_t i = 0; i < left->size(); ++i) {
      (*output)[i] = (*left)[i] * right;
    }
  }

  inline static void Convolution2DForwardFunc(
    const CPUTensor<DType>* input, const CPUTensor<DType>* weight,
    const int padding_height, const int padding_width,
    const int stride_height, const int stride_width,
    CPUTensor<DType>* unpack, CPUTensor<DType>* output);

  inline static void Convolution2DBackwardFunc(
    const CPUTensor<DType>* output, const CPUTensor<DType>* weight,
    const int padding_height, const int padding_width,
    const int stride_height, const int stride_width,
    CPUTensor<DType>* pack, CPUTensor<DType>* input);

  inline static void Convolution2DUpdateFunc(
    const CPUTensor<DType>* input, const CPUTensor<DType>* output,
    const int padding_height, const int padding_width,
    const int stride_height, const int stride_width,
    CPUTensor<DType>* unpack, CPUTensor<DType>* update);

  // batch parallel
  inline static void Convolution2DForwardFunc(
    const CPUTensor<DType>* input, const CPUTensor<DType>* weight,
    const int padding_height, const int padding_width,
    const int stride_height, const int stride_width,
    vector<shared_ptr<CPUTensor<DType> > >* unpack_batch,
    CPUTensor<DType>* output);

  inline static void Convolution2DBackwardFunc(
    const CPUTensor<DType>* output, const CPUTensor<DType>* weight,
    const int padding_height, const int padding_width,
    const int stride_height, const int stride_width,
    vector<shared_ptr<CPUTensor<DType> > >* pack_batch,
    CPUTensor<DType>* input);

  inline static void Convolution2DUpdateFunc(
    const CPUTensor<DType>* input, const CPUTensor<DType>* output,
    const int padding_height, const int padding_width,
    const int stride_height, const int stride_width,
    vector<shared_ptr<CPUTensor<DType> > >* unpack_batch,
    vector<shared_ptr<CPUTensor<DType> > >* update_batch,
    CPUTensor<DType>* update);

  inline static void MaxPooling2DForwardFunc(
    const CPUTensor<DType>* input,
    const int filter_height, const int filter_width,
    const int stride_width, const int stride_height,
    CPUTensor<int>* max_index, CPUTensor<DType>* output);

  inline static void MaxPooling2DBackwardFunc(
    const CPUTensor<DType>* output, const CPUTensor<int>* max_index,
    const int filter_height, const int filter_width,
    const int stride_height, const int stride_width,
    CPUTensor<DType>* input);

  // naive parallel
  inline static void Convolution2DForwardFunc(
    const CPUTensor<DType>* input, const CPUTensor<DType>* weight,
    const int stride_height, const int stride_width,
    CPUTensor<DType>* output);

  static void MakeBinaryMaskFunc(const DType low, const DType high,
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

  static void ConstantDistributionFunc(
    const DType val, CPUTensor<DType>* output) {
    output->Fill(val);
  }

  static void NormalDistributionFunc(
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
    #pragma omp parallel for
    for (size_t i = 0; i < size; ++i) {
      (*output)[i] = var_nor();
    }
  }

  static void UniformDistributionFunc(
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
    #pragma omp parallel for
    for (size_t i = 0; i < size; ++i) {
      (*output)[i] = var_uni();
    }
  }

  static void CopyFunc(const DType* source, const size_t size,
    DType* target) {
    BlitzCPUCopy(source, size, target);
  }

  static float EvaluateFunc(
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

      if ((*target)[i * dim + max_index] == static_cast<DType>(1)) {
        #pragma omp atomic
        correct += 1.0f;
      }
    }

    return correct / num_sample;
  }

 private:
  inline static void Unpack2DParallelFunc(
    const DType* input, const int channel,
    const int input_height, const int input_width,
    const int filter_height, const int filter_width,
    const int output_height, const int output_width,
    const int padding_height, const int padding_width,
    const int stride_height, const int stride_width,
    DType* unpack);

  inline static void Pack2DParallelFunc(
    const DType* pack, const int channel,
    const int input_height, const int input_width,
    const int filter_height, const int filter_width,
    const int output_height, const int output_width,
    const int padding_height, const int padding_width,
    const int stride_height, const int stride_width,
    DType* input); 

  inline static void Unpack2DFunc(
    const DType* input, const int channel,
    const int input_height, const int input_width,
    const int filter_height, const int filter_width,
    const int output_height, const int output_width,
    const int padding_height, const int padding_width,
    const int stride_height, const int stride_width,
    DType* unpack); 

  inline static void Pack2DFunc(
    const DType* pack, const int channel,
    const int input_height, const int input_width,
    const int filter_height, const int filter_width,
    const int output_height, const int output_width,
    const int padding_height, const int padding_width,
    const int stride_height, const int stride_width,
    DType* input);
};

}  // namespace blitz

#include "backend/cpu_backend_conv-inl.h"
#include "backend/cpu_backend_pool-inl.h"
#include "backend/cpu_backend_pack-inl.h"

#endif  // SRC_BACKEND_CPU_BACKEND_H_
