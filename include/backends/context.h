#ifndef INCLUDE_BACKENDS_BACKEND_CONTEXT_H_
#define INCLUDE_BACKENDS_BACKEND_CONTEXT_H_

#include "backends/shape.h"
#include "utils/common.h"
#include "utils/blitz_define.h"
#include "utils/blitz_algorithm_function.h"
#include "utils/blitz_shape_function.h"

namespace blitz {

template<template <typename> class TensorType, typename DType>
class ConvolutionContext {
 public:
  ConvolutionContext(const Shape& input_shape, const Shape& filter_shape,
    size_t pad_h, size_t pad_w, size_t str_h, size_t str_w) :
    pad_h_(pad_h), pad_w_(pad_w), str_h_(str_h), str_w_(str_w),
    conv_algorithm_(BLITZ_CONVOLUTION_NAIVE_DIRECT) {
    // shape decode
    Blitz2DBuffer(input_shape, &N_, &C_, &H_, &W_);
    Blitz2DFilter(filter_shape, &KF_, &CF_, &R_, &S_);
    CHECK_EQ(CF_, C_);
    P_ = (H_ + 2 * pad_h_ - R_) / str_h_ + 1;
    Q_ = (W_ + 2 * pad_w_ - S_) / str_w_ + 1;
  }

  size_t pad_h() {
    return this->pad_h_;
  }

  size_t pad_w() {
    return this->pad_w_;
  }

  size_t str_h() {
    return this->str_h_;
  }

  size_t str_w() {
    return this->str_w_;
  }

  BLITZ_ALGORITHM algorithm() {
    return this->conv_algorithm_;
  }

  TensorType<DType>* workspace() {
    return this->workspace_.get();
  }

  void CheckInputDataLayout(size_t NIN, size_t C, size_t H, size_t W) {
    CHECK_EQ(NIN, N_);
    CHECK_EQ(C, C_);
    CHECK_EQ(H, H_);
    CHECK_EQ(W, W_);
  }

  void CheckFilterDataLayout(size_t KF, size_t CF, size_t R, size_t S) {
    CHECK_EQ(KF, KF_);
    CHECK_EQ(CF, CF_);
    CHECK_EQ(R, R_);
    CHECK_EQ(S, S_);
  }

  void CheckOutputDataLayout(size_t NOUT, size_t K, size_t P, size_t Q) {
    CHECK_EQ(NOUT, N_);
    CHECK_EQ(K, KF_);
    CHECK_EQ(P, P_);
    CHECK_EQ(Q, Q_);
  }

  void InitAlgorithmForUser(BLITZ_ALGORITHM algorithm);

  void InitAlgorithmForMemory(size_t memory_size = 2 << 20);

  void InitAlgorithmForSpeed(size_t memory_size = 2 << 20);

 private:
  size_t pad_h_, pad_w_;
  size_t str_h_, str_w_;
  size_t N_, C_, H_, W_;
  size_t KF_, CF_, R_, S_;
  size_t P_, Q_;
  BLITZ_ALGORITHM conv_algorithm_;
  shared_ptr<TensorType<DType> > workspace_;
};

#define INSTANTIATE_CONTEXT(type, tensor) \
  char BlitzInstantiation##type##ContextGuard##tensor; \
  template class type##Context<tensor, float> \

}  // namespace blitz

#endif  // INCLUDE_BACKENDS_BACKEND_CONTEXT_H_
