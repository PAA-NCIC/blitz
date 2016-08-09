#ifndef SRC_TRANSFORM_CROSS_ENTROPY_MULTI_H_
#define SRC_TRANSFORM_CROSS_ENTROPY_MULTI_H_

#include "transform/cost.h"

namespace blitz {

template<template <typename> class TensorType, typename DType>
class CrossEntropyMulti : public Cost<TensorType, DType> {
 public:
  explicit CrossEntropyMulti(const DType scale = 1.0) : scale_(scale) {}
  ~CrossEntropyMulti() {}

  virtual DType Apply(const shared_ptr<TensorType<DType> > output,
    const shared_ptr<TensorType<DType> > target);

  virtual void Derivative(
    const shared_ptr<TensorType<DType> > output,
    const shared_ptr<TensorType<DType> > target,
    shared_ptr<TensorType<DType> > result);

 private:
  const float scale_;
};

}  // namespace blitz

#endif  // SRC_TRANSFORM_CROSS_ENTROPY_MULTI_H_
