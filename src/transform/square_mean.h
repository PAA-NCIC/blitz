#ifndef SRC_TRANSFORM_SQUARE_MEAN_H_
#define SRC_TRANSFORM_SQUARE_MEAN_H_

#include "transform/cost.h"

namespace blitz {

template<template <typename> class TensorType, typename DType>
class SquareMean : public Cost<TensorType, DType> {
 public:
  SquareMean() {}
  ~SquareMean() {}

  virtual DType Apply(const shared_ptr<TensorType<DType> > output,
    const shared_ptr<TensorType<DType> > target);

  virtual void Derivative(
    const shared_ptr<TensorType<DType> > output,
    const shared_ptr<TensorType<DType> > target,
    shared_ptr<TensorType<DType> > result);
};

}  // namespace blitz

#endif  // SRC_TRANSFORM_SQUARE_MEAN_H_
