#ifndef INCLUDE_TRANSFORM_CROSS_ENTROPY_BINARY_H_
#define INCLUDE_TRANSFORM_CROSS_ENTROPY_BINARY_H_

#include "transform/cost.h"

namespace blitz {

template<template <typename> class TensorType, typename DType>
class CrossEntropyBinary : public Cost<TensorType, DType> {
 public:
  explicit CrossEntropyBinary(const DType scale = 1.0) : scale_(scale) {}
  ~CrossEntropyBinary() {}

  virtual DType Apply(const shared_ptr<TensorType<DType> > output,
    const shared_ptr<TensorType<DType> > target);

  virtual void Derivative(
    const shared_ptr<TensorType<DType> > output,
    const shared_ptr<TensorType<DType> > target,
    shared_ptr<TensorType<DType> > result);

 private:
  const DType scale_;

  DISABLE_COPY_AND_ASSIGN(CrossEntropyBinary);
};

}  // namespace blitz

#endif  // INCLUDE_TRANSFORM_CROSS_ENTROPY_BINARY_H_
