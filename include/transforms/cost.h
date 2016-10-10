#ifndef INCLUDE_TRANSFORMS_COST_H_
#define INCLUDE_TRANSFORMS_COST_H_

#include "backends/tensor.h"
#include "utils/common.h"

namespace blitz {

template<template <typename> class TensorType, typename DType>
class Cost {
 public:
  Cost() {}  // indicate pure virtual
  virtual ~Cost() {}  // ensure pure virtual

  virtual DType Apply(const shared_ptr<TensorType<DType> > output,
    const shared_ptr<TensorType<DType> > target) = 0;

  virtual void Derivative(
    const shared_ptr<TensorType<DType> > output,
    const shared_ptr<TensorType<DType> > target,
    shared_ptr<TensorType<DType> > result) = 0;

  DISABLE_COPY_AND_ASSIGN(Cost);
};

}  // namespace blitz

#endif  // INCLUDE_TRANSFORMS_COST_H_
