#ifndef SRC_TRANSFORM_COST_H_
#define SRC_TRANSFORM_COST_H_

#include "util/common.h"
#include "backend/tensor.h"

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

#endif  // SRC_TRANSFORM_COST_H_
