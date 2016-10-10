#ifndef INCLUDE_TRANSFORMS_ACTIVATION_H_
#define INCLUDE_TRANSFORMS_ACTIVATION_H_

#include "backends/tensor.h"
#include "utils/common.h"

namespace blitz {

template<template <typename> class TensorType, typename DType>
class Activation {
 public:
  Activation() {}  // indicate pure virtual
  virtual ~Activation() {}  // ensure pure virtual

  virtual void Apply(const shared_ptr<TensorType<DType> > input,
    shared_ptr<TensorType<DType> > output) = 0;
  virtual void Derivative(const shared_ptr<TensorType<DType> > input,
    shared_ptr<TensorType<DType> > output) = 0;

  DISABLE_COPY_AND_ASSIGN(Activation);
};

}  // namespace blitz

#endif  // INCLUDE_TRANSFORMS_ACTIVATION_H_
