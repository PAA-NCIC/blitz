#ifndef INCLUDE_TRANSFORMS_RECTLIN_H_
#define INCLUDE_TRANSFORMS_RECTLIN_H_

#include "transforms/activation.h"

namespace blitz {

template<template <typename> class TensorType, typename DType>
class Rectlin : public Activation<TensorType, DType> {
 public:
  explicit Rectlin(const DType slope = 0.0) : slope_(slope) {}
  // indicate pure virtual
  ~Rectlin() {}

  virtual void Apply(const shared_ptr<TensorType<DType> > input,
    shared_ptr<TensorType<DType> > output);

  virtual void Derivative(const shared_ptr<TensorType<DType> > input,
    shared_ptr<TensorType<DType> > output);

 private:
  const DType slope_;

  DISABLE_COPY_AND_ASSIGN(Rectlin);
};

}  // namespace blitz

#endif  // INCLUDE_TRANSFORMS_RECTLIN_H_
