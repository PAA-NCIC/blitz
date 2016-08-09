#ifndef SRC_TRANSFORM_RECTLIN_H_
#define SRC_TRANSFORM_RECTLIN_H_

#include "transform/activation.h"

namespace blitz {

template<template <typename> class TensorType, typename DType>
class Rectlin : public Activation<TensorType, DType> {
 public:
  explicit Rectlin(const float slope = 0.0f) : slope_(slope) {}
  // indicate pure virtual
  ~Rectlin() {}

  virtual void Apply(const shared_ptr<TensorType<DType> > input,
    shared_ptr<TensorType<DType> > output);

  virtual void Derivative(const shared_ptr<TensorType<DType> > input,
    shared_ptr<TensorType<DType> > output);

 private:
  const float slope_;
};

}  // namespace blitz

#endif  // SRC_TRANSFORM_RECTLIN_H_
