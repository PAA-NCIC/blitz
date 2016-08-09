#ifndef SRC_TRANSFORM_LOGISTIC_H_
#define SRC_TRANSFORM_LOGISTIC_H_

#include "transform/activation.h"

namespace blitz {

template<template <typename> class TensorType, typename DType>
class Logistic : public Activation<TensorType, DType> {
 public:
  explicit Logistic(const bool short_cut = true) : short_cut_(short_cut) {}
  // indicate pure virtual
  ~Logistic() {}

  virtual void Apply(const shared_ptr<TensorType<DType> > input,
    shared_ptr<TensorType<DType> > output);

  virtual void Derivative(const shared_ptr<TensorType<DType> > input,
    shared_ptr<TensorType<DType> > output);

 private:
  const bool short_cut_;
};

}  // namespace blitz

#endif  // SRC_TRANSFORM_LOGISTIC_H_
