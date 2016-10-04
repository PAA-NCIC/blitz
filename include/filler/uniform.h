#ifndef INCLUDE_FILLER_UNIFORM_H_
#define INCLUDE_FILLER_UNIFORM_H_

#include <string>

#include "filler/filler.h"

namespace blitz {

template<template <typename> class TensorType, typename DType>
class Uniform : public Filler<TensorType, DType> {
 public:
  typedef typename Filler<TensorType, DType>::LayerWeightIterator
    LayerWeightIterator;

 public:
  explicit Uniform(const string& name,
    const DType low = 0.0, const DType high = 0.0) :
    Filler<TensorType, DType>(name),
    low_(low), high_(high) {}
  ~Uniform() {}

  virtual void FillImpl(LayerWeightIterator layer_weight_it);

 private:
  const float low_;
  const float high_;

  DISABLE_COPY_AND_ASSIGN(Uniform);
};

}  // namespace blitz

#endif  // INCLUDE_FILLER_UNIFORM_H_
