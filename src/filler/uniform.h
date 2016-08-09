#ifndef SRC_FILLER_UNIFORM_H_
#define SRC_FILLER_UNIFORM_H_

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
    const DType low = 0.0f, const DType high = 0.0f) :
    Filler<TensorType, DType>(name),
    low_(low), high_(high) {}
  ~Uniform() {}

  virtual void FillImpl(LayerWeightIterator layer_weight_it);

 private:
  const float low_;
  const float high_;
};

}  // namespace blitz

#endif  // SRC_FILLER_UNIFORM_H_
