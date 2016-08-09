#ifndef SRC_FILLER_CONSTANT_H_
#define SRC_FILLER_CONSTANT_H_

#include <string>

#include "filler/filler.h"

namespace blitz {

template<template <typename> class TensorType, typename DType>
class Constant : public Filler<TensorType, DType> {
 public:
  typedef typename Filler<TensorType, DType>::LayerWeightIterator
    LayerWeightIterator;

 public:
  explicit Constant(const string& name, const DType val = 0.0f) :
    Filler<TensorType, DType>(name), val_(val) {}

  virtual void FillImpl(LayerWeightIterator layer_weight_it);

 private:
  const DType val_;
};

}  // namespace blitz

#endif  // SRC_FILLER_CONSTANT_H_
