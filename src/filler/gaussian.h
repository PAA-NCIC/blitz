#ifndef SRC_FILLER_GAUSSIAN_H_
#define SRC_FILLER_GAUSSIAN_H_

#include <string>

#include "filler/filler.h"

namespace blitz {

template<template <typename> class TensorType, typename DType>
class Gaussian : public Filler<TensorType, DType> {
 public:
  typedef typename Filler<TensorType, DType>::LayerWeightIterator
    LayerWeightIterator;

 public:
  explicit Gaussian(const string& name,
    const DType loc = 0.0f, const DType scale = 1.0f) :
    Filler<TensorType, DType>(name),
    loc_(loc), scale_(scale) {}
  ~Gaussian() {}

  virtual void FillImpl(LayerWeightIterator layer_weight_it);

 private:
  const DType loc_;
  const DType scale_;
};

}  // namespace blitz

#endif  // SRC_FILLER_GAUSSIAN_H_

