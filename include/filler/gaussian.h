#ifndef INCLUDE_FILLER_GAUSSIAN_H_
#define INCLUDE_FILLER_GAUSSIAN_H_

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
    const DType loc = 0.0, const DType scale = 1.0) :
    Filler<TensorType, DType>(name),
    loc_(loc), scale_(scale) {}
  ~Gaussian() {}

  virtual void FillImpl(LayerWeightIterator layer_weight_it);

 private:
  const DType loc_;
  const DType scale_;

  DISABLE_COPY_AND_ASSIGN(Gaussian);
};

}  // namespace blitz

#endif  // INCLUDE_FILLER_GAUSSIAN_H_
