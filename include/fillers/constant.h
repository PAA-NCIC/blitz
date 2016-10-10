#ifndef INCLUDE_FILLERS_H_
#define INCLUDE_FILLERS_H_

#include <string>

#include "fillers/filler.h"

namespace blitz {

template<template <typename> class TensorType, typename DType>
class Constant : public Filler<TensorType, DType> {
 public:
  typedef typename Filler<TensorType, DType>::LayerWeightIterator
    LayerWeightIterator;

 public:
  explicit Constant(const string& name, const DType val) :
    Filler<TensorType, DType>(name), val_(val) {}

  virtual void FillImpl(LayerWeightIterator layer_weight_it);

 private:
  const DType val_;
 
  DISABLE_COPY_AND_ASSIGN(Constant);
};

}  // namespace blitz

#endif  // INCLUDE_FILLERS_CONSTANT_H_
