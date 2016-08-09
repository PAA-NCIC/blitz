#ifndef SRC_FILLER_FILLER_WRAPPER_H_
#define SRC_FILLER_FILLER_WRAPPER_H_

#include <map>
#include <string>

#include "filler/filler.h"

namespace blitz {

template<template <typename> class TensorType, typename DType>
class FillerWrapper {
 public:
  explicit FillerWrapper(
    map<string, shared_ptr<Filler<TensorType, DType> > > fillers) :
    fillers_(fillers) {}

  void Fill();

  void AddLayer(const string& filler_name, const string& layer_name,
    shared_ptr<TensorType<DType> > weight);

 private:
  map<string, shared_ptr<Filler<TensorType, DType> > > fillers_;
};

}  // namespace blitz

#endif  // SRC_FILLER_FILLER_WRAPPER_H_
