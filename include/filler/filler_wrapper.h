#ifndef INCLUDE_FILLER_FILLER_WRAPPER_H_
#define INCLUDE_FILLER_FILLER_WRAPPER_H_

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

  DISABLE_COPY_AND_ASSIGN(FillerWrapper);
};

}  // namespace blitz

#endif  // INCLUDE_FILLER_FILLER_WRAPPER_H_
