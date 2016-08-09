#include "filler/filler_wrapper.h"
#include "backend/backends.h"

namespace blitz {

template<template <typename> class TensorType, typename DType>
void FillerWrapper<TensorType, DType>::Fill() {
  typename map<string, shared_ptr<Filler<TensorType, DType> > >::iterator
    filler_it = fillers_.begin();

  for (; filler_it != fillers_.end(); ++filler_it) {
#ifdef BLITZ_DEVELOP
    LOG(INFO) << "Filler: " << filler_it->first;
#endif
    shared_ptr<Filler<TensorType, DType> > filler =
      filler_it->second;
    filler->Fill();
  }
}

template<template <typename> class TensorType, typename DType>
void FillerWrapper<TensorType, DType>::AddLayer(const string& filler_name,
  const string& layer_name, shared_ptr<TensorType<DType> > weight) {
  if (fillers_.find(filler_name) == fillers_.end()) {
    LOG(ERROR) << "no such filler: " << filler_name;
  }
  shared_ptr<Filler<TensorType, DType> > filler = fillers_[filler_name];
  filler->AddLayer(layer_name, weight);
}

INSTANTIATE_CLASS(FillerWrapper);

}  // namespace blitz
