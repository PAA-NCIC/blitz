#ifndef INCLUDE_DATA_DATA_ITERATOR_H_
#define INCLUDE_DATA_DATA_ITERATOR_H_

#include <string>
#include <vector>

#include "util/common.h"
#include "backend/shape.h"

namespace blitz {

template<template <typename> class TensorType, typename DType>
class DataIterator {
 public:
  explicit DataIterator(const string& data_path, const Shape& input_shape,
    const int batch_size, const int pool_size = 3000) :
    data_path_(data_path), input_shape_(input_shape), batch_size_(batch_size),
    pool_size_(pool_size), current_begin_index_(0), total_(0) {}

  void Init();

  shared_ptr<TensorType<DType> > GenerateTensor(const int index);

  // getters
  const Shape& input_shape() const {
    return this->input_shape_;
  }

  const string& data_path() const {
    return this->data_path_;
  }

  int batch_size() const {
    return this->batch_size_;
  }

  int total() const {
    return this->total_;
  }

 private:
  void CopyFileBuffer(int begin_offset);

  const string data_path_;

  const Shape input_shape_;

  const int batch_size_;

  int pool_size_;
  int current_begin_index_;
  int total_;

  vector<string> files_;
  vector<int> file_row_mapping_;
  vector<shared_ptr<TensorType<DType> > > tensor_pool_;

  DISABLE_COPY_AND_ASSIGN(DataIterator); 
};

}  // namespace blitz

#endif  // INCLUDE_DATA_DATA_ITERATOR_H_


