#ifndef INCLUDE_BACKENDS_SHAPE_H_
#define INCLUDE_BACKENDS_SHAPE_H_

#include <vector>

#include "utils/common.h"

namespace blitz {

enum BLITZ_DATA_LAYOUT {
  BLITZ_FLAT = 0,
  BLITZ_BUFFER_NCHW = 1,
  BLITZ_BUFFER_NHWC = 2,
  BLITZ_FILTER_KCRS = 3,
  BLITZ_FILTER_RSCK = 4,
  BLITZ_PACK_PQRSC = 5,
  BLITZ_PACK_PQCRS = 6,
  BLITZ_PACK_CRSPQ = 7,
  BLITZ_PACK_RSCPQ = 8,
  BLITZ_SHAPE_UNDEFINED = 9
};

// Rule of three: use self-defined copy assignment to restore size_ to 0
class Shape {
 public:
  explicit Shape(const size_t dimension) :
    size_(0), dimension_(dimension),
    shape_(dimension), data_layout_(BLITZ_FLAT) {}

  explicit Shape(const size_t dimension, BLITZ_DATA_LAYOUT data_layout) :
    size_(0), dimension_(dimension),
    shape_(dimension), data_layout_(data_layout) {}

  explicit Shape(const std::vector<size_t>& shape) :
    size_(0), dimension_(shape.size()),
    shape_(shape), data_layout_(BLITZ_FLAT) {}

  explicit Shape(const std::vector<size_t>& shape, BLITZ_DATA_LAYOUT data_layout) :
    size_(0), dimension_(shape.size()),
    shape_(shape), data_layout_(data_layout) {}

  // copy constructor
  Shape(const Shape& shape) :
    size_(0), dimension_(shape.dimension_),
    shape_(shape.shape_), data_layout_(shape.data_layout_) {}

  // setters
  void set_data_layout(BLITZ_DATA_LAYOUT data_layout) {
    this->data_layout_ = data_layout;
  }

  // getters
  size_t dimension() const {
    return dimension_;
  }

  size_t size() const {
    if (size_ == 0) {
      size_ = new size_t();
      (*size_) = 1;
      for (size_t i = 0; i < dimension_; ++i) {
        if (shape_[i] != 0) {
          (*size_) *= shape_[i];
        }
      }
    }
    return *(size_);
  }

  BLITZ_DATA_LAYOUT data_layout() const {
    return this->data_layout_;
  }

  // operator
  size_t operator[](size_t index) const {
    return shape_[index];
  }

  size_t& operator[](size_t index) {
    return shape_[index];
  }

  // copy assignment
  Shape& operator=(const Shape& other) {  // check for self-assignment
    if(&other == this)
      return *this;  // reuse storage when possible
    // copy data fields
    size_ = 0;
    dimension_ = other.dimension_;
    shape_ = other.shape_;
    data_layout_ = other.data_layout_;
    return *this;
  }  // note: copy-and-swap would always cause a reallocation

  ~Shape() {
    delete this->size_;
  }

 private:
  mutable size_t* size_;
  size_t dimension_;
  std::vector<size_t> shape_;
  BLITZ_DATA_LAYOUT data_layout_;
};

}  // namespace blitz

#endif  // INCLUDE_BACKENDS_SHAPE_H_
