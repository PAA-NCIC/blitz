#ifndef SRC_BACKEND_TENSOR_H_
#define SRC_BACKEND_TENSOR_H_

#include "util/common.h"
#include "backend/shape.h"

namespace blitz {

template<typename DType = float>
class Tensor {
 public:
  explicit Tensor(const Shape& shape) :
    shape_(shape), start_index_(0) {}

  explicit Tensor(DType* data, const Shape& shape,
    const size_t start_index) :
    data_(data), shape_(shape), start_index_(start_index) {}

  virtual ~Tensor() {}

  // getter
  const Shape& shape() const {
    return shape_;
  }

  size_t size() const {
    return this->shape_.size();
  }

  // operator
  DType& operator[](size_t index) const {
#ifdef BLITZ_DEVELOP
    CHECK_LT(index, this->shape_.size());
#endif  // BLITZ_DEVELOP
    return data_[index];
  }

  DType& operator[](size_t index) {
#ifdef BLITZ_DEVELOP
    CHECK_LT(index, this->shape_.size());
#endif  // BLITZ_DEVELOP
    return data_[index];
  }

  DType* data() {
    return data_;
  }

  const DType* data() const {
    return data_;
  }

  virtual void Fill(const DType value) = 0;
  virtual void Reshape() = 0;
  virtual DType* Slice(const size_t index) = 0;
  virtual const DType* Slice(const size_t index) const = 0;

 protected:
  virtual void Allocate() = 0;

  DType* data_;
  const Shape shape_;
  const size_t start_index_;
};

}  // namespace blitz

#endif  // SRC_BACKEND_TENSOR_H_
