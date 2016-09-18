#ifndef SRC_BACKEND_SHAPE_H_
#define SRC_BACKEND_SHAPE_H_

#include <vector>
#include <cstddef>

namespace blitz {

class Shape {
 public:
  explicit Shape(const size_t dimension) :
    size_(0), dimension_(dimension),
    shape_(dimension) {}

  explicit Shape(const std::vector<size_t>& shape) :
    size_(0), dimension_(shape.size()),
    shape_(shape) {}

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

  // operator
  size_t operator[](size_t index) const {
    // TODO(keren) index range check
    return shape_[index];
  }

  size_t& operator[](size_t index) {
    // TODO(keren) index range check
    return shape_[index];
  }

  Shape& operator=(const Shape& other) {  // check for self-assignment
    if(&other == this)
      return *this;  // reuse storage when possible

    // copy data fields
    size_ = 0;
    dimension_ = other.dimension_;
    shape_ = other.shape_;

    return *this;
  }  // note: copy-and-swap would always cause a reallocation

 private:
  mutable size_t* size_;
  size_t dimension_;
  std::vector<size_t> shape_;
};

}  // namespace blitz

#endif  // SRC_BACKEND_SHAPE_H_
