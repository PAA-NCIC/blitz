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

  size_t dimension() {
    return dimension_;
  }

  // operator
  const size_t& operator[](size_t index) const {
    // TODO(keren) index range check
    return shape_[index];
  }

  size_t& operator[](size_t index) {
    // TODO(keren) index range check
    return shape_[index];
  }

  size_t size() const {
    if (size_ == 0) {
      size_ = new size_t();

      (*size_) = 1;
      for (size_t i = 0; i < dimension_; ++i) {
        (*size_) *= shape_[i];
      }
    }

    return *(size_);
  }

 private:
  mutable size_t* size_;
  const size_t dimension_;
  std::vector<size_t> shape_;
};

}  // namespace blitz

#endif  // SRC_BACKEND_SHAPE_H_
