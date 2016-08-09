#ifndef SRC_BACKEND_GPU_TENSOR_H_
#define SRC_BACKEND_GPU_TENSOR_H_

#include "backend/tensor.h"

namespace blitz {

template<typename DType = float>
class GPUTensor : public Tensor<DType> {
 public:
  explicit GPUTensor(const Shape& shape) : Tensor<DType>(shape) {
    this->Allocate();
  }

  explicit GPUTensor(shared_ptr<DType> data, const Shape& shape) :
    Tensor<DType>(data, shape) {}

  virtual void Fill(const DType value) {
  }

  virtual void Reshape() {
  }

  virtual DType* Slice(const size_t index) {
    return new DType(0);
  }

  virtual const DType* Slice(size_t index) const {
    return new DType(0);
  }

 protected:
  virtual void Allocate() {
  }
};

}  // namespace blitz

#endif  // SRC_BACKEND_GPU_TENSOR_H_
