#ifndef INCLUDE_BACKEND_GPU_TENSOR_H_
#define INCLUDE_BACKEND_GPU_TENSOR_H_

#include "backend/tensor.h"

namespace blitz {

template<typename DType = float>
class GPUTensor : public Tensor<DType> {
 public:
  explicit GPUTensor(const Shape& shape) :
    Tensor<DType>(shape) {
    this->Allocate();
  }

  explicit GPUTensor(DType* data, const Shape& shape) :
    Tensor<DType>(data, shape) {}

  ~GPUTensor();

  virtual void Fill(const DType value);
  virtual DType* Slice(const size_t index);
  virtual const DType* Slice(size_t index) const;
  virtual void OutputCSV(ofstream* ofs) const;

 protected:
  virtual void Allocate();

  DISABLE_COPY_AND_ASSIGN(GPUTensor);
};

}  // namespace blitz

#endif  // INCLUDE_BACKEND_GPU_TENSOR_H_
