#ifndef INCLUDE_BACKENDS_CPU_TENSOR_H_
#define INCLUDE_BACKENDS_CPU_TENSOR_H_

#include "backends/tensor.h"

namespace blitz {

template<typename DType = float>
class CPUTensor : public Tensor<DType> {
 public:
  explicit CPUTensor(const Shape& shape) :
    Tensor<DType>(shape) {
    this->Allocate();
  }

  explicit CPUTensor(DType* data, const Shape& shape) :
    Tensor<DType>(data, shape) {}

  ~CPUTensor();

  virtual void Fill(DType value);
  virtual DType* Slice(size_t index);
  virtual const DType* Slice(size_t index) const;
  virtual void OutputCSV(ofstream* ofs) const;

 protected:
  virtual void Allocate();

  DISABLE_COPY_AND_ASSIGN(CPUTensor);
};

}  // namespace blitz

#endif  // INCLUDE_BACKENDS_CPU_TENSOR_H_
