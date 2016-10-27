#ifndef INCLUDE_BACKENDS_MIC_TENSOR_H_
#define INCLUDE_BACKENDS_MIC_TENSOR_H_

#include "backends/tensor.h"

namespace blitz {

template<typename DType = float>
class MICTensor : public Tensor<DType> {
 public:
  explicit MICTensor(const Shape& shape) :
    Tensor<DType>(shape) {
    this->Allocate();
  }

  explicit MICTensor(DType* data, const Shape& shape) :
    Tensor<DType>(data, shape) {}

  ~MICTensor();

  virtual void Fill(DType value);
  virtual DType* Slice(size_t index);
  virtual const DType* Slice(size_t index) const;
  virtual void OutputCSV(ofstream* ofs) const;

 protected:
  virtual void Allocate();

  DISABLE_COPY_AND_ASSIGN(MICTensor);
};

}  // namespace blitz

#endif  // INCLUDE_BACKENDS_MIC_TENSOR_H_
