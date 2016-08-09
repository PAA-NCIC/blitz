#ifndef SRC_BACKEND_CPU_TENSOR_H_
#define SRC_BACKEND_CPU_TENSOR_H_

#include "util/common.h"
#include "backend/tensor.h"

namespace blitz {

template<typename DType = float>
class CPUTensor : public Tensor<DType> {
 public:
  explicit CPUTensor(const Shape& shape) : Tensor<DType>(shape) {
    this->Allocate();
  }

  explicit CPUTensor(DType* data, const Shape& shape) :
    Tensor<DType>(data, shape) {}

  ~CPUTensor() {
    free(this->data_);
  }

  virtual void Fill(DType value) {
    size_t size = this->shape_.size();
    if (value == 0) {
      memset(this->data_, 0, sizeof(DType) * size);
    } else {
      #pragma omp parallel for
      for (size_t i = 0; i < size; ++i) {
        *(this->data_ + i) = value;
      }
    }
  }

  virtual void Reshape() {
  }

  virtual DType* Slice(size_t index) {
#ifdef BLITZ_DEVELOP
    CHECK_LT(index, this->shape_.size());
#endif  // BLITZ_DEVELOP
    return this->data_ + index;
  }

  virtual const DType* Slice(size_t index) const {
#ifdef BLITZ_DEVELOP
    CHECK_LT(index, this->shape_.size());
#endif  // BLITZ_DEVELOP
    return this->data_ + index;
  }

 protected:
  virtual void Allocate() {
    size_t size = this->shape_.size();
    // TODO(keren) aligned allocation
#ifdef BLITZ_ALIGNMENT_SIZE
    int ret = posix_memalign((void**)&(this->data_), BLITZ_ALIGNMENT_SIZE,
      sizeof(DType) * size);
    if (ret) {
      LOG(ERROR) << "Alignment error!";
    }
    this->Fill(0); 
#else
    this->data_ = (DType*)calloc(size, sizeof(DType));
#endif
  }
};

}  // namespace blitz

#endif  // SRC_BACKEND_CPU_TENSOR_H_
