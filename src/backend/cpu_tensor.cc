#include "backend/cpu_tensor.h"

namespace blitz {

template<typename DType>
CPUTensor<DType>::~CPUTensor() {
  free(this->data_);
}

template<typename DType>
inline void CPUTensor<DType>::Fill(DType value) {
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

template<typename DType>
inline void CPUTensor<DType>::Reshape() {
}

template<typename DType>
inline DType* CPUTensor<DType>::Slice(size_t index) {
#ifdef BLITZ_DEVELOP
  CHECK_LT(index, this->shape_.size());
#endif  // BLITZ_DEVELOP
  return this->data_ + index;
}

template<typename DType>
inline const DType* CPUTensor<DType>::Slice(size_t index) const {
#ifdef BLITZ_DEVELOP
  CHECK_LT(index, this->shape_.size());
#endif  // BLITZ_DEVELOP
  return this->data_ + index;
}

template<typename DType>
inline void CPUTensor<DType>::Allocate() {
  size_t size = this->shape_.size();
  // TODO(keren) aligned allocation
#ifdef BLITZ_ALIGNMENT_SIZE
  int ret = posix_memalign(reinterpret_cast<void**>(&(this->data_)),
    BLITZ_ALIGNMENT_SIZE, sizeof(DType) * size);
  if (ret) {
    LOG(ERROR) << "Alignment error!";
  }
  this->Fill(0);
#else
  this->data_ = reinterpret_cast<DType*>(
    calloc(size, sizeof(DType)));
#endif
}

template<typename DType>
inline void CPUTensor<DType>::OutputCSV(ofstream* ofs) const {
  size_t num_sample = this->shape_[0];
  size_t dim = this->size() / num_sample;

  for (size_t i = 0; i < num_sample; ++i) {
    for (size_t j = 0; j < dim; ++j) {
      (*ofs) << this->data_[i * dim + j] << ", ";
    }
    (*ofs) << std::endl;
  }
}

INSTANTIATE_TENSOR(CPUTensor);

}  // namespace blitz
