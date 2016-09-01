#include "backend/gpu_tensor.h"

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

namespace blitz {

template<typename DType>
GPUTensor<DType>::~GPUTensor() {
  cudaFree(this->data_);
}

template<typename DType>
inline void GPUTensor<DType>::Fill(DType value) {
  if (value == 0) {
    cudaMemset(this->data_, 0, sizeof(DType) * this->size());
  } else {
    // TODO(Keren)
    thrust::device_ptr<DType> dptr =
      thrust::device_pointer_cast(this->data_);
    thrust::fill(dptr, dptr + this->size(), value);
  }
}

template<typename DType>
inline void GPUTensor<DType>::Reshape() {
}

template<typename DType>
inline DType* GPUTensor<DType>::Slice(size_t index) {
  // TODO(keren) error
  return this->data_ + index;
}

template<typename DType>
inline const DType* GPUTensor<DType>::Slice(size_t index) const {
  // TODO(keren) error
  return this->data_ + index;
}

template<typename DType>
inline void GPUTensor<DType>::Allocate() {
  cudaMalloc(&(this->data_), sizeof(DType) * this->size());
  this->Fill(0);
}

template<typename DType>
inline void GPUTensor<DType>::OutputCSV(ofstream* ofs) const {
}

INSTANTIATE_TENSOR(GPUTensor);

}  // namespace blitz
