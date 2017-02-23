#include "backends/backend.h"

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/count.h>
#include <thrust/functional.h>
#include <thrust/transform_reduce.h>

#include "backends/gpu_tensor.h"
#include "utils/blitz_gpu_function.h"

namespace blitz {

template<typename DType>
class Backend<GPUTensor, DType> {
 public:
#include "gpu_backend_math-inl.h"
#include "gpu_backend_conv-inl.h"
#include "gpu_backend_transform-inl.h"
#include "gpu_backend_pool-inl.h"
};

INSTANTIATE_BACKEND(GPUTensor);

}  // namespace blitz
