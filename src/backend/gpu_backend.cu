#include "backend/gpu_backend.h"

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cudnn.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/count.h>
#include <thrust/functional.h>
#include <thrust/transform_reduce.h>

#include <string>
#include <vector>

#include "util/common.h"
#include "util/blitz_gpu_function.h"
#include "kernel/sass_function.h"

namespace blitz {

#include "gpu_backend_common-inl.h"
#include "gpu_backend_conv-inl.h"
#include "gpu_backend_pack-inl.h"
#include "gpu_backend_pool-inl.h"

INSTANTIATE_BACKEND(GPUTensor);

}  // namespace blitz
