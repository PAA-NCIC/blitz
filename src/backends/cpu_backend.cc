#include "backends/backend.h"

#include <omp.h>

#include "backends/cpu_tensor.h"
#include "utils/blitz_cpu_function.h"
#include "utils/blitz_cpu_avx.h"

namespace blitz {

template<typename DType>
class Backend<CPUTensor, DType> {
 public:
#include "cpu_backend_math-inl.h"
#include "cpu_backend_conv-inl.h"
#include "cpu_backend_transform-inl.h"
#include "cpu_backend_pool-inl.h"
};

INSTANTIATE_BACKEND(CPUTensor);

}  // namespace blitz
