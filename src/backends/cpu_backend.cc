#include "backends/cpu_backend.h"

#include <omp.h>

namespace blitz {

#include "cpu_backend_math-inl.h"
#include "cpu_backend_conv-inl.h"
#include "cpu_backend_transform-inl.h"
#include "cpu_backend_pool-inl.h"
#include "cpu_backend_dispatch-inl.h"

INSTANTIATE_BACKEND(CPUTensor);

}  // namespace blitz
