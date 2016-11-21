#include "backends/cpu_backend.h"

#include <omp.h>

#include "utils/blitz_cpu_function.h"
#include "utils/blitz_cpu_avx.h"
#include "utils/blitz_shape_function.h"

namespace blitz {

#include "cpu_backend_common-inl.h"
#include "cpu_backend_conv-inl.h"
#include "cpu_backend_pack-inl.h"
#include "cpu_backend_pool-inl.h"
#include "cpu_backend_dispatch-inl.h"

INSTANTIATE_BACKEND(CPUTensor);

}  // namespace blitz
