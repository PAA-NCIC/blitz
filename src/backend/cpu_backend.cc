#include "backend/cpu_backend.h"

#include <omp.h>

#include "util/common.h"
#include "util/blitz_cpu_function.h"
#include "util/blitz_cpu_avx.h"

namespace blitz {

#include "cpu_backend_common-inl.h"
#include "cpu_backend_conv-inl.h"
#include "cpu_backend_pack-inl.h"
#include "cpu_backend_pool-inl.h"

INSTANTIATE_BACKEND(CPUTensor);

}  // namespace blitz
