#include "backends/mic_backend.h"

#include <omp.h>

#include "utils/blitz_cpu_function.h"
#include "kernels/xsmm_function.h"

namespace blitz {

#include "mic_backend_common-inl.h"
#include "mic_backend_conv-inl.h"
#include "mic_backend_pack-inl.h"
#include "mic_backend_pool-inl.h"

INSTANTIATE_BACKEND(MICTensor);

}  // namespace blitz
