#include "backend/cpu_backend.h"

namespace blitz {

#include "backend/cpu_backend_common-inl.h"
#include "backend/cpu_backend_conv-inl.h"
#include "backend/cpu_backend_pack-inl.h"
#include "backend/cpu_backend_pool-inl.h"

INSTANTIATE_BACKEND(CPUTensor);

}  // namespace blitz
