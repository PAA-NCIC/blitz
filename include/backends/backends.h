#ifndef INCLUDE_BACKENDS_BACKENDS_H_
#define INCLUDE_BACKENDS_BACKENDS_H_

#include "backends/backend.h"
#include "backends/cpu_tensor.h"
#include "utils/blitz_cpu_function.h"
#ifdef BLITZ_USE_GPU
#include "utils/blitz_gpu_function.h"
#include "backends/gpu_tensor.h"
#endif
#ifdef BLITZ_USE_MIC
#include "backends/mic_tensor.h"
#endif

#endif  // INCLUDE_BACKENDS_BACKENDS_H_
