#ifndef INCLUDE_BACKENDS_BACKENDS_H_
#define INCLUDE_BACKENDS_BACKENDS_H_

#include "backends/cpu_backend.h"
#include "backends/cpu_tensor.h"
#ifdef BLITZ_USE_GPU
  #include "backends/gpu_backend.h"
  #include "backends/gpu_tensor.h"
#endif
#ifdef BLITZ_USE_MIC
  #include "backends/mic_backend.h"
  #include "backends/mic_tensor.h"
#endif

#endif  // INCLUDE_BACKENDS_BACKENDS_H_
