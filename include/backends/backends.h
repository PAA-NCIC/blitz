#ifndef INCLUDE_BACKENDS_BACKENDS_H_
#define INCLUDE_BACKENDS_BACKENDS_H_

#ifdef BLITZ_CPU_ONLY
  #include "backends/cpu_backend.h"
  #include "backends/cpu_tensor.h"
#else
  #include "backends/cpu_backend.h"
  #include "backends/cpu_tensor.h"
  #include "backends/gpu_backend.h"
  #include "backends/gpu_tensor.h"
#endif

#endif  // INCLUDE_BACKENDS_BACKENDS_H_
