#ifndef INCLUDE_BACKEND_BACKENDS_H_
#define INCLUDE_BACKEND_BACKENDS_H_

#ifdef BLITZ_CPU_ONLY
  #include "backend/cpu_backend.h"
  #include "backend/cpu_tensor.h"
#else
  #include "backend/cpu_backend.h"
  #include "backend/cpu_tensor.h"
  #include "backend/gpu_backend.h"
  #include "backend/gpu_tensor.h"
#endif

#endif  // INCLUDE_BACKEND_BACKENDS_H_
