#ifndef SRC_BACKEND_BACKENDS_H_
#define SRC_BACKEND_BACKENDS_H_

#ifdef CPU_ONLY
#include "backend/cpu_backend.h"
#include "backend/cpu_tensor.h"
#else
#include "backend/cpu_backend.h"
#include "backend/cpu_tensor.h"
#include "backend/gpu_backend.h"
#include "backend/gpu_tensor.h"
#endif

#endif  // SRC_BACKEND_BACKENDS_H_
