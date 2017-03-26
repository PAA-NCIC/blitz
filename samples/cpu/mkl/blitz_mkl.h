#include <cstdio>
#include <cstdlib>
#include <mkl_dnn.h>

#define BLITZ_CPU_TIMER_START(elapsed_time, t1) \
  do { \
    elapsed_time = 0.0; \
    gettimeofday(&t1, NULL); \
  } while (0) 

#define BLITZ_CPU_TIMER_END(elapsed_time, t1, t2) \
  do { \
    gettimeofday(&t2, NULL); \
    elapsed_time = (t2.tv_sec - t1.tv_sec) * 1000.0; \
    elapsed_time += (t2.tv_usec - t1.tv_usec) / 1000.0; \
    elapsed_time /= 1000.0; \
  } while (0)

#define BLITZ_CPU_TIMER_INFO(computations, elapsed_time) \
  do { \
    printf("Running time: %f\n", elapsed_time);\
    printf("GFLOPS: %f\n", computations / (elapsed_time * 1e9)); \
  } while (0) \

#define MKL_CHECK_ERR(f, err) do { \
  (err) = (f); \
  if ((err) != E_SUCCESS) { \
    printf("[%s:%d] err (%d)\n", __FILE__, __LINE__, err); \
    exit(1); \
  } \
} while(0)

static dnnError_t init_conversion(dnnPrimitive_t *cv, float **ptr_out,
  dnnLayout_t lt_pr, dnnLayout_t lt_us, float *ptr_us) {
  dnnError_t err;
  *ptr_out = NULL;
  if (!dnnLayoutCompare_F32(lt_pr, lt_us)) {
    MKL_CHECK_ERR(dnnConversionCreate_F32(cv, lt_us, lt_pr), err);
    MKL_CHECK_ERR(dnnAllocateBuffer_F32((void**)ptr_out, lt_pr), err);
  } else {
    *ptr_out = ptr_us;
  }
  return E_SUCCESS;
}
