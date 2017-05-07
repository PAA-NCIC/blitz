#ifndef INCLUDE_UTIL_BLITZ_DEFINE_H_
#define INCLUDE_UTIL_BLITZ_DEFINE_H_

namespace blitz {

#ifdef BLITZ_SSE
#define CBLOCK 192
#define VEC_LEN 8  // register blocking
#define PQBLOCK 108 // divided by PQREG
#define KBLOCK 128 // divided by VEC_LEN * KREG
#define PQREG 4
#define KREG 6
#elif BLITZ_AVX
#define CBLOCK 192
#define VEC_LEN 8  // register blocking
#define PQBLOCK 108 // divided by PQREG
#define KBLOCK 128 // divided by VEC_LEN * KREG
#define PQREG 2
#define KREG 4
#elif BLITZ_AVX2
#define CBLOCK 192
#define VEC_LEN 8  // register blocking
#define PQBLOCK 108 // divided by PQREG
#define KBLOCK 128 // divided by VEC_LEN * KREG
#define PQREG 6
#define KREG 2
#define IREG 2
#elif BLITZ_AVX3
#define CBLOCK 192
#define VEC_LEN 8  // register blocking
#define PQBLOCK 108 // divided by PQREG
#define KBLOCK 128 // divided by VEC_LEN * KREG
#define PQREG 6
#define KREG 2
#define IREG 2
#elif BLITZ_AVX512
#define CBLOCK 80
#define VEC_LEN 16  // register blocking
#define PQBLOCK 72 // divided by PQREG
#define KBLOCK 192 // divided by VEC_LEN * KREG
#define PQREG 4
#define KREG 6
#endif
#define NBLOCK 1

enum BLITZ_ALGORITHM {
  BLITZ_CONVOLUTION_SASS_GEMM = 0,
  BLITZ_CONVOLUTION_SASS_DIRECT = 1,
  BLITZ_CONVOLUTION_CUDNN = 2,
  BLITZ_CONVOLUTION_BLAS_GEMM = 3,
  BLITZ_CONVOLUTION_BLAS_GEMM_BATCH = 4,
  BLITZ_CONVOLUTION_XSMM_DIRECT = 5,
  BLITZ_CONVOLUTION_NAIVE_DIRECT = 6,
  BLITZ_CONVOLUTION_VECTOR_DIRECT = 7,
  BLITZ_BLAS_GEMM = 8,
  BLITZ_SASS_GEMM = 9,
  BLITZ_ALGORITHM_UNDEFINED = 10
};

enum BLITZ_DATA_LAYOUT {
  BLITZ_FLAT = 0,
  BLITZ_BUFFER_NCHW = 1,
  BLITZ_BUFFER_NHWC = 2,
  BLITZ_FILTER_KCRS = 3,
  BLITZ_FILTER_RSCK = 4,
  BLITZ_UNPACK_PQRSC = 5,
  BLITZ_UNPACK_PQCRS = 6,
  BLITZ_UNPACK_CRSPQ = 7,
  BLITZ_UNPACK_RSCPQ = 8,
  BLITZ_SHAPE_UNDEFINED = 9
};

}  // namespace blitz

#endif  // INCLUDE_UTIL_BLITZ_ALGORITHM_FUNCTION_H_

