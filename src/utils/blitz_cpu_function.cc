#include "utils/blitz_cpu_function.h"

namespace blitz {

template<>
void BlitzCPUGemm<float>(
  float* A, float* B, float* C,
  bool transa, bool transb,
  float alpha, float beta,
  size_t M, size_t N, size_t K) {
  CBLAS_TRANSPOSE TransA = transa ? CblasTrans : CblasNoTrans;
  size_t lda = transa ? M : K;
  CBLAS_TRANSPOSE TransB = transb ? CblasTrans : CblasNoTrans;
  size_t ldb = transb ? K : N;
  cblas_sgemm(CblasRowMajor,
    TransA, TransB, 
    M, N, K,
    alpha,
    A, lda, B, ldb,
    beta,
    C, N);
}

template<>
void BlitzCPUGemm<double>(
  double* A, double* B, double* C,
  bool transa, bool transb,
  double alpha, double beta,
  size_t M, size_t N, size_t K) {
  CBLAS_TRANSPOSE TransA = transa ? CblasTrans : CblasNoTrans;
  size_t lda = transa ? M : K;
  CBLAS_TRANSPOSE TransB = transb ? CblasTrans : CblasNoTrans;
  size_t ldb = transb ? K : N;
  cblas_dgemm(CblasRowMajor,
    TransA, TransB,
    M, N, K,
    alpha,
    A, lda,
    B, ldb,
    beta,
    C, N);
}

template<>
void BlitzCPUCopy<float>(const float* X, float* Y, size_t N) {
  cblas_scopy(N, X, 1, Y, 1);
}

template<>
void BlitzCPUCopy<double>(const double* X, double* Y, size_t N) {
  cblas_dcopy(N, X, 1, Y, 1);
}

}  // namespace blitz

