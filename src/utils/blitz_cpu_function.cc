#include "utils/blitz_cpu_function.h"
#include "utils/common.h"

namespace blitz {

template<>
void BlitzCPUGemm<float>(
  bool transa, bool transb,
  int M, int N, int K,
  float* A, float* B, float* C,
  float alpha, float beta) {
  CBLAS_TRANSPOSE TransA = transa ? CblasTrans : CblasNoTrans;
  int lda = transa ? M : K;
  CBLAS_TRANSPOSE TransB = transb ? CblasTrans : CblasNoTrans;
  int ldb = transb ? K : N;
  cblas_sgemm(CblasRowMajor, TransA, TransB, M, N, K, alpha, A, lda,
      B, ldb, beta, C, N);
}

template<>
void BlitzCPUGemm<double>(
  bool transa, bool transb,
  int M, int N, int K,
  double* A, double* B, double* C,
  double alpha, double beta) {
  CBLAS_TRANSPOSE TransA = transa ? CblasTrans : CblasNoTrans;
  int lda = transa ? M : K;
  CBLAS_TRANSPOSE TransB = transb ? CblasTrans : CblasNoTrans;
  int ldb = transb ? K : N;
  cblas_dgemm(CblasRowMajor, TransA, TransB, M, N, K, alpha, A, lda,
      B, ldb, beta, C, N);
}

template<>
void BlitzCPUCopy<float>(const float* X, int N, float* Y) {
  cblas_scopy(N, X, 1, Y, 1);
}

template<>
void BlitzCPUCopy<double>(const double* X, int N, double* Y) {
  cblas_dcopy(N, X, 1, Y, 1);
}

}  // namespace blitz

