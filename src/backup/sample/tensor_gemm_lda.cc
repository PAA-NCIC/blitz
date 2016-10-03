#include <iostream>
#include "backend/backends.h"
#include "util/blitz_cpu_function.h"

using namespace blitz;

int main() {
  int left = 5;
  int right = 8;
  int common = 8;
  float* A = (float*)malloc(sizeof(float) * left * common);
  float* B = (float*)malloc(sizeof(float) * right * common);
  float* C = (float*)malloc(sizeof(float) * left * right);
  for (int i = 0; i < left * common; ++i) {
    A[i] = 1;
  }
  for (int i = 0; i < right * common; ++i) {
    B[i] = 1;
  }
  for (int i = 0; i < right * left; ++i) {
    C[i] = 0;
  }
  CBLAS_TRANSPOSE TransA = CblasTrans;
  int lda = left;
  CBLAS_TRANSPOSE TransB = CblasNoTrans;
  int ldb = right;
  cblas_sgemm(CblasRowMajor, TransA, TransB, 4, right, common, 1.0, A, lda,
      B, ldb, 0.0, C, ldb);
  cblas_sgemm(CblasRowMajor, TransA, TransB, 1, right, common, 1.0, A + 4, lda,
      B, ldb, 0.0, C + 4 * right, ldb);
  for (int i = 0; i < left; ++i) {
    for (int j = 0; j < right; ++j) {
      std::cout << C[i * right + j] << " ";
    }
    std::cout << std::endl;
  }
  return 0;
}
