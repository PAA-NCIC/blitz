#ifndef INCLUDE_UTIL_BLITZ_ALGORITHM_FUNCTION_H_
#define INCLUDE_UTIL_BLITZ_ALGORITHM_FUNCTION_H_

namespace blitz {

enum BLITZ_ALGORITHM {
	BLITZ_CONVOLUTION_SASS_GEMM = 0,
	BLITZ_CONVOLUTION_SASS_DIRECT = 1,
	BLITZ_CONVOLUTION_CUDNN = 2,
	BLITZ_CONVOLUTION_BLAS_GEMM = 3,
	BLITZ_CONVOLUTION_BLAS_GEMM_BATCH = 4,
	BLITZ_CONVOLUTION_XSMM_DIRECT = 5,
	BLITZ_UNDEFINED = 6
};

inline BLITZ_ALGORITHM BlitzParseAlgorithm(const string& algorithm) {
	if (algorithm == "sass_gemm") {
		return BLITZ_CONVOLUTION_SASS_GEMM;
	} else if (algorithm == "sass_direct") {
		return BLITZ_CONVOLUTION_SASS_DIRECT;
	} else if (algorithm == "cudnn") {
		return BLITZ_CONVOLUTION_CUDNN;
	} else if (algorithm == "blas_gemm") {
		return BLITZ_CONVOLUTION_BLAS_GEMM;
	} else if (algorithm == "blas_gemm_batch") {
		return BLITZ_CONVOLUTION_BLAS_GEMM_BATCH;
	} else if (algorithm == "xsmm_direct") {
		return BLITZ_CONVOLUTION_XSMM_DIRECT;
	} else {
		return BLITZ_UNDEFINED;
	}
}

}  // namespace blitz

#endif  // INCLUDE_UTIL_BLITZ_ALGORITHM_FUNCTION_H_
