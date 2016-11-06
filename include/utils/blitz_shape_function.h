#ifndef INCLUDE_UTIL_BLITZ_SHAPE_FUNCTION_H_
#define INCLUDE_UTIL_BLITZ_SHAPE_FUNCTION_H_

#include "backends/shape.h"

namespace blitz {

inline BLITZ_DATA_LAYOUT BlitzParseShape(const string& data_layout) {
	if (data_layout == "convolution_nchw") {
		return BLITZ_CONVOLUTION_BUFFER_NCHW;
	} else if (data_layout == "convolution_nhwc") {
		return BLITZ_CONVOLUTION_BUFFER_NHWC;
	} else if (data_layout == "convolution_kcrs") {
		return BLITZ_CONVOLUTION_FILTER_KCRS;
	} else if (data_layout == "convolution_rsck") {
		return BLITZ_CONVOLUTION_FILTER_RSCK;
	} else if (data_layout == "pooling_nchw") {
		return BLITZ_POOLING_BUFFER_NCHW;
	} else if (data_layout == "pooling_nhwc") {
		return BLITZ_POOLING_BUFFER_NHWC;
	} else if (data_layout == "flat") {
		return BLITZ_FLAT;
	} else {
		return BLITZ_UNDEFINED;
	}
}

inline void BlitzConvolution2DShape(BLITZ_DATA_LAYOUT data_layout,
	const Shape* shape, size_t* N, size_t* C, size_t* H, size_t* W) {
	CHECK_EQ(shape->dimension(), 4);
	switch (data_layout) {
		case BLITZ_CONVOLUTION_BUFFER_NCHW:
			*N = (*shape)[0];
			*C = (*shape)[1];
			*H = (*shape)[2];
			*W = (*shape)[3];
			break;
		case BLITZ_CONVOLUTION_BUFFER_NHWC:
			*N = (*shape)[0];
			*H = (*shape)[1];
			*W = (*shape)[2];
			*C = (*shape)[3];
			break;
		default:
			LOG(FATAL) << "Blitz unsupport convolution data layout: " << data_layout;
			break;
	}
}

inline void BlitzConvolution2DFilter(BLITZ_DATA_LAYOUT data_layout,
	const Shape* shape, size_t* K, size_t* C, size_t* R, size_t* S) {
	CHECK_EQ(shape->dimension(), 4);
	switch (data_layout) {
		case BLITZ_CONVOLUTION_FILTER_KCRS:
			*K = (*shape)[0];
			*C = (*shape)[1];
			*R = (*shape)[2];
			*R = (*shape)[3];
			break;
		case BLITZ_CONVOLUTION_FILTER_RSCK:
			*R = (*shape)[0];
			*S = (*shape)[1];
			*C = (*shape)[2];
			*K = (*shape)[3];
			break;
		default:
			LOG(FATAL) << "Blitz unsupport convolution data layout: " << data_layout;
			break;
	}
}

inline void BlitzPooling2DShape(BLITZ_DATA_LAYOUT data_layout,
	const Shape* shape, size_t* N, size_t* C, size_t* H, size_t* W) {
	CHECK_EQ(shape->dimension(), 4);
	switch (data_layout) {
		case BLITZ_POOLING_BUFFER_NCHW:
			*N = (*shape)[0];
			*C = (*shape)[1];
			*H = (*shape)[2];
			*W = (*shape)[3];
			break;
		case BLITZ_POOLING_BUFFER_NHWC:
			*N = (*shape)[0];
			*H = (*shape)[1];
			*W = (*shape)[2];
			*C = (*shape)[3];
			break;
		default:
			LOG(FATAL) << "Blitz unsupport convolution data layout: " << data_layout;
			break;
	}
}

}  // namespace blitz

#endif  // INCLUDE_UTIL_BLITZ_SHAPE_FUNCTION_H_

