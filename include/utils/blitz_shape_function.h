#ifndef INCLUDE_UTIL_BLITZ_SHAPE_FUNCTION_H_
#define INCLUDE_UTIL_BLITZ_SHAPE_FUNCTION_H_

#include "backends/shape.h"

namespace blitz {

inline BLITZ_DATA_LAYOUT BlitzParseShape(const string& data_layout) {
  if (data_layout == "nchw") {
    return BLITZ_BUFFER_NCHW;
  } else if (data_layout == "nhwc") {
    return BLITZ_BUFFER_NHWC;
  } else if (data_layout == "kcrs") {
    return BLITZ_FILTER_KCRS;
  } else if (data_layout == "rsck") {
    return BLITZ_FILTER_RSCK;
  } else {
    return BLITZ_SHAPE_UNDEFINED;
  }
}

inline void Blitz2DBuffer(const Shape& shape, size_t* N, size_t* C, size_t* H, size_t* W) {
  CHECK_EQ(shape.dimension(), 4);
  switch (shape.data_layout()) {
    case BLITZ_BUFFER_NCHW:
      *N = shape[0];
      *C = shape[1];
      *H = shape[2];
      *W = shape[3];
      break;
    case BLITZ_BUFFER_NHWC:
      *N = shape[0];
      *H = shape[1];
      *W = shape[2];
      *C = shape[3];
      break;
    default:
      LOG(FATAL) << "Blitz unsupport convolution data layout: " << shape.data_layout();
      break;
  }
}

inline void Blitz2DFilter(const Shape& shape, size_t* K, size_t* C, size_t* R, size_t* S) {
  CHECK_EQ(shape.dimension(), 4);
  switch (shape.data_layout()) {
    case BLITZ_FILTER_KCRS:
      *K = shape[0];
      *C = shape[1];
      *R = shape[2];
      *S = shape[3];
      break;
    case BLITZ_FILTER_RSCK:
      *R = shape[0];
      *S = shape[1];
      *C = shape[2];
      *K = shape[3];
      break;
    default:
      LOG(FATAL) << "Blitz unsupport convolution data layout: " << shape.data_layout();
      break;
  }
}

}  // namespace blitz

#endif  // INCLUDE_UTIL_BLITZ_SHAPE_FUNCTION_H_

