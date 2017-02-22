#ifndef INCLUDE_UTIL_BLITZ_DISPATCH_FUNCTION_H_
#define INCLUDE_UTIL_BLITZ_DISPATCH_FUNCTION_H_

#include "backends/tensor.h"
#include "utils/blitz_define.h"
#include "utils/blitz_impl_function.h"

namespace blitz {

namespace utils {

template<template <typename> class TensorType, typename DType>
void Unpack2DDispatch(
  const DType *I,
  DType *U,
  size_t C, size_t H, size_t W,
  size_t R, size_t S,
  size_t P, size_t Q,
  size_t pad_h, size_t pad_w,
  size_t str_h, size_t str_w,
  BLITZ_DATA_LAYOUT input_data_layout) {
  switch (input_data_layout) {
    case BLITZ_BUFFER_NHWC:
      UnpackImpl<TensorType, DType, BLITZ_BUFFER_NHWC>(
        I, U,
        C, H, W, R, S, P, Q,
        pad_h, pad_w, str_h, str_w);
      break;
    case BLITZ_BUFFER_NCHW:
      UnpackImpl<TensorType, DType, BLITZ_BUFFER_NCHW>(
        I, U,
        C, H, W, R, S, P, Q,
        pad_h, pad_w, str_h, str_w);
      break;
    default:
      LOG(FATAL) << "Unsupported input data layout: " << input_data_layout;
  }
}

template<template <typename> class TensorType, typename DType>
static void Pack2DDispatch(
  const DType *U,
  DType *I,
  size_t C, size_t H, size_t W,
  size_t R, size_t S,
  size_t P, size_t Q,
  size_t pad_h, size_t pad_w,
  size_t str_h, size_t str_w,
  BLITZ_DATA_LAYOUT input_data_layout) {
  switch (input_data_layout) {
    case BLITZ_BUFFER_NHWC:
      PackImpl<TensorType, DType, BLITZ_BUFFER_NHWC>(
        U, I,
        C, H, W, R, S, P, Q,
        pad_h, pad_w, str_h, str_w);
      break;
    case BLITZ_BUFFER_NCHW:
      PackImpl<TensorType, DType, BLITZ_BUFFER_NCHW>(
        U, I,
        C, H, W, R, S, P, Q,
        pad_h, pad_w, str_h, str_w);
      break;
    default:
      LOG(FATAL) << "Unsupported input data layout: " << input_data_layout;
  }
}

template<template <typename> class TensorType, typename DType>
void Convolution2DForwardGEMMDispatch(
  const DType* U,
  const DType* F,
  DType* O,
  size_t K, size_t PQ, size_t CRS,
  BLITZ_DATA_LAYOUT input_data_layout,
  BLITZ_DATA_LAYOUT output_data_layout) {
  if (input_data_layout == BLITZ_BUFFER_NCHW) {
    if (output_data_layout == BLITZ_BUFFER_NCHW) {
      BlitzGemm<TensorType, DType>(const_cast<DType*>(F), // KCRS
        const_cast<DType*>(U), // CRSPQ
        O, // KPQ
        false, false,
        static_cast<DType>(1), static_cast<DType>(0),
        K, PQ, CRS);
    } else if (output_data_layout == BLITZ_BUFFER_NHWC) {
      BlitzGemm<TensorType, DType>(const_cast<DType*>(U), // CRSPQ
        const_cast<DType*>(F), // KCRS
        O, // PQK
        true, true,
        static_cast<DType>(1), static_cast<DType>(0),
        PQ, K, CRS);
    } else {
      LOG(FATAL) << "Unsupported output data layout: " << output_data_layout;
    }
  } else if (input_data_layout == BLITZ_BUFFER_NHWC) {
    if (output_data_layout == BLITZ_BUFFER_NCHW) {
      BlitzGemm<TensorType, DType>(const_cast<DType*>(F), // RSCK
        const_cast<DType*>(U), // PQRSC
        O, // KPQ
        true, true,
        static_cast<DType>(1), static_cast<DType>(0),
        K, PQ, CRS);
    } else if (output_data_layout == BLITZ_BUFFER_NHWC) {
      BlitzGemm<TensorType, DType>(const_cast<DType*>(U), // PQRSC
        const_cast<DType*>(F), // RSCK
        O, // PQK
        false, false,
        static_cast<DType>(1), static_cast<DType>(0),
        PQ, K, CRS);
    } else {
      LOG(FATAL) << "Unsupported output data layout: " << output_data_layout;
    }
  }
}

template<template <typename> class TensorType, typename DType>
void Convolution2DBackwardGEMMDispatch(
  const DType* F,
  const DType* O,
  DType* U,
  size_t K, size_t PQ, size_t CRS,
  BLITZ_DATA_LAYOUT input_data_layout,
  BLITZ_DATA_LAYOUT output_data_layout) {
  if (input_data_layout == BLITZ_BUFFER_NCHW) {
    if (output_data_layout == BLITZ_BUFFER_NCHW) {
      BlitzGemm<TensorType, DType>(const_cast<DType*>(F), // KCRS
        const_cast<DType*>(O), // KPQ
        U, // CRSPQ
        true, false,
        static_cast<DType>(1), static_cast<DType>(0),
        CRS, PQ, K);
    } else if (output_data_layout == BLITZ_BUFFER_NHWC) {
      BlitzGemm<TensorType, DType>(const_cast<DType*>(F), // KCRS
        const_cast<DType*>(O), // PQK
        U, // CRSPQ
        true, true,
        static_cast<DType>(1), static_cast<DType>(0),
        CRS, PQ, K);
    } else {
      LOG(FATAL) << "Unsupported input data layout: " << output_data_layout;
    }
  } else if (input_data_layout == BLITZ_BUFFER_NHWC) {
    if (output_data_layout == BLITZ_BUFFER_NCHW) {
      BlitzGemm<TensorType, DType>(const_cast<DType*>(O), // KPQ
        const_cast<DType*>(F), // RSCK
        U, // PQRSC
        true, true,
        static_cast<DType>(1), static_cast<DType>(0),
        PQ, CRS, K);
    } else if (output_data_layout == BLITZ_BUFFER_NHWC) {
      BlitzGemm<TensorType, DType>(const_cast<DType*>(O), // PQK
        const_cast<DType*>(F), // RSCK
        U, // PQRSC
        false, true,
        static_cast<DType>(1), static_cast<DType>(0),
        PQ, CRS, K);
    } else {
      LOG(FATAL) << "Unsupported input data layout: " << output_data_layout;
    }
  }
}

template<template <typename> class TensorType, typename DType>
void Convolution2DUpdateGEMMDispatch(
  const DType* U,
  const DType* O,
  DType* UP,
  size_t K, size_t CRS, size_t PQ,
  BLITZ_DATA_LAYOUT input_data_layout,
  BLITZ_DATA_LAYOUT output_data_layout) {
  if (input_data_layout == BLITZ_BUFFER_NCHW) {
    if (output_data_layout == BLITZ_BUFFER_NCHW) {
      BlitzGemm<TensorType, DType>(const_cast<DType*>(O), // KPQ
        const_cast<DType*>(U), // CRSPQ
        UP, // KCRS
        false, true,
        static_cast<DType>(1), static_cast<DType>(1),
        K, CRS, PQ);
    } else if (output_data_layout == BLITZ_BUFFER_NHWC) {
      BlitzGemm<TensorType, DType>(const_cast<DType*>(O), // PQK
        const_cast<DType*>(U), // CRSPQ
        UP, // KCRS
        true, true,
        static_cast<DType>(1), static_cast<DType>(1),
        K, CRS, PQ);
    } else {
      LOG(FATAL) << "Unsupported output data layout: " << output_data_layout;
    }
  } else if (input_data_layout == BLITZ_BUFFER_NHWC) {
    if (output_data_layout == BLITZ_BUFFER_NCHW) {
      BlitzGemm<TensorType, DType>(const_cast<DType*>(U), // PQRSC
        const_cast<DType*>(O), // KPQ
        UP, // RSCK
        true, true,
        static_cast<DType>(1), static_cast<DType>(1),
        CRS, K, PQ);
    } else if (output_data_layout == BLITZ_BUFFER_NHWC) {
      BlitzGemm<TensorType, DType>(const_cast<DType*>(U), // PQRSC
        const_cast<DType*>(O), // PQK
        UP, // RSCK
        true, false,
        static_cast<DType>(1), static_cast<DType>(1),
        CRS, K, PQ);
    } else {
      LOG(FATAL) << "Unsupported output data layout: " << output_data_layout;
    }
  }
}

}  // namespace utils

}  // namespace blitz

#endif  // INCLUDE_UTIL_BLITZ_DISPATCH_FUNCTION_H_
