#ifndef INCLUDE_UTIL_BLITZ_IMPL_FUNCTION_H_
#define INCLUDE_UTIL_BLITZ_IMPL_FUNCTION_H_

#include "backends/shape.h"

namespace blitz {

template<template <typename> class TensorType, typename DType, BLITZ_DATA_LAYOUT DataLayout>
void ConvolutionForwardNaiveImpl(
  const DType* I,
  const DType* F,
  DType* O,
  size_t N,
  size_t C, size_t H, size_t W,
  size_t R, size_t S,
  size_t K, size_t P, size_t Q,
  size_t pad_h, size_t pad_w,
  size_t str_h, size_t str_w);

template<template <typename> class TensorType, typename DType, BLITZ_DATA_LAYOUT DataLayout>
void ConvolutionBackwardNaiveImpl(
  const DType* O,
  const DType* F,
  DType* I,
  size_t N,
  size_t C, size_t H, size_t W,
  size_t R, size_t S,
  size_t K, size_t P, size_t Q,
  size_t pad_h, size_t pad_w,
  size_t str_h, size_t str_w);

template<template <typename> class TensorType, typename DType, BLITZ_DATA_LAYOUT DataLayout>
void ConvolutionUpdateNaiveImpl(
  const DType* I,
  const DType* O,
  DType* F,
  size_t N,
  size_t C, size_t H, size_t W,
  size_t R, size_t S,
  size_t K, size_t P, size_t Q,
  size_t pad_h, size_t pad_w,
  size_t str_h, size_t str_w);

template<template <typename> class TensorType, typename DType, BLITZ_DATA_LAYOUT DataLayout>
void ConvolutionForwardVectorImpl(
  const DType* I,
  const DType* F,
  DType* O,
  size_t N,
  size_t C, size_t H, size_t W,
  size_t R, size_t S,
  size_t K, size_t P, size_t Q,
  size_t pad_h, size_t pad_w,
  size_t str_h, size_t str_w);

template<template <typename> class TensorType, typename DType, BLITZ_DATA_LAYOUT DataLayout>
void ConvolutionBackwardVectorImpl(
  const DType* O,
  const DType* F,
  DType* I,
  size_t N,
  size_t C, size_t H, size_t W,
  size_t R, size_t S,
  size_t K, size_t P, size_t Q,
  size_t pad_h, size_t pad_w,
  size_t str_h, size_t str_w);

template<template <typename> class TensorType, typename DType, BLITZ_DATA_LAYOUT DataLayout>
void ConvolutionUpdateVectorImpl(
  const DType* I,
  const DType* O,
  DType* F,
  size_t N,
  size_t C, size_t H, size_t W,
  size_t R, size_t S,
  size_t K, size_t P, size_t Q,
  size_t pad_h, size_t pad_w,
  size_t str_h, size_t str_w);

template<template <typename> class TensorType, typename DType,
  BLITZ_DATA_LAYOUT SourceDataLayout, BLITZ_DATA_LAYOUT DestDataLayout>
void TransformBufferImpl(
  const DType* source,
  DType* dest,
  size_t N,
  size_t C, size_t H, size_t W);

template<template <typename> class TensorType, typename DType,
  BLITZ_DATA_LAYOUT SourceDataLayout, BLITZ_DATA_LAYOUT DestDataLayout>
void TransformFilterImpl(
  const DType* source,
  DType* dest,
  size_t K,
  size_t C, size_t R, size_t S);

template<template <typename> class TensorType, typename DType, BLITZ_DATA_LAYOUT DataLayout>
void PackImpl(
  const DType* I,
  DType* unpack,
  size_t C, size_t H, size_t W,
  size_t R, size_t S,
  size_t P, size_t Q,
  size_t pad_h, size_t pad_w,
  size_t str_h, size_t str_w);

template<template <typename> class TensorType, typename DType, BLITZ_DATA_LAYOUT DataLayout>
void UnpackImpl(
  const DType* I,
  DType* unpack,
  size_t C, size_t H, size_t W,
  size_t R, size_t S,
  size_t P, size_t Q,
  size_t pad_h, size_t pad_w,
  size_t str_h, size_t str_w);

template<template <typename> class TensorType, typename DType, BLITZ_DATA_LAYOUT DataLayout>
void MaxPoolingForwardImpl(
  const DType* I,
  DType* O,
  size_t* max_index,
  size_t N,
  size_t C, size_t H, size_t W,
  size_t K, size_t P, size_t Q,
  size_t R, size_t S,
  size_t str_h, size_t str_w);

template<template <typename> class TensorType, typename DType, BLITZ_DATA_LAYOUT DataLayout>
void MaxPoolingBackwardImpl(
  const DType* O,
  DType* I,
  const size_t* max_index,
  size_t N,
  size_t C, size_t H, size_t W,
  size_t K, size_t P, size_t Q);

template<template <typename> class TensorType, typename DType>
void BlitzGemm(
  DType* A, DType* B, DType* C,
  bool transa, bool transb,
  DType alpha, DType beta,
  size_t M, size_t N, size_t K);

}  // namespace blitz

#endif  // INCLUDE_UTIL_BLITZ_IMPL_FUNCTION_H_
