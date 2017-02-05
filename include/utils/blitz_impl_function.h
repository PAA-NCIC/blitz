#ifndef INCLUDE_UTIL_BLITZ_IMPL_FUNCTION_H_
#define INCLUDE_UTIL_BLITZ_IMPL_FUNCTION_H_

namespace blitz {

template<template <typename> class TensorType, typename DType, typename DataLayout>
void PackImpl(
  const DType* I,
  DType* unpack,
  size_t C, size_t H, size_t W,
  size_t R, size_t S,
  size_t P, size_t Q,
  size_t pad_h, size_t pad_w,
  size_t str_h, size_t str_w);

template<template <typename> class TensorType, typename DType, typename DataLayout>
void UnpackCHWImpl(
  const DType* I,
  DType* unpack,
  size_t C, size_t H, size_t W,
  size_t R, size_t S,
  size_t P, size_t Q,
  size_t pad_h, size_t pad_w,
  size_t str_h, size_t str_w);

template<template <typename> class TensorType, typename DType, typename DataLayout>
void MaxPoolingForwardImpl(
  const DType* I,
  DType* O,
  size_t* max_index,
  size_t N,
  size_t C, size_t H, size_t W,
  size_t K, size_t P, size_t Q,
  size_t R, size_t S,
  size_t str_h, size_t str_w);

template<template <typename> class TensorType, typename DType, typename DataLayout>
void MaxPoolingBackwardImpl(
  const DType* O,
  DType* I,
  const size_t* max_index,
  size_t N,
  size_t C, size_t H, size_t W,
  size_t K, size_t P, size_t Q);

template<template <typename> class TensorType, typename DType, typename InputDataLayout, typename OutputDataLayout>
void Convolution2DForwardGEMMDispatch(
  DType* unpack,
  DType* O,
  DType* F,
  size_t K, size_t PQ, size_t CRS);

template<template <typename> class TensorType, typename DType, typename InputDataLayout, typename OutputDataLayout>
void Convolution2DBackwardGEMMDispatch(
  DType* F,
  DType* O,
  DType* unpack,
  size_t K, size_t PQ, size_t CRS);

template<template <typename> class TensorType, typename DType, typename InputDataLayout, typename OutputDataLayout>
void Convolution2DUpdateGEMMDispatch(
  DType* unpack,
  DType* O,
  DType* update,
  size_t K, size_t CRS, size_t PQ);

}  // namespace blitz

#endif  // INCLUDE_UTIL_BLITZ_IMPL_FUNCTION_H_
