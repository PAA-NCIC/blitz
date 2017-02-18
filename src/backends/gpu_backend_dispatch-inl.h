#ifndef SRC_BACKENDS_GPU_BACKEND_DISPATCH_INL_H_
#define SRC_BACKENDS_GPU_BACKEND_DISPATCH_INL_H_

//static void Unpack2DDispatch(
//  const DType *input,
//  DType *unpack,
//  size_t C, size_t H, size_t W,
//  size_t R, size_t S,
//  size_t P, size_t Q,
//  size_t pad_h, size_t pad_w,
//  size_t str_h, size_t str_w,
//  BLITZ_DATA_LAYOUT input_data_layout) {
//  switch (input_data_layout) {
//    case BLITZ_BUFFER_NCHW:
//      UnpackImpl<GPUTensor, DType, BLITZ_BUFFER_NCHW>(
//        I, U,
//        C, H, W, R, S, P, Q,
//        pad_h, pad_w, str_h, str_w);
//      break;
//    default:
//      LOG(FATAL) << "Unsupported input data layout: " << input_data_layout;
//  }
//}
//
//static void Pack2DDispatch(
//  const DType *unpack,
//  DType *input,
//  size_t C, size_t H, size_t W,
//  size_t R, size_t S,
//  size_t P, size_t Q,
//  size_t pad_h, size_t pad_w,
//  size_t str_h, size_t str_w,
//  BLITZ_DATA_LAYOUT input_data_layout) {
//  switch (input_data_layout) {
//    case BLITZ_BUFFER_NCHW:
//      PackImpl<GPUTensor, DType, BLITZ_BUFFER_NCHW>(
//        U, I,
//        C, H, W, R, S, P, Q,
//        pad_h, pad_w, str_h, str_w);
//      break;
//    default:
//      LOG(FATAL) << "Unsupported input data layout: " << input_data_layout;
//  }
//}

#endif  // SRC_BACKENDS_GPU_BACKEND_DISPATCH_INL_H_
