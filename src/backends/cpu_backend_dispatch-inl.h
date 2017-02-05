#ifndef SRC_BACKENDS_CPU_BACKEND_DISPATCH_INL_H_
#define SRC_BACKENDS_CPU_BACKEND_DISPATCH_INL_H_

template<typename DType>
void Backend<CPUTensor, DType>::Convolution2DForwardGEMMDispatch(
  DType* unpack,
  DType* output,
  DType* filter,
  size_t K, size_t PQ, size_t CRS,
  BLITZ_DATA_LAYOUT input_data_layout,
  BLITZ_DATA_LAYOUT output_data_layout) {
  if (input_data_layout == BLITZ_BUFFER_NCHW) {
    if (output_data_layout == BLITZ_BUFFER_NCHW) {
      BlitzCPUGemm(filter, // KCRS
        unpack, // CRSPQ
        output, // KPQ
        false, false,
        static_cast<DType>(1), static_cast<DType>(0),
        K, PQ, CRS);
    } else if (output_data_layout == BLITZ_BUFFER_NHWC) {
      BlitzCPUGemm(unpack, // CRSPQ
        filter, // KCRS
        output, // PQK
        true, true,
        static_cast<DType>(1), static_cast<DType>(0),
        PQ, K, CRS);
    } else {
      LOG(FATAL) << "Unsupported output data layout: " << output_data_layout;
    }
  } else if (input_data_layout == BLITZ_BUFFER_NHWC) {
    if (output_data_layout == BLITZ_BUFFER_NCHW) {
      BlitzCPUGemm(filter, // KRSC
        unpack, // PQRSC
        output, // KPQ
        false, true,
        static_cast<DType>(1), static_cast<DType>(0),
        K, PQ, CRS);
    } else if (output_data_layout == BLITZ_BUFFER_NHWC) {
      BlitzCPUGemm(unpack, // PQRSC
        filter, // KRSC
        output, // PQK
        false, true,
        static_cast<DType>(1), static_cast<DType>(0),
        PQ, K, CRS);
    } else {
      LOG(FATAL) << "Unsupported output data layout: " << output_data_layout;
    }
  }
}

template<typename DType>
void Backend<CPUTensor, DType>::Convolution2DBackwardGEMMDispatch(
  DType* filter,
  DType* output,
  DType* unpack,
  size_t K, size_t PQ, size_t CRS,
  BLITZ_DATA_LAYOUT input_data_layout,
  BLITZ_DATA_LAYOUT output_data_layout) {
  if (input_data_layout == BLITZ_BUFFER_NCHW) {
    if (output_data_layout == BLITZ_BUFFER_NCHW) {
      BlitzCPUGemm(filter, // KCRS
        output, // KPQ
        unpack, // CRSPQ
        true, false,
        static_cast<DType>(1), static_cast<DType>(0),
        CRS, PQ, K);
    } else if (output_data_layout == BLITZ_BUFFER_NHWC) {
      BlitzCPUGemm(filter, // KCRS
        output, // PQK
        unpack, // CRSPQ
        true, true,
        static_cast<DType>(1), static_cast<DType>(0),
        CRS, PQ, K);
    } else {
      LOG(FATAL) << "Unsupported input data layout: " << output_data_layout;
    }
  } else if (input_data_layout == BLITZ_BUFFER_NHWC) {
    if (output_data_layout == BLITZ_BUFFER_NCHW) {
      BlitzCPUGemm(output, // KPQ
        filter, // KRSC
        unpack, // PQRSC
        true, false,
        static_cast<DType>(1), static_cast<DType>(0),
        PQ, CRS, K);
    } else if (output_data_layout == BLITZ_BUFFER_NHWC) {
      BlitzCPUGemm(output, // PQK
        filter, // KRSC
        unpack, // PQRSC
        false, false,
        static_cast<DType>(1), static_cast<DType>(0),
        PQ, CRS, K);
    } else {
      LOG(FATAL) << "Unsupported input data layout: " << output_data_layout;
    }
  }
}

template<typename DType>
void Backend<CPUTensor, DType>::Convolution2DUpdateGEMMDispatch(
  DType* unpack,
  DType* output,
  DType* update,
  size_t K, size_t CRS, size_t PQ,
  BLITZ_DATA_LAYOUT input_data_layout,
  BLITZ_DATA_LAYOUT output_data_layout) {
  if (input_data_layout == BLITZ_BUFFER_NCHW) {
    if (output_data_layout == BLITZ_BUFFER_NCHW) {
      BlitzCPUGemm(output, // KPQ
        unpack, // CRSPQ
        update, // KCRS
        false, true,
        static_cast<DType>(1), static_cast<DType>(1),
        K, CRS, PQ);
    } else if (output_data_layout == BLITZ_BUFFER_NHWC) {
      BlitzCPUGemm(output, // PQK
        unpack, // CRSPQ
        update, // KCRS
        true, true,
        static_cast<DType>(1), static_cast<DType>(1),
        K, CRS, PQ);
    } else {
      LOG(FATAL) << "Unsupported output data layout: " << output_data_layout;
    }
  } else if (input_data_layout == BLITZ_BUFFER_NHWC) {
    if (output_data_layout == BLITZ_BUFFER_NCHW) {
      BlitzCPUGemm(output, // KPQ
        unpack, // PQRSC
        update, // KRSC
        false, false,
        static_cast<DType>(1), static_cast<DType>(1),
        K, CRS, PQ);
    } else if (output_data_layout == BLITZ_BUFFER_NHWC) {
      BlitzCPUGemm(output, // PQK
        unpack, // PQRSC
        update, // KRSC
        true, false,
        static_cast<DType>(1), static_cast<DType>(1),
        K, CRS, PQ);
    } else {
      LOG(FATAL) << "Unsupported output data layout: " << output_data_layout;
    }
  }
}

#endif  // SRC_BACKENDS_CPU_BACKEND_DISPATCH_INL_H_
