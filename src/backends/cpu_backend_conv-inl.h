#ifndef SRC_BACKENDS_CPU_BACKEND_CONV_INL_H_
#define SRC_BACKENDS_CPU_BACKEND_CONV_INL_H_

template<typename DType>
void Backend<CPUTensor, DType>::Convolution2DForwardFunc(
  const CPUTensor<DType>* input,
  const CPUTensor<DType>* filter,
  CPUTensor<DType>* output,
  CPUTensor<DType>* workspace,
  size_t padding_height,
  size_t padding_width,
  size_t stride_height,
  size_t stride_width,
  BLITZ_ALGORITHM algorithm) {
  // shape decode
  size_t NIN, C, H, W;
  size_t KF, CF, R, S;
  size_t NOUT, K, P, Q;
  Blitz2DBuffer(input->data_layout(), input->shape_ptr(), &NIN, &C, &H, &W);
  Blitz2DFilter(filter->data_layout(), filter->shape_ptr(), &KF, &CF, &R, &S);
  Blitz2DBuffer(output->data_layout(), output->shape_ptr(), &NOUT, &K, &P, &Q);
  CHECK_EQ(NIN, NOUT);
  CHECK_EQ(KF, K);
  CHECK_EQ(CF, C);
  // offset
  size_t nCHW = 0;
  size_t nKPQ = 0;
  // dims
  const size_t CHW = C * H * W;
  const size_t PQ = P * Q;
  const size_t KPQ = K * PQ;
  const size_t CRS = C * R * S;
  // time counter
  #ifdef BLITZ_PERFORMANCE
  timeval start, end;
  double elapsed_time;
  BLITZ_CPU_TIMER_START(elapsed_time, start);
  #endif  // BLITZ_PERFORMANCE
  switch (algorithm) { // NCHW & NHWC
    case BLITZ_CONVOLUTION_BLAS_GEMM_BATCH: {
      #pragma omp parallel private(nCHW, nKPQ)
      {
        const size_t tid = omp_get_thread_num();
        const size_t workspace_unpack_offset = tid * CRS * PQ;
        DType* workspace_unpack_slice = workspace->Slice(workspace_unpack_offset);
        #pragma omp for
        for (size_t n = 0; n < NIN; ++n) {
          nCHW = n * CHW;
          nKPQ = n * KPQ;
          BLITZ_DATA_LAYOUT unpack_data_layout = Unpack2DFunc(
            input->Slice(nCHW),
            workspace_unpack_slice,
            C, H, W,
            R, S,
            P, Q,
            padding_height, padding_width,
            stride_height, stride_width,
            input->data_layout());
          Convolution2DForwardGEMMDispatch(workspace_unpack_slice,
            output->Slice(nKPQ),
            const_cast<CPUTensor<DType>*>(filter)->data(),
            K, PQ, CRS,
            unpack_data_layout,
            output->data_layout(),
            filter->data_layout());
        }
      }
      break;
    }
    case BLITZ_CONVOLUTION_BLAS_GEMM: {
      for (size_t n = 0; n < NIN; ++n) {
        nCHW = n * CHW;
        nKPQ = n * KPQ;
        BLITZ_DATA_LAYOUT unpack_data_layout = Unpack2DFunc(
	  input->Slice(nCHW),
          workspace->data(),
          C, H, W,
          R, S,
          P, Q,
          padding_height, padding_width,
          stride_height, stride_width,
          input->data_layout());
        Convolution2DForwardGEMMDispatch(workspace->data(),
          output->Slice(nKPQ),
          const_cast<CPUTensor<DType>*>(filter)->data(),
          K, PQ, CRS,
          unpack_data_layout,
          output->data_layout(),
          filter->data_layout());
      }
      break;
    }
    default:
      LOG(FATAL) << "Unsupported algorithm type: " << algorithm;
      break;
  }
  #ifdef BLITZ_PERFORMANCE
  double computations = static_cast<double>(KPQ) * static_cast<double>(CRS) * static_cast<double>(2 * NIN);
  BLITZ_CPU_TIMER_END(elapsed_time, start, end);
  BLITZ_CPU_TIMER_INFO(computations, elapsed_time);
  #endif  // BLITZ_PERFORMANCE
}

template<typename DType>
void Backend<CPUTensor, DType>::Convolution2DBackwardFunc(
  const CPUTensor<DType>* output,
  const CPUTensor<DType>* filter,
  CPUTensor<DType>* input,
  CPUTensor<DType>* workspace,
  size_t padding_height,
  size_t padding_width,
  size_t stride_height,
  size_t stride_width,
  BLITZ_ALGORITHM algorithm) {
  // shape decode
  size_t NIN, C, H, W;
  size_t KF, CF, R, S;
  size_t NOUT, K, P, Q;
  Blitz2DBuffer(input->data_layout(), input->shape_ptr(), &NIN, &C, &H, &W);
  Blitz2DFilter(filter->data_layout(), filter->shape_ptr(), &KF, &CF, &R, &S);
  Blitz2DBuffer(output->data_layout(), output->shape_ptr(), &NOUT, &K, &P, &Q);
  CHECK_EQ(NIN, NOUT);
  CHECK_EQ(KF, K);
  CHECK_EQ(CF, C);
  // dims
  const size_t CHW = C * H * W;
  const size_t PQ = P * Q;
  const size_t KPQ = K * PQ;
  const size_t CRS = C * R * S;
  // offset
  size_t nCHW = 0;
  size_t nKPQ = 0;
  input->Fill(0);
  // time counter
  #ifdef BLITZ_PERFORMANCE
  timeval start, end;
  double elapsed_time;
  BLITZ_CPU_TIMER_START(elapsed_time, start);
  #endif  // BLITZ_PERFORMANCE
  switch (algorithm) {
    case BLITZ_CONVOLUTION_BLAS_GEMM_BATCH: {
      #pragma omp parallel private(nCHW, nKPQ) 
      {
        const size_t tid = omp_get_thread_num();
        const size_t workspace_unpack_offset = tid * CRS * PQ;
        #ifdef BLITZ_PERFORMANCE
          #pragma omp for private(start, end)
        #else
          #pragma omp for
        #endif
        for (size_t n = 0; n < NIN; ++n) {
          nCHW = n * CHW;
          nKPQ = n * KPQ;
          #ifdef BLITZ_PERFORMANCE
          start = system_clock::now();
          #endif  // BLITZ_PERFORMANCE
          BlitzCPUGemm(const_cast<CPUTensor<DType>*>(filter)->data(),
            const_cast<CPUTensor<DType>*>(output)->Slice(nKPQ),
            workspace->Slice(workspace_unpack_offset),
            true, false,
            static_cast<DType>(1), static_cast<DType>(0),
            CRS, PQ, K);
          #ifdef BLITZ_PERFORMANCE
          end = system_clock::now();
          gemm_time[tid] += end - start;
          start = system_clock::now();
          #endif  // BLITZ_PERFORMANCE
          Pack2DFunc(workspace->Slice(workspace_unpack_offset),
            input->Slice(nCHW),
            C, H, W,
            R, S,
            P, Q,
            padding_height, padding_width,
            stride_height, stride_width);
        }
      }
      break;
    }
    case BLITZ_CONVOLUTION_BLAS_GEMM: {
      for (size_t n = 0; n < NIN; ++n) {
        nCHW = n * CHW;
        nKPQ = n * KPQ;
        BlitzCPUGemm(const_cast<CPUTensor<DType>*>(filter)->data(),
          const_cast<CPUTensor<DType>*>(output)->Slice(nKPQ),
          workspace->data(),
          true, false,
          static_cast<DType>(1), static_cast<DType>(0),
          CRS, PQ, K);
        Pack2DFunc(workspace->data(),
          input->Slice(nCHW),
          C, H, W,
          R, S,
          P, Q,
          padding_height, padding_width,
          stride_height, stride_width);
      }
      break;
    }
    default:
      LOG(FATAL) << "Unsupported algorithm type: " << algorithm;
      break;
  }
  #ifdef BLITZ_PERFORMANCE
  double computations = static_cast<double>(KPQ) * static_cast<double>(CRS) * static_cast<double>(2 * NIN);
  BLITZ_CPU_TIMER_END(elapsed_time, start, end);
  BLITZ_CPU_TIMER_INFO(computations, elapsed_time);
  #endif  // BLITZ_PERFORMANCE
}

template<typename DType>
void Backend<CPUTensor, DType>::Convolution2DUpdateFunc(
  const CPUTensor<DType>* input,
  const CPUTensor<DType>* output,
  CPUTensor<DType>* update,
  CPUTensor<DType>* workspace,
  size_t padding_height,
  size_t padding_width,
  size_t stride_height,
  size_t stride_width,
  BLITZ_ALGORITHM algorithm) {
  // shape decode
  size_t NIN, C, H, W;
  size_t KF, CF, R, S;
  size_t NOUT, K, P, Q;
  Blitz2DBuffer(input->data_layout(), input->shape_ptr(), &NIN, &C, &H, &W);
  Blitz2DFilter(update->data_layout(), update->shape_ptr(), &KF, &CF, &R, &S);
  Blitz2DBuffer(output->data_layout(), output->shape_ptr(), &NOUT, &K, &P, &Q);
  CHECK_EQ(NIN, NOUT);
  CHECK_EQ(KF, K);
  CHECK_EQ(CF, C);
  // dims
  const size_t CHW = C * H * W;
  const size_t PQ = P * Q;
  const size_t KPQ = K * PQ;
  const size_t CRS = C * R * S;
  // offset
  size_t nCHW = 0;
  size_t nKPQ = 0;
  workspace->Fill(0);
  // time counter
  #ifdef BLITZ_PERFORMANCE
  timeval start, end;
  double elapsed_time;
  BLITZ_CPU_TIMER_START(elapsed_time, start);
  #endif  // BLITZ_PERFORMANCE
  switch (algorithm) {
    case BLITZ_CONVOLUTION_BLAS_GEMM_BATCH: {
      #pragma omp parallel private(nCHW, nKPQ)
      {
        const size_t tid = omp_get_thread_num();
        const size_t workspace_unpack_size = CRS * PQ;
        const size_t workspace_update_size = K * CRS;
        const size_t workspace_unpack_offset = tid * (workspace_unpack_size + workspace_update_size);
        const size_t workspace_update_offset = workspace_unpack_offset + workspace_unpack_size;
        #pragma omp for
        for (size_t n = 0; n < NIN; ++n) {
          nCHW = n * CHW;
          nKPQ = n * KPQ;
          BLITZ_DATA_LAYOUT unpack_data_layout = Unpack2DFunc(input->Slice(nCHW),
            workspace->Slice(workspace_unpack_offset),
            C, H, W,
            R, S,
            P, Q,
            padding_height, padding_width,
            stride_height, stride_width,
            input->data_layout());
          Convolution2DUpdateGEMMDispatch(
            workspace->Slice(workspace_unpack_offset),
            const_cast<CPUTensor<DType>*>(output)->Slice(nKPQ),
            workspace->Slice(workspace_update_offset),
            K, CRS, PQ,
            unpack_data_layout,
            output->data_layout(),
            update->data_layout());
        }
        for (size_t i = 0; i < update->size(); ++i) {
          #pragma omp atomic
          (*update)[i] += *(workspace->Slice(workspace_update_offset + i));
        }
      }
      break;
    }
    case BLITZ_CONVOLUTION_BLAS_GEMM: {
      for (size_t n = 0; n < NIN; ++n) {
        nCHW = n * CHW;
        nKPQ = n * KPQ;
        BLITZ_DATA_LAYOUT unpack_data_layout = Unpack2DFunc(input->Slice(nCHW),
          workspace->data(),
          C, H, W,
          R, S,
          P, Q,
          padding_height, padding_width,
          stride_height, stride_width,
          input->data_layout());
        Convolution2DUpdateGEMMDispatch(
          workspace->data(),
          const_cast<CPUTensor<DType>*>(output)->Slice(nKPQ),
          update->data(),
          K, CRS, PQ,
          unpack_data_layout,
          output->data_layout(),
          update->data_layout());
      }
      break;
    }
    default:
      LOG(FATAL) << "Unsupported algorithm type: " << algorithm;
      break;
  }
  #ifdef BLITZ_PERFORMANCE
  double computations = static_cast<double>(KPQ) * static_cast<double>(CRS) * static_cast<double>(2 * NIN);
  BLITZ_CPU_TIMER_END(elapsed_time, start, end);
  BLITZ_CPU_TIMER_INFO(computations, elapsed_time);
  #endif  // BLITZ_PERFORMANCE
}

#endif  // SRC_BACKENDS_CPU_BACKEND_CONV_INL_H_
