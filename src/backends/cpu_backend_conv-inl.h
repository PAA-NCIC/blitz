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
  time_point<system_clock> start, end;
  duration<float> unpack_time[BLITZ_NUM_THREADS];
  duration<float> gemm_time[BLITZ_NUM_THREADS];
  for (size_t i = 0; i < BLITZ_NUM_THREADS; ++i) {
    unpack_time[i] = duration<float>::zero();
    gemm_time[i] = duration<float>::zero();
  }
  float total_unpack_time = 0.0;
  float total_gemm_time = 0.0;
  #endif  // BLITZ_PERFORMANCE
  switch (algorithm) { // NCHW & NHWC
    case BLITZ_CONVOLUTION_BLAS_GEMM_BATCH: {
      #pragma omp parallel private(nCHW, nKPQ)
      {
        const size_t tid = omp_get_thread_num();
        const size_t workspace_unpack_offset = tid * CRS * PQ;
        DType* workspace_unpack_slice = workspace->Slice(workspace_unpack_offset);
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
          BLITZ_DATA_LAYOUT unpack_data_layout = Unpack2DFunc(
            input->Slice(nCHW),
            workspace_unpack_slice,
            C, H, W,
            R, S,
            P, Q,
            padding_height, padding_width,
            stride_height, stride_width,
            input->data_layout());
          #ifdef BLITZ_PERFORMANCE
          end = system_clock::now();
          unpack_time[tid] += end -start;
          start = system_clock::now();
          #endif  // BLITZ_PERFORMANCE
          Convolution2DForwardGEMMDispatch(workspace_unpack_slice,
            output->Slice(nKPQ),
            const_cast<CPUTensor<DType>*>(filter)->data(),
            K, PQ, CRS,
            unpack_data_layout,
            output->data_layout(),
            filter->data_layout());
          #ifdef BLITZ_PERFORMANCE
          end = system_clock::now();
          gemm_time[tid] += end - start;
          #endif  // BLITZ_PERFORMANCE
        }
      }
      #ifdef BLITZ_PERFORMANCE
      for (size_t i = 0; i < BLITZ_NUM_THREADS; ++i) {
        total_unpack_time += unpack_time[i].count();
        total_gemm_time += gemm_time[i].count();
      }
      total_unpack_time /= BLITZ_NUM_THREADS;
      total_gemm_time /= BLITZ_NUM_THREADS;
      #endif
      break;
    }
    case BLITZ_CONVOLUTION_BLAS_GEMM: {
      for (size_t n = 0; n < NIN; ++n) {
        nCHW = n * CHW;
        nKPQ = n * KPQ;
        #ifdef BLITZ_PERFORMANCE
        start = system_clock::now();
        #endif
        BLITZ_DATA_LAYOUT unpack_data_layout = Unpack2DFunc(
	  input->Slice(nCHW),
          workspace->data(),
          C, H, W,
          R, S,
          P, Q,
          padding_height, padding_width,
          stride_height, stride_width,
          input->data_layout());
        #ifdef BLITZ_PERFORMANCE
        end = system_clock::now();
        unpack_time[0] += end - start;
        start = system_clock::now();
        #endif
        Convolution2DForwardGEMMDispatch(workspace->data(),
          output->Slice(nKPQ),
          const_cast<CPUTensor<DType>*>(filter)->data(),
          K, PQ, CRS,
          unpack_data_layout,
          output->data_layout(),
          filter->data_layout());
        #ifdef BLITZ_PERFORMANCE
        end = system_clock::now();
        gemm_time[0] += end - start;
        total_unpack_time = unpack_time[0].count();
        total_gemm_time = gemm_time[0].count();
        #endif
      }
      break;
    }
    default:
      LOG(FATAL) << "Unsupported algorithm type: " << algorithm;
      break;
  }
  #ifdef BLITZ_PERFORMANCE
  double computations = static_cast<double>(KPQ) * static_cast<double>(CRS) * static_cast<double>(2 * NIN);
  LOG(INFO) << "Forward convolution compute: " << total_gemm_time;
  LOG(INFO) << "Forward convolution transform: " << total_unpack_time;
  LOG(INFO) << "Forward convolution compute gflops: " << computations / (total_gemm_time * 1e9);
  LOG(INFO) << "Forward convolution total gflops: " << computations / ((total_gemm_time + total_unpack_time) * 1e9);
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
  time_point<system_clock> start, end;
  duration<float> pack_time[BLITZ_NUM_THREADS];
  duration<float> gemm_time[BLITZ_NUM_THREADS];
  for (size_t i = 0; i < BLITZ_NUM_THREADS; ++i) {
    pack_time[i] = duration<float>::zero();
    gemm_time[i] = duration<float>::zero();
  }
  float total_pack_time = 0.0;
  float total_gemm_time = 0.0;
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
          // gemm generate
          // (input_channel * filter_height * filter_width)
          // (output_width * output_height) *
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
          // pack
          // (input_channel * filter_height * filter_width) *
          // (output_width * output_height)
          // to
          // (input_channel) *
          // (input_height * input_width)
          Pack2DFunc(workspace->Slice(workspace_unpack_offset),
            input->Slice(nCHW),
            C, H, W,
            R, S,
            P, Q,
            padding_height, padding_width,
            stride_height, stride_width);
          #ifdef BLITZ_PERFORMANCE
          end = system_clock::now();
          pack_time[tid] += end - start;
          #endif  // BLITZ_PERFORMANCE
        }
        #ifdef BLITZ_PERFORMANCE
        for (size_t i = 0; i < BLITZ_NUM_THREADS; ++i) {
          total_pack_time += pack_time[i].count();
          total_gemm_time += gemm_time[i].count();
        }
        total_pack_time /= BLITZ_NUM_THREADS;
        total_gemm_time /= BLITZ_NUM_THREADS;
        #endif
      }
      break;
    }
    case BLITZ_CONVOLUTION_BLAS_GEMM: {
      for (size_t n = 0; n < NIN; ++n) {
        nCHW = n * CHW;
        nKPQ = n * KPQ;
        #ifdef BLITZ_PERFORMANCE
        start = system_clock::now();
        #endif
        // gemm generate
        // (input_channel * filter_height * filter_width)
        // (output_width * output_height) *
        BlitzCPUGemm(const_cast<CPUTensor<DType>*>(filter)->data(),
          const_cast<CPUTensor<DType>*>(output)->Slice(nKPQ),
          workspace->data(),
          true, false,
          static_cast<DType>(1), static_cast<DType>(0),
          CRS, PQ, K);
        #ifdef BLITZ_PERFORMANCE
        end = system_clock::now();
        gemm_time[0] += end - start;
        start = system_clock::now();
        #endif
        // pack
        // (input_channel * filter_height * filter_width)
        // (output_width * output_height)
        // to
        // (input_channel) *
        // (input_height * input_width)
        Pack2DFunc(workspace->data(),
          input->Slice(nCHW),
          C, H, W,
          R, S,
          P, Q,
          padding_height, padding_width,
          stride_height, stride_width);
        #ifdef BLITZ_PERFORMANCE
        end = system_clock::now();
        pack_time[0] += end - start;
        total_pack_time = pack_time[0].count();
        total_gemm_time = gemm_time[0].count();
        #endif
      }
      break;
    }
    default:
      LOG(FATAL) << "Unsupported algorithm type: " << algorithm;
      break;
  }
  #ifdef BLITZ_PERFORMANCE
  double computations = static_cast<double>(KPQ) * static_cast<double>(CRS) * static_cast<double>(2 * NIN);
  LOG(INFO) << "Backward convolution compute: " << total_gemm_time;
  LOG(INFO) << "Backward convolution transform: " << total_pack_time;
  LOG(INFO) << "Backward convolution compute gflops: " << computations / (total_gemm_time * 1e9);
  LOG(INFO) << "Backward convolution total gflops: " << computations / ((total_pack_time + total_gemm_time) * 1e9);
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
  #ifdef BLITZ_PERFORMANCE
  time_point<system_clock> start, end;
  duration<float> unpack_time[BLITZ_NUM_THREADS];
  duration<float> gemm_time[BLITZ_NUM_THREADS];
  for (size_t i = 0; i < BLITZ_NUM_THREADS; ++i) {
    unpack_time[i] = duration<float>::zero();
    gemm_time[i] = duration<float>::zero();
  }
  float total_unpack_time = 0.0;
  float total_gemm_time = 0.0;
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
          // unpack
          // (input_channel) *
          // (input_width * input_height)
          // to
          // (input_channel * filter_height * filter_width)
          // (output_width * output_height)
          BLITZ_DATA_LAYOUT unpack_data_layout = Unpack2DFunc(input->Slice(nCHW),
            workspace->Slice(workspace_unpack_offset),
            C, H, W,
            R, S,
            P, Q,
            padding_height, padding_width,
            stride_height, stride_width,
            input->data_layout());
          #ifdef BLITZ_PERFORMANCE
          end = system_clock::now();
          unpack_time[tid] += end - start;
          start = system_clock::now();
          #endif  // BLITZ_PERFORMANCE
          Convolution2DUpdateGEMMDispatch(
            workspace->Slice(workspace_unpack_offset),
            const_cast<CPUTensor<DType>*>(output)->Slice(nKPQ),
            workspace->Slice(workspace_update_offset),
            K, CRS, PQ,
            unpack_data_layout,
            output->data_layout(),
            update->data_layout());
          #ifdef BLITZ_PERFORMANCE
          end = system_clock::now();
          gemm_time[tid] += end - start;
          #endif  // BLITZ_PERFORMANCE
        }
        for (size_t i = 0; i < update->size(); ++i) {
          #pragma omp atomic
          (*update)[i] += *(workspace->Slice(workspace_update_offset + i));
        }
        #ifdef BLITZ_PERFORMANCE
        for (size_t i = 0; i < BLITZ_NUM_THREADS; ++i) {
          total_unpack_time += unpack_time[i].count();
          total_gemm_time += gemm_time[i].count();
        }
        total_unpack_time /= BLITZ_NUM_THREADS;
        total_gemm_time /= BLITZ_NUM_THREADS;
        #endif
      }
      break;
    }
    case BLITZ_CONVOLUTION_BLAS_GEMM: {
      for (size_t n = 0; n < NIN; ++n) {
        nCHW = n * CHW;
        nKPQ = n * KPQ;
        #ifdef BLITZ_PERFORMANCE
        start = system_clock::now();
        #endif
        // unpack
        // (input_channel) *
        // (input_width * input_height)
        // to
        // (input_channel * filter_height * filter_width)
        // (output_width * output_height)
        BLITZ_DATA_LAYOUT unpack_data_layout = Unpack2DFunc(input->Slice(nCHW),
          workspace->data(),
          C, H, W,
          R, S,
          P, Q,
          padding_height, padding_width,
          stride_height, stride_width,
          input->data_layout());
        #ifdef BLITZ_PERFORMANCE
        end = system_clock::now();
        unpack_time[0] += end - start;
        start = system_clock::now();
        #endif
        Convolution2DUpdateGEMMDispatch(
          workspace->data(),
          const_cast<CPUTensor<DType>*>(output)->Slice(nKPQ),
          update->data(),
          K, CRS, PQ,
          unpack_data_layout,
          output->data_layout(),
          update->data_layout());
        #ifdef BLITZ_PERFORMANCE
        end = system_clock::now();
        gemm_time[0] += end - start;
        total_gemm_time = gemm_time[0].count();
        total_unpack_time = unpack_time[0].count();
        #endif
      }
      break;
    }
    default:
      LOG(FATAL) << "Unsupported algorithm type: " << algorithm;
      break;
  }
  #ifdef BLITZ_PERFORMANCE
  double computations = static_cast<double>(KPQ) * static_cast<double>(CRS) * static_cast<double>(2 * NIN);
  LOG(INFO) << "Backward convolution update compute: " << total_gemm_time;
  LOG(INFO) << "Backward convolution update transform: " << total_unpack_time;
  LOG(INFO) << "Backward convolution update compute gflops: " << computations / (total_gemm_time * 1e9);
  LOG(INFO) << "Backward convolution update total gflops: " << computations / ((total_unpack_time + total_gemm_time) * 1e9);
  #endif  // BLITZ_PERFORMANCE
}

#endif  // SRC_BACKENDS_CPU_BACKEND_CONV_INL_H_
