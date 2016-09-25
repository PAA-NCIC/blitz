#ifndef SRC_BACKEND_CPU_BACKEND_CONV_INL_H_
#define SRC_BACKEND_CPU_BACKEND_CONV_INL_H_

template<typename DType>
void Backend<CPUTensor, DType>::Convolution2DForwardFunc(
  const CPUTensor<DType>* input, const CPUTensor<DType>* filter,
  size_t padding_height, size_t padding_width,
  size_t stride_height, size_t stride_width,
  CPUTensor<DType>* workspace,
  CPUTensor<DType>* output,
  const string& kernel) {
  // shape decode
  // input
  const Shape& input_shape = input->shape();
  size_t batch_size = input_shape[0];
  size_t input_channel = input_shape[1];
  size_t input_height = input_shape[2];
  size_t input_width = input_shape[3];
  // filter
  const Shape& filter_shape = filter->shape();
  size_t filter_height = filter_shape[2];
  size_t filter_width = filter_shape[3];
  // output
  const Shape& output_shape = output->shape();
  size_t output_channel = output_shape[1];
  size_t output_height = output_shape[2];
  size_t output_width = output_shape[3];
  // offset
  size_t input_batch_offset = 0;
  size_t output_batch_offset = 0;
  const size_t input_batch_size = input_channel * input_height * input_width;
  const size_t output_batch_size = output_channel * output_height * output_width;
  // dims
  size_t dim_left = output_channel;
  size_t dim_right = output_height * output_width;
  size_t dim_common = input_channel * filter_height * filter_width;
  // time counter
  #ifdef BLITZ_PERFORMANCE
  time_point<system_clock> start, end;
  duration<double> gemm_time = duration<double>::zero();
  duration<double> unpack_time = duration<double>::zero();
  double total_gemm_time = 0.0;
  double total_unpack_time = 0.0;
  #endif  // BLITZ_PERFORMANCE
  if (kernel == "blas_batch") {
    #pragma omp parallel private(input_batch_offset, output_batch_offset)
    {
      const size_t tid = omp_get_thread_num();
      const size_t workspace_unpack_offset = tid *
        input_channel * filter_height * filter_width *
        output_width * output_height;
      #ifdef BLITZ_PERFORMANCE
        #pragma omp for private(start, end)
      #else
        #pragma omp for
      #endif
      for (size_t batch_index = 0; batch_index < batch_size; ++batch_index) {
        input_batch_offset = batch_index * input_batch_size;
        output_batch_offset = batch_index *  output_batch_size;
        #ifdef BLITZ_PERFORMANCE
        start = system_clock::now();
        #endif  // BLITZ_PERFORMANCE
        // unpack
        // (input_channel) *
        // (input_width * input_height)
        // to
        // (input_channel * filter_height * filter_width)
        // (output_width * output_height)
        Unpack2DFunc(input->Slice(input_batch_offset),
          input_channel, input_height, input_width,
          filter_height, filter_width,
          output_height, output_width,
          padding_height, padding_width,
          stride_height, stride_width,
          workspace->Slice(workspace_unpack_offset));
        #ifdef BLITZ_PERFORMANCE
        end = system_clock::now();
        #pragma omp critical
        unpack_time += end - start;
        start = system_clock::now();
        #endif  // BLITZ_PERFORMANCE
        // gemm generate
        // (output_channel) * (output_height * output_width)
        BlitzCPUGemm(false, false,
        dim_left, dim_right, dim_common,
        const_cast<CPUTensor<DType>*>(filter)->data(),
        workspace->Slice(workspace_unpack_offset),
        output->Slice(output_batch_offset),
        static_cast<DType>(1), static_cast<DType>(0));
        #ifdef BLITZ_PERFORMANCE
        end = system_clock::now();
        #pragma omp critical
        gemm_time += end - start;
        #endif  // BLITZ_PERFORMANCE
      }
      #ifdef BLITZ_PERFORMANCE
      if (tid == 0) {
        total_unpack_time = unpack_time.count() /
          omp_get_num_threads();
        total_gemm_time = gemm_time.count() /
          omp_get_num_threads();
      }
      #endif
    }
  } else if (kernel == "blas") {  // default blas
    for (size_t batch_index = 0; batch_index < batch_size; ++batch_index) {
      input_batch_offset = batch_index * input_batch_size;
      output_batch_offset = batch_index *  output_batch_size;
      #ifdef BLITZ_PERFORMANCE
      start = system_clock::now();
      #endif
      // unpack
      // (input_channel) *
      // (input_width * input_height)
      // to
      // (input_channel * filter_height * filter_width)
      // (output_width * output_height)
      Unpack2DFunc(input->Slice(input_batch_offset),
        input_channel, input_height, input_width,
        filter_height, filter_width,
        output_height, output_width,
        padding_height, padding_width,
        stride_height, stride_width,
        workspace->data());
      #ifdef BLITZ_PERFORMANCE
      end = system_clock::now();
      unpack_time += end - start;
      start = system_clock::now();
      #endif
      // gemm generate
      // (output_channel) * (output_height * output_width)
      BlitzCPUGemm(false, false,
        dim_left, dim_right, dim_common,
        const_cast<CPUTensor<DType>*>(filter)->data(),
        workspace->data(),
        output->Slice(output_batch_offset),
        static_cast<DType>(1), static_cast<DType>(0));
      #ifdef BLITZ_PERFORMANCE
      end = system_clock::now();
      gemm_time += end - start;
      total_unpack_time = unpack_time.count();
      total_gemm_time = gemm_time.count();
      #endif
    }
  }
  #ifdef BLITZ_PERFORMANCE
  LOG(INFO) << "Forward convolution gemm: " << total_gemm_time;
  LOG(INFO) << "Forward convolution unpack: " << total_unpack_time;
  #endif  // BLITZ_PERFORMANCE
}

template<typename DType>
void Backend<CPUTensor, DType>::Convolution2DBackwardFunc(
  const CPUTensor<DType>* output, const CPUTensor<DType>* filter,
  size_t padding_height, size_t padding_width,
  size_t stride_height, size_t stride_width,
  CPUTensor<DType>* workspace,
  CPUTensor<DType>* input,
  const string& kernel) {
  // shape decode
  // input
  const Shape& input_shape = input->shape();
  size_t batch_size = input_shape[0];
  size_t input_channel = input_shape[1];
  size_t input_height = input_shape[2];
  size_t input_width = input_shape[3];
  // filter
  const Shape& filter_shape = filter->shape();
  size_t filter_height = filter_shape[2];
  size_t filter_width = filter_shape[3];
  // output
  const Shape& output_shape = output->shape();
  size_t output_channel = output_shape[1];
  size_t output_height = output_shape[2];
  size_t output_width = output_shape[3];
  // offset
  size_t input_batch_offset = 0;
  size_t output_batch_offset = 0;
  const size_t input_batch_size = input_channel * input_height * input_width;
  const size_t output_batch_size = output_channel * output_height * output_width;
  // dims
  size_t dim_left = input_channel * filter_height * filter_width;
  size_t dim_right = output_height * output_width;
  size_t dim_common = output_channel;
  input->Fill(0);
  #ifdef BLITZ_PERFORMANCE
  time_point<system_clock> start, end;
  duration<double> gemm_time = duration<double>::zero();
  duration<double> pack_time = duration<double>::zero();
  double total_gemm_time = 0.0;
  double total_pack_time = 0.0;
  #endif  // BLITZ_PERFORMANCE
  if (kernel == "blas_batch") {
    #pragma omp parallel private(input_batch_offset, output_batch_offset) 
    {
      const size_t tid = omp_get_thread_num();
      const size_t workspace_unpack_offset = tid *
        input_channel * filter_height * filter_width *
        output_width * output_height;
      #ifdef BLITZ_PERFORMANCE
        #pragma omp for private(start, end)
      #else
        #pragma omp for
      #endif
      for (size_t batch_index = 0; batch_index < batch_size; ++batch_index) {
        input_batch_offset = batch_index * input_batch_size;
        output_batch_offset = batch_index * output_batch_size;
        #ifdef BLITZ_PERFORMANCE
        start = system_clock::now();
        #endif  // BLITZ_PERFORMANCE
        // gemm generate
        // (output_width * output_height) *
        // (input_channel * filter_height * filter_width)
        BlitzCPUGemm(true, false, dim_left, dim_right, dim_common,
        const_cast<CPUTensor<DType>*>(filter)->data(),
        const_cast<CPUTensor<DType>*>(output)->Slice(output_batch_offset),
        workspace->Slice(workspace_unpack_offset),
        static_cast<DType>(1), static_cast<DType>(0));
        #ifdef BLITZ_PERFORMANCE
        end = system_clock::now();
        #pragma omp critical
        gemm_time += end - start;
        start = system_clock::now();
        #endif  // BLITZ_PERFORMANCE
        // pack
        // (input_channel * filter_height * filter_width) *
        // (output_width * output_height)
        // to
        // (input_channel) *
        // (input_height * input_width)
        Pack2DFunc(workspace->Slice(workspace_unpack_offset),
          input_channel, input_height, input_width,
          filter_height, filter_width,
          output_height, output_width,
          padding_height, padding_width,
          stride_height, stride_width,
          input->Slice(input_batch_offset));
        #ifdef BLITZ_PERFORMANCE
        end = system_clock::now();
        #pragma omp critical
        pack_time += end - start;
        #endif  // BLITZ_PERFORMANCE
      }
      #ifdef BLITZ_PERFORMANCE
      if (tid == 0) {
        total_pack_time = pack_time.count() /
          omp_get_num_threads();
        total_gemm_time = gemm_time.count() /
          omp_get_num_threads();
      }
      #endif
    }
  } else if (kernel == "blas") {
    for (size_t batch_index = 0; batch_index < batch_size; ++batch_index) {
      input_batch_offset = batch_index * input_batch_size;
      output_batch_offset = batch_index * output_batch_size;
      #ifdef BLITZ_PERFORMANCE
      start = system_clock::now();
      #endif
      // gemm generate
      // (output_width * output_height) *
      // (input_channel * filter_height * filter_width)
      BlitzCPUGemm(true, false, dim_left, dim_right, dim_common,
      const_cast<CPUTensor<DType>*>(filter)->data(),
      const_cast<CPUTensor<DType>*>(output)->Slice(output_batch_offset),
      workspace->data(),
      static_cast<DType>(1), static_cast<DType>(0));
      #ifdef BLITZ_PERFORMANCE
      end = system_clock::now();
      gemm_time += end - start;
      start = system_clock::now();
      #endif
      // pack
      // (input_channel * filter_height * filter_width)
      // (output_width * output_height)
      // to
      // (input_channel) *
      // (input_height * input_width)
      Pack2DFunc(workspace->data(),
        input_channel, input_height, input_width,
        filter_height, filter_width,
        output_height, output_width,
        padding_height, padding_width,
        stride_height, stride_width,
        input->Slice(input_batch_offset));
      #ifdef BLITZ_PERFORMANCE
      end = system_clock::now();
      pack_time += end - start;
      total_pack_time = pack_time.count();
      total_gemm_time = gemm_time.count();
      #endif
    }
  }
  #ifdef BLITZ_PERFORMANCE
  LOG(INFO) << "Backward convolution gemm: " << total_gemm_time;
  LOG(INFO) << "Backward convolution pack: " << total_pack_time;
  #endif  // BLITZ_PERFORMANCE
}

template<typename DType>
void Backend<CPUTensor, DType>::Convolution2DUpdateFunc(
  const CPUTensor<DType>* input, const CPUTensor<DType>* output,
  size_t padding_height, size_t padding_width,
  size_t stride_height, size_t stride_width,
  CPUTensor<DType>* workspace,
  CPUTensor<DType>* update,
  const string& kernel) {
  // extract shapes
  // input
  const Shape& input_shape = input->shape();
  size_t batch_size = input_shape[0];
  size_t input_channel = input_shape[1];
  size_t input_height = input_shape[2];
  size_t input_width = input_shape[3];
  // filter
  const Shape& filter_shape = update->shape();
  size_t filter_height = filter_shape[2];
  size_t filter_width = filter_shape[3];
  // output
  const Shape& output_shape = output->shape();
  size_t output_channel = output_shape[1];
  size_t output_height = output_shape[2];
  size_t output_width = output_shape[3];
  // offset
  size_t input_batch_offset = 0;
  size_t output_batch_offset = 0;
  const size_t input_batch_size = input_channel * input_height * input_width;
  const size_t output_batch_size = output_channel * output_height * output_width;
  // dims
  size_t dim_left = output_channel;
  size_t dim_right = input_channel * filter_height * filter_width;
  size_t dim_common = output_height * output_width;
  #ifdef BLITZ_PERFORMANCE
  time_point<system_clock> start, end;
  duration<double> gemm_time = duration<double>::zero();
  duration<double> unpack_time = duration<double>::zero();
  double total_gemm_time = 0;
  double total_unpack_time = 0;
  #endif  // BLITZ_PERFORMANCE
  if (kernel == "blas_batch") {
    #pragma omp parallel private(input_batch_offset, output_batch_offset)
    {
      const size_t tid = omp_get_thread_num();
      const size_t workspace_unpack_size = input_channel *
        filter_height * filter_width * output_height * output_width;
      const size_t workspace_update_size = output_channel *
        input_channel * filter_height * filter_width;
      const size_t workspace_unpack_offset = tid *
        (workspace_unpack_size + workspace_update_size);
      const size_t workspace_update_offset = workspace_unpack_offset +
        workspace_update_size;
      #ifdef BLITZ_PERFORMANCE
        #pragma omp for private(start, end)
      #else
        #pragma omp for
      #endif
      for (size_t batch_index = 0; batch_index < batch_size; ++batch_index) {
        input_batch_offset = batch_index * input_batch_size;
        output_batch_offset = batch_index * output_batch_size;
        #ifdef BLITZ_PERFORMANCE
        start = system_clock::now();
        #endif  // BLITZ_PERFORMANCE
        // unpack
        // (input_channel) *
        // (input_width * input_height)
        // to
        // (input_channel * filter_height * filter_width)
        // (output_width * output_height)
        Unpack2DFunc(input->Slice(input_batch_offset),
          input_channel, input_height, input_width,
          filter_height, filter_width,
          output_height, output_width,
          padding_height, padding_width,
          stride_height, stride_width,
          workspace->Slice(workspace_unpack_offset));
        #ifdef BLITZ_PERFORMANCE
        end = system_clock::now();
        #pragma omp critical
        unpack_time += end - start;
        start = system_clock::now();
        #endif  // BLITZ_PERFORMANCE
        // gemm generate
        // (output_channel) *
        // (input_channel * filter_height * filter_width)
        BlitzCPUGemm(false, true, dim_left, dim_right, dim_common,
          const_cast<CPUTensor<DType>*>(output)->Slice(output_batch_offset),
          workspace->Slice(workspace_unpack_offset),
          workspace->Slice(workspace_update_offset),
          static_cast<DType>(1), static_cast<DType>(1));
        #ifdef BLITZ_PERFORMANCE
        end = system_clock::now();
        #pragma omp critical
        gemm_time += end - start;
        #endif  // BLITZ_PERFORMANCE
      }
      for (size_t i = 0; i < update->size(); ++i) {
        #pragma omp atomic
        (*update)[i] += *(workspace->Slice(workspace_update_offset + i));
      }
      #ifdef BLITZ_PERFORMANCE
      if (tid == 0) {
        total_unpack_time = unpack_time.count() /
          omp_get_num_threads();
        total_gemm_time = gemm_time.count() /
          omp_get_num_threads();
      }
      #endif
    }
  } else if (kernel == "blas") {
    for (size_t batch_index = 0; batch_index < batch_size; ++batch_index) {
      input_batch_offset = batch_index * input_batch_size;
      output_batch_offset = batch_index * output_batch_size;
      #ifdef BLITZ_PERFORMANCE
      start = system_clock::now();
      #endif
      // unpack
      // (input_channel) *
      // (input_width * input_height)
      // to
      // (input_channel * filter_height * filter_width)
      // (output_width * output_height)
      Unpack2DFunc(input->Slice(input_batch_offset),
        input_channel, input_height, input_width,
        filter_height, filter_width,
        output_height, output_width,
        padding_height, padding_width,
        stride_height, stride_width,
        workspace->data());
      #ifdef BLITZ_PERFORMANCE
      end = system_clock::now();
      unpack_time += end - start;
      start = system_clock::now();
      #endif
      // gemm generate
      // (output_channel) *
      // (input_channel * filter_height * filter_width)
      BlitzCPUGemm(false, true,
        dim_left, dim_right, dim_common,
        const_cast<CPUTensor<DType>*>(output)->Slice(output_batch_offset),
        workspace->data(), update->data(),
        static_cast<DType>(1), static_cast<DType>(1));
      #ifdef BLITZ_PERFORMANCE
      end = system_clock::now();
      gemm_time += end - start;
      total_gemm_time = gemm_time.count();
      total_unpack_time = unpack_time.count();
      #endif
    }
  }
  #ifdef BLITZ_PERFORMANCE
  LOG(INFO) << "Backward convolution filter gemm: " << total_gemm_time;
  LOG(INFO) << "Backward convolution filter unpack: " << total_unpack_time;
  #endif  // BLITZ_PERFORMANCE
}

#endif  // SRC_BACKEND_CPU_BACKEND_CONV_INL_H_

