#ifndef SRC_BACKEND_GPU_BACKEND_CONV_INL_H_
#define SRC_BACKEND_GPU_BACKEND_CONV_INL_H_

template<typename DType>
void Backend<GPUTensor, DType>::Convolution2DForwardFunc(
  const GPUTensor<DType>* input, const GPUTensor<DType>* filter,
  size_t padding_height, size_t padding_width,
  size_t stride_height, size_t stride_width,
  GPUTensor<DType>* unpack, GPUTensor<DType>* output,
  const string& kernel) {
  // shape decode
  // input
  const Shape& input_shape = input->shape();
  int batch_size = input_shape[0];
  int input_channel = input_shape[1];
  int input_height = input_shape[2];
  int input_width = input_shape[3];
  // filter
  const Shape& filter_shape = filter->shape();
  int filter_height = filter_shape[2];
  int filter_width = filter_shape[3];
  // output
  const Shape& output_shape = output->shape();
  int output_channel = output_shape[1];
  int output_height = output_shape[2];
  int output_width = output_shape[3];

  int batch_input_offset = 0;
  int batch_output_offset = 0;
  int dim_left = output_channel;
  int dim_right = output_height * output_width;
  int dim_common = input_channel * filter_height * filter_width;
#ifdef BLITZ_DEVELOP
  LOG(INFO) << "dim left: " << dim_left;
  LOG(INFO) << "dim right: " << dim_right;
  LOG(INFO) << "dim common: " << dim_common;
#endif
#ifdef BLITZ_PERFORMANCE  // only valid for a single thread
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  float gemm_time = 0;
  float elapsed_time = 0;
  float unpack_time = 0;
#endif  // BLITZ_PERFORMANCE

  if (kernel == "asm_direct") {
    BlitzSass2DConvolution(batch_size, input_channel, input_height,
      input_width, filter_height, filter_width, output_channel,
      output_height, output_width, stride_height, stride_width,
      const_cast<DType*>(input->data()), output->data(),
      const_cast<DType*>(filter->data()), "forward");
  } else {
    for (int batch_index = 0; batch_index < batch_size; ++batch_index) {
#ifdef BLITZ_PERFORMANCE
      cudaEventRecord(start);
#endif
      // unpack
      // (input_channel) *
      // (input_width * input_height)
      // to
      // (output_width * output_height)
      // (input_channel * filter_height * filter_width)
      Unpack2DParallelFunc(input->Slice(batch_input_offset),
          input_channel, input_height, input_width,
          filter_height, filter_width, output_height, output_width,
          padding_height, padding_width,
          stride_height, stride_width, unpack->data());
#ifdef BLITZ_PERFORMANCE
      cudaEventRecord(stop);
      cudaEventSynchronize(stop);
      cudaEventElapsedTime(&elapsed_time, start, stop);
      elapsed_time /= 1000.0;
      unpack_time += elapsed_time;
#endif

#ifdef BLITZ_PERFORMANCE
      cudaEventRecord(start);
#endif
      // gemm generate
      // (output_channel) * (output_height * output_width)
      if (kernel == "blas") {
        BlitzGPUGemm(false, true, dim_left, dim_right, dim_common,
            const_cast<GPUTensor<DType>*>(filter)->data(),
            unpack->data(), output->Slice(batch_output_offset),
            static_cast<DType>(1), static_cast<DType>(0));
      } else if (kernel == "asm") {
        BlitzSassGemm(false, true, dim_left, dim_right, dim_common,
            const_cast<GPUTensor<DType>*>(filter)->data(),
            unpack->data(), output->Slice(batch_output_offset),
            static_cast<DType>(1), static_cast<DType>(0));
      }
#ifdef BLITZ_PERFORMANCE
      cudaEventRecord(stop);
      cudaEventSynchronize(stop);
      cudaEventElapsedTime(&elapsed_time, start, stop);
      elapsed_time /= 1000.0;
      gemm_time += elapsed_time;
#endif

      batch_input_offset += input_channel * input_height * input_width;
      batch_output_offset += output_channel * output_height * output_width;
    }
  }

#ifdef BLITZ_PERFORMANCE
  LOG(INFO) << "Forward convolution gemm: " << gemm_time;
  LOG(INFO) << "Forward convolution unpack: " << unpack_time;
#endif  // BLITZ_PERFORMANCE
}

template<typename DType>
void Backend<GPUTensor, DType>::Convolution2DBackwardFunc(
  const GPUTensor<DType>* output, const GPUTensor<DType>* filter,
  size_t padding_height, size_t padding_width,
  size_t stride_height, size_t stride_width,
  GPUTensor<DType>* pack, GPUTensor<DType>* input,
  const string& kernel) {
  // shape decode
  // input
  const Shape& input_shape = input->shape();
  int batch_size = input_shape[0];
  int input_channel = input_shape[1];
  int input_height = input_shape[2];
  int input_width = input_shape[3];
  // filter
  const Shape& filter_shape = filter->shape();
  int filter_height = filter_shape[2];
  int filter_width = filter_shape[3];
  // output
  const Shape& output_shape = output->shape();
  int output_channel = output_shape[1];
  int output_height = output_shape[2];
  int output_width = output_shape[3];

  int batch_input_offset = 0;
  int batch_output_offset = 0;
  int dim_left = output_height * output_width;
  int dim_right = input_channel * filter_height * filter_width;
  int dim_common = output_channel;
#ifdef BLITZ_DEVELOP
  LOG(INFO) << "dim left: " << dim_left;
  LOG(INFO) << "dim right: " << dim_right;
  LOG(INFO) << "dim common: " << dim_common;
#endif
  input->Fill(0);
  #ifdef BLITZ_PERFORMANCE
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  float gemm_time = 0;
  float elapsed_time = 0;
  float pack_time = 0;
  #endif  // BLITZ_PERFORMANCE

  if (kernel == "asm_direct") {
    BlitzSass2DConvolution(batch_size, input_channel, input_height,
      input_width, filter_height, filter_width, output_channel,
      output_height, output_width, stride_height, stride_width,
      input->data(), const_cast<DType*>(output->data()),
      const_cast<DType*>(filter->data()), "backward");
  } else {
    for (int batch_index = 0; batch_index < batch_size; ++batch_index) {
      #ifdef BLITZ_PERFORMANCE
      cudaEventRecord(start);
      #endif
      // gemm generate
      // (output_width * output_height) *
      // (input_channel * filter_height * filter_width)
      if (kernel == "blas") {
        BlitzGPUGemm(true, false, dim_left, dim_right, dim_common,
        const_cast<GPUTensor<DType>*>(output)->Slice(batch_output_offset),
        const_cast<GPUTensor<DType>*>(filter)->data(),
        pack->data(), static_cast<DType>(1), static_cast<DType>(0));
      } else if (kernel == "asm") {
        BlitzSassGemm(true, false, dim_left, dim_right, dim_common,
        const_cast<GPUTensor<DType>*>(output)->Slice(batch_output_offset),
        const_cast<GPUTensor<DType>*>(filter)->data(),
        pack->data(), static_cast<DType>(1), static_cast<DType>(0));
      }
      #ifdef BLITZ_PERFORMANCE
      cudaEventRecord(stop);
      cudaEventSynchronize(stop);
      cudaEventElapsedTime(&elapsed_time, start, stop);
      elapsed_time /= 1000.0;
      gemm_time += elapsed_time;
      #endif

      #ifdef BLITZ_PERFORMANCE
      cudaEventRecord(start);
      #endif
      // pack
      // (output_width * output_height)
      // (input_channel * filter_height * filter_width)
      // to
      // (input_channel) *
      // (input_height * input_width)
      Pack2DParallelFunc(pack->data(), input_channel, input_height, input_width,
        filter_height, filter_width, output_height, output_width,
        padding_height, padding_width, stride_height, stride_width,
        input->Slice(batch_input_offset));
      #ifdef BLITZ_PERFORMANCE
      cudaEventRecord(stop);
      cudaEventSynchronize(stop);
      cudaEventElapsedTime(&elapsed_time, start, stop);
      elapsed_time /= 1000.0;
      pack_time += elapsed_time;
      #endif
      batch_input_offset += input_channel * input_height * input_width;
      batch_output_offset += output_channel * output_height * output_width;
    }
  }

  #ifdef BLITZ_PERFORMANCE
  LOG(INFO) << "Backward convolution gemm: " << gemm_time;
  LOG(INFO) << "Backward convolution pack: " << pack_time;
  #endif  // BLITZ_PERFORMANCE
}

template<typename DType>
void Backend<GPUTensor, DType>::Convolution2DUpdateFunc(
  const GPUTensor<DType>* input, const GPUTensor<DType>* output,
  size_t padding_height, size_t padding_width,
  size_t stride_height, size_t stride_width,
  GPUTensor<DType>* unpack, GPUTensor<DType>* update,
  const string& kernel) {
  // shape decode
  // input
  const Shape& input_shape = input->shape();
  int batch_size = input_shape[0];
  int input_channel = input_shape[1];
  int input_height = input_shape[2];
  int input_width = input_shape[3];
  // filter
  const Shape& filter_shape = update->shape();
  int filter_height = filter_shape[2];
  int filter_width = filter_shape[3];
  // output
  const Shape& output_shape = output->shape();
  int output_channel = output_shape[1];
  int output_height = output_shape[2];
  int output_width = output_shape[3];

  int batch_input_offset = 0;
  int batch_output_offset = 0;
  int dim_left = output_channel;
  int dim_right = input_channel * filter_height * filter_width;
  int dim_common = output_height * output_width;
#ifdef BLITZ_DEVELOP
  LOG(INFO) << "dim left: " << dim_left;
  LOG(INFO) << "dim right: " << dim_right;
  LOG(INFO) << "dim common: " << dim_common;
#endif
  #ifdef BLITZ_PERFORMANCE  // only valid for a single thread
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  float gemm_time = 0;
  float elapsed_time = 0;
  float unpack_time = 0;
  #endif  // BLITZ_PERFORMANCE

  if (kernel == "asm_direct") {
    // Transpose to [IC * IH * IW] * [batch_size]
    BlitzGPUTrans(batch_size, input_channel * input_height * input_width,
      const_cast<DType*>(input->data()), unpack->data());
    BlitzSass2DConvolution(batch_size, input_channel, input_height,
      input_width, filter_height, filter_width, output_channel,
      output_height, output_width, stride_height, stride_width,
      const_cast<DType*>(unpack->data()), const_cast<DType*>(output->data()),
      update->data(), "update");
    BlitzGPUTrans(input_channel * input_height * input_width, batch_size,
      unpack->data(), const_cast<DType*>(input->data()));
    // Transpose to IC * IH * IW * batch_size
  } else {
    for (int batch_index = 0; batch_index < batch_size; ++batch_index) {
      #ifdef BLITZ_PERFORMANCE
      cudaEventRecord(start);
      #endif
      // unpack
      // (input_channel) *
      // (input_width * input_height)
      // to
      // (output_width * output_height)
      // (input_channel * filter_height * filter_width)
      Unpack2DParallelFunc(input->Slice(batch_input_offset),
        input_channel, input_height, input_width,
        filter_height, filter_width,
        output_height, output_width,
        padding_height, padding_width,
        stride_height, stride_width, unpack->data());
      #ifdef BLITZ_PERFORMANCE
      cudaEventRecord(stop);
      cudaEventSynchronize(stop);
      cudaEventElapsedTime(&elapsed_time, start, stop);
      elapsed_time /= 1000.0;
      unpack_time += elapsed_time;
      #endif

      #ifdef BLITZ_PERFORMANCE
      cudaEventRecord(start);
      #endif
      // gemm generate
      // (output_channel) *
      // (input_channel * filter_height * filter_width)
      if (kernel == "blas") {
        BlitzGPUGemm(false, false, dim_left, dim_right, dim_common,
          const_cast<GPUTensor<DType>*>(output)->Slice(batch_output_offset),
          unpack->data(), update->data(),
          static_cast<DType>(1), static_cast<DType>(1));
      } else if (kernel == "asm") {
        BlitzSassGemm(false, false, dim_left, dim_right, dim_common,
          const_cast<GPUTensor<DType>*>(output)->Slice(batch_output_offset),
          unpack->data(), update->data(),
          static_cast<DType>(1), static_cast<DType>(1));
      }
      batch_input_offset += input_channel * input_height * input_width;
      batch_output_offset += output_channel * output_height * output_width;
      #ifdef BLITZ_PERFORMANCE
      cudaEventRecord(stop);
      cudaEventSynchronize(stop);
      cudaEventElapsedTime(&elapsed_time, start, stop);
      elapsed_time /= 1000.0;
      gemm_time += elapsed_time;
      #endif
    }
  }

  #ifdef BLITZ_PERFORMANCE
  LOG(INFO) << "Backward convolution filter gemm: " << gemm_time;
  LOG(INFO) << "Backward convolution filter unpack: " << unpack_time;
  #endif  // BLITZ_PERFORMANCE
}

// batch parallel
template<typename DType>
void Backend<GPUTensor, DType>::Convolution2DForwardFunc(
  const GPUTensor<DType>* input, const GPUTensor<DType>* filter,
  size_t padding_height, size_t padding_width,
  size_t stride_height, size_t stride_width,
  vector<shared_ptr<GPUTensor<DType> > >* unpack_batch,
  GPUTensor<DType>* output) {}

template<typename DType>
void Backend<GPUTensor, DType>::Convolution2DBackwardFunc(
  const GPUTensor<DType>* output, const GPUTensor<DType>* filter,
  size_t padding_height, size_t padding_width,
  size_t stride_height, size_t stride_width,
  vector<shared_ptr<GPUTensor<DType> > >* pack_batch,
  GPUTensor<DType>* input) {}

template<typename DType>
void Backend<GPUTensor, DType>::Convolution2DUpdateFunc(
  const GPUTensor<DType>* input, const GPUTensor<DType>* output,
  size_t padding_height, size_t padding_width,
  size_t stride_height, size_t stride_width,
  vector<shared_ptr<GPUTensor<DType> > >* unpack_batch,
  vector<shared_ptr<GPUTensor<DType> > >* update_batch,
  GPUTensor<DType>* update) {}

// naive parallel
template<typename DType>
void Backend<GPUTensor, DType>::Convolution2DForwardFunc(
  const GPUTensor<DType>* input, const GPUTensor<DType>* filter,
  size_t stride_height, size_t stride_width,
  GPUTensor<DType>* output) {}

#endif  // SRC_BACKEND_GPU_BACKEND_CONV_INL_H_
