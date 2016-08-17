#ifndef SRC_BACKEND_CPU_BACKEND_CONV_INL_H_
#define SRC_BACKEND_CPU_BACKEND_CONV_INL_H_

#include <vector>

template<typename DType>
void Backend<CPUTensor, DType>::Convolution2DForwardFunc(
  const CPUTensor<DType>* input, const CPUTensor<DType>* filter,
  const int padding_height, const int padding_width,
  const int stride_height, const int stride_width,
  CPUTensor<DType>* unpack, CPUTensor<DType>* output,
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
#ifdef BLITZ_PERFORMANCE  // only valid for a single thread
  time_point<system_clock> start, end;
  duration<double> gemm_time =
    duration<double>::zero();
  duration<double> unpack_time =
    duration<double>::zero();
#endif  // BLITZ_PERFORMANCE

  for (int batch_index = 0; batch_index < batch_size; ++batch_index) {
#ifdef BLITZ_PERFORMANCE
    start = system_clock::now();
#endif
    // unpack
    // (input_channel) *
    // (input_width * input_height)
    // to
    // (input_channel * filter_height * filter_width)
    // (output_width * output_height)
    Unpack2DParallelFunc(input->Slice(batch_input_offset),
        input_channel, input_height, input_width,
        filter_height, filter_width, output_height, output_width,
        padding_height, padding_width,
        stride_height, stride_width, unpack->data());
#ifdef BLITZ_PERFORMANCE
    end = system_clock::now();
    unpack_time += end - start;
#endif

#ifdef BLITZ_PERFORMANCE
    start = system_clock::now();
#endif
    // gemm generate
    // (output_channel) * (output_height * output_width)
    BlitzCPUGemm(false, false, dim_left, dim_right, dim_common,
        const_cast<CPUTensor<DType>*>(filter)->data(),
        unpack->data(), output->Slice(batch_output_offset),
        static_cast<DType>(1), static_cast<DType>(0));
#ifdef BLITZ_PERFORMANCE
    end = system_clock::now();
    gemm_time += end - start;
#endif

    batch_input_offset += input_channel * input_height * input_width;
    batch_output_offset += output_channel * output_height * output_width;
  }

#ifdef BLITZ_PERFORMANCE
  LOG(INFO) << "Forward convolution gemm: " << gemm_time.count();
  LOG(INFO) << "Forward convolution unpack: " << unpack_time.count();
#endif  // BLITZ_PERFORMANCE
}

template<typename DType>
void Backend<CPUTensor, DType>::Convolution2DBackwardFunc(
  const CPUTensor<DType>* output, const CPUTensor<DType>* filter,
  const int padding_height, const int padding_width,
  const int stride_height, const int stride_width,
  CPUTensor<DType>* pack, CPUTensor<DType>* input,
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
  int dim_left = input_channel * filter_height * filter_width;
  int dim_right = output_height * output_width;
  int dim_common = output_channel;
  input->Fill(0);
  #ifdef BLITZ_PERFORMANCE  // only valid for a single thread
  time_point<system_clock> start, end;
  duration<double> gemm_time = duration<double>::zero();
  duration<double> pack_time = duration<double>::zero();
  #endif  // BLITZ_PERFORMANCE

  for (int batch_index = 0; batch_index < batch_size; ++batch_index) {
    #ifdef BLITZ_PERFORMANCE
    start = system_clock::now();
    #endif
    // gemm generate
    // (output_width * output_height) *
    // (input_channel * filter_height * filter_width)
    BlitzCPUGemm(true, false, dim_left, dim_right, dim_common,
    const_cast<CPUTensor<DType>*>(filter)->data(),
    const_cast<CPUTensor<DType>*>(output)->Slice(batch_output_offset),
    pack->data(), static_cast<DType>(1), static_cast<DType>(0));
    #ifdef BLITZ_PERFORMANCE
    end = system_clock::now();
    gemm_time += end - start;
    #endif

    #ifdef BLITZ_PERFORMANCE
    start = system_clock::now();
    #endif
    // pack
    // (input_channel * filter_height * filter_width)
    // (output_width * output_height)
    // to
    // (input_channel) *
    // (input_height * input_width)
    Pack2DParallelFunc(pack->data(), input_channel, input_height, input_width,
      filter_height, filter_width, output_height, output_width,
      padding_height, padding_width, stride_height, stride_width,
      input->Slice(batch_input_offset));
    #ifdef BLITZ_PERFORMANCE
    end = system_clock::now();
    pack_time += end - start;
    #endif
    batch_input_offset += input_channel * input_height * input_width;
    batch_output_offset += output_channel * output_height * output_width;
  }

  #ifdef BLITZ_PERFORMANCE
  LOG(INFO) << "Backward convolution gemm: " << gemm_time.count();
  LOG(INFO) << "Backward convolution pack: " << pack_time.count();
  #endif  // BLITZ_PERFORMANCE
}

template<typename DType>
void Backend<CPUTensor, DType>::Convolution2DUpdateFunc(
  const CPUTensor<DType>* input, const CPUTensor<DType>* output,
  const int padding_height, const int padding_width,
  const int stride_height, const int stride_width,
  CPUTensor<DType>* unpack, CPUTensor<DType>* update,
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
  #ifdef BLITZ_PERFORMANCE  // only valid for a single thread
  time_point<system_clock> start, end;
  duration<double> gemm_time = duration<double>::zero();
  duration<double> unpack_time = duration<double>::zero();
  #endif  // BLITZ_PERFORMANCE

  for (int batch_index = 0; batch_index < batch_size; ++batch_index) {
    #ifdef BLITZ_PERFORMANCE
    start = system_clock::now();
    #endif
    // unpack
    // (input_channel) *
    // (input_width * input_height)
    // to
    // (input_channel * filter_height * filter_width)
    // (output_width * output_height)
    Unpack2DParallelFunc(input->Slice(batch_input_offset),
      input_channel, input_height, input_width,
      filter_height, filter_width,
      output_height, output_width,
      padding_height, padding_width,
      stride_height, stride_width, unpack->data());
    #ifdef BLITZ_PERFORMANCE
    end = system_clock::now();
    unpack_time += end - start;
    #endif

    #ifdef BLITZ_PERFORMANCE
    start = system_clock::now();
    #endif
    // gemm generate
    // (output_channel) *
    // (input_channel * filter_height * filter_width)
    BlitzCPUGemm(false, true, dim_left, dim_right, dim_common,
      const_cast<CPUTensor<DType>*>(output)->Slice(batch_output_offset),
      unpack->data(), update->data(),
      static_cast<DType>(1), static_cast<DType>(1));
    batch_input_offset += input_channel * input_height * input_width;
    batch_output_offset += output_channel * output_height * output_width;
    #ifdef BLITZ_PERFORMANCE
    end = system_clock::now();
    gemm_time += end - start;
    #endif
  }
  #ifdef BLITZ_PERFORMANCE
  LOG(INFO) << "Backward convolution filter gemm: " << gemm_time.count();
  LOG(INFO) << "Backward convolution filter unpack: " << unpack_time.count();
  #endif  // BLITZ_PERFORMANCE
}

// batch parallel
template<typename DType>
void Backend<CPUTensor, DType>::Convolution2DForwardFunc(
  const CPUTensor<DType>* input, const CPUTensor<DType>* filter,
  const int padding_height, const int padding_width,
  const int stride_height, const int stride_width,
  vector<shared_ptr<CPUTensor<DType> > >* unpack_batch,
  CPUTensor<DType>* output) {
  // shape decode
  // input
  const Shape& input_shape = input->shape();
  const int batch_size = input_shape[0];
  const int input_channel = input_shape[1];
  const int input_height = input_shape[2];
  const int input_width = input_shape[3];
  // filter
  const Shape& filter_shape = filter->shape();
  const int filter_height = filter_shape[2];
  const int filter_width = filter_shape[3];
  // output
  const Shape& output_shape = output->shape();
  const int output_channel = output_shape[1];
  const int output_height = output_shape[2];
  const int output_width = output_shape[3];

  int batch_input_offset;
  int batch_output_offset;
  const int input_batch_offset = input_channel * input_height * input_width;
  const int output_batch_offset = output_channel * output_height * output_width;
  const int dim_left = output_channel;
  const int dim_right = output_height * output_width;
  const int dim_common = input_channel * filter_height * filter_width;

  #ifdef BLITZ_PERFORMANCE
  time_point<system_clock> start, end;
  duration<double> gemm_time = duration<double>::zero();
  duration<double> unpack_time = duration<double>::zero();
  double average_gemm_time = 0.0;
  double average_unpack_time = 0.0;
  #endif  // BLITZ_PERFORMANCE

  #pragma omp parallel private(batch_input_offset, batch_output_offset)
  {
    int tid = omp_get_thread_num();

    #ifdef BLITZ_PERFORMANCE
      #pragma omp for private(start, end)
    #else
      #pragma omp for
    #endif
    for (int batch_index = 0; batch_index < batch_size; ++batch_index) {
      batch_input_offset = batch_index * input_batch_offset;
      batch_output_offset = batch_index *  output_batch_offset;
      #ifdef BLITZ_PERFORMANCE
      start = system_clock::now();
      #endif  // BLITZ_PERFORMANCE
      // unpack
      // (input_channel) *
      // (input_width * input_height)
      // to
      // (input_channel * filter_height * filter_width)
      // (output_width * output_height)
      Unpack2DFunc(input->Slice(batch_input_offset),
        input_channel, input_height, input_width, filter_height, filter_width,
        output_height, output_width, padding_height, padding_width,
        stride_height, stride_width,
        (*unpack_batch)[tid]->data());
      #ifdef BLITZ_PERFORMANCE
      end = system_clock::now();
      #pragma omp critical
      unpack_time += end - start;
      #endif  // BLITZ_PERFORMANCE

      #ifdef BLITZ_PERFORMANCE
      start = system_clock::now();
      #endif  // BLITZ_PERFORMANCE
      // gemm generate
      // (output_channel) * (output_height * output_width)
      BlitzCPUGemm(false, false, dim_left, dim_right, dim_common,
      const_cast<CPUTensor<DType>*>(filter)->data(),
      (*unpack_batch)[tid]->data(),
      output->Slice(batch_output_offset),
      static_cast<DType>(1), static_cast<DType>(0));

      #ifdef BLITZ_PERFORMANCE
      end = system_clock::now();
      #pragma omp critical
      gemm_time += end - start;
      #endif  // BLITZ_PERFORMANCE
    }
    #ifdef BLITZ_PERFORMANCE
    if (tid == 0) {
      average_unpack_time = unpack_time.count() /
        omp_get_num_threads();
      average_gemm_time = gemm_time.count() /
        omp_get_num_threads();
    }
    #endif
  }
  #ifdef BLITZ_PERFORMANCE
  LOG(INFO) << "Forward convolution average gemm: " << average_gemm_time;
  LOG(INFO) << "Forward convolution average unpack: " << average_unpack_time;
  #endif  // BLITZ_PERFORMANCE
}

template<typename DType>
void Backend<CPUTensor, DType>::Convolution2DBackwardFunc(
  const CPUTensor<DType>* output, const CPUTensor<DType>* filter,
  const int padding_height, const int padding_width,
  const int stride_height, const int stride_width,
  vector<shared_ptr<CPUTensor<DType> > >* pack_batch,
  CPUTensor<DType>* input) {
  // shape decode
  // input
  const Shape& input_shape = input->shape();
  const int batch_size = input_shape[0];
  const int input_channel = input_shape[1];
  const int input_height = input_shape[2];
  const int input_width = input_shape[3];
  // filter
  const Shape& filter_shape = filter->shape();
  const int filter_height = filter_shape[2];
  const int filter_width = filter_shape[3];
  // output
  const Shape& output_shape = output->shape();
  const int output_channel = output_shape[1];
  const int output_height = output_shape[2];
  const int output_width = output_shape[3];

  int batch_input_offset;
  int batch_output_offset;
  const int input_batch_offset = input_channel * input_height * input_width;
  const int output_batch_offset = output_channel * output_height * output_width;
  const int dim_left = input_channel * filter_height * filter_width;
  const int dim_right = output_height * output_width;
  const int dim_common = output_channel;
  input->Fill(0);

  #ifdef BLITZ_PERFORMANCE
  time_point<system_clock> start, end;
  duration<double> gemm_time = duration<double>::zero();
  duration<double> pack_time = duration<double>::zero();
  double average_gemm_time = 0.0;
  double average_pack_time = 0.0;
  #endif  // BLITZ_PERFORMANCE

  #pragma omp parallel private(batch_input_offset, batch_output_offset)
  {
    int tid = omp_get_thread_num();

    #ifdef BLITZ_PERFORMANCE
      #pragma omp for private(start, end)
    #else
      #pragma omp for
    #endif
    for (int batch_index = 0; batch_index < batch_size; ++batch_index) {
      batch_input_offset = batch_index * input_batch_offset;
      batch_output_offset = batch_index * output_batch_offset;
      #ifdef BLITZ_PERFORMANCE
      start = system_clock::now();
      #endif  // BLITZ_PERFORMANCE
      // gemm generate
      // (output_width * output_height) *
      // (input_channel * filter_height * filter_width)
      BlitzCPUGemm(true, false, dim_left, dim_right, dim_common,
      const_cast<CPUTensor<DType>*>(filter)->data(),
      const_cast<CPUTensor<DType>*>(output)->Slice(batch_output_offset),
      (*pack_batch)[tid]->data(),
      static_cast<DType>(1), static_cast<DType>(0));
      #ifdef BLITZ_PERFORMANCE
      end = system_clock::now();
      #pragma omp critical
      gemm_time += end - start;
      #endif  // BLITZ_PERFORMANCE

      #ifdef BLITZ_PERFORMANCE
      start = system_clock::now();
      #endif  // BLITZ_PERFORMANCE
      // pack
      // (input_channel * filter_height * filter_width) *
      // (output_width * output_height)
      // to
      // (input_channel) *
      // (input_height * input_width)
      Pack2DFunc((*pack_batch)[tid]->data(),
        input_channel, input_height, input_width,
        filter_height, filter_width, output_height, output_width,
        padding_height, padding_width, stride_height, stride_width,
        input->Slice(batch_input_offset));
      #ifdef BLITZ_PERFORMANCE
      end = system_clock::now();
      #pragma omp critical
      pack_time += end - start;
      #endif  // BLITZ_PERFORMANCE
    }
    #ifdef BLITZ_PERFORMANCE
    if (tid == 0) {
      average_pack_time = pack_time.count() /
        omp_get_num_threads();
      average_gemm_time = gemm_time.count() /
        omp_get_num_threads();
    }
    #endif
  }

  #ifdef BLITZ_PERFORMANCE
  LOG(INFO) << "Backward convolution average gemm: " << average_gemm_time;
  LOG(INFO) << "Backward convolution average pack: " << average_pack_time;
  #endif  // BLITZ_PERFORMANCE
}

template<typename DType>
void Backend<CPUTensor, DType>::Convolution2DUpdateFunc(
  const CPUTensor<DType>* input, const CPUTensor<DType>* output,
  const int padding_height, const int padding_width,
  const int stride_height, const int stride_width,
  vector<shared_ptr<CPUTensor<DType> > >* unpack_batch,
  vector<shared_ptr<CPUTensor<DType> > >* update_batch,
  CPUTensor<DType>* update) {
  // shape decode
  // input
  const Shape& input_shape = input->shape();
  const int batch_size = input_shape[0];
  const int input_channel = input_shape[1];
  const int input_height = input_shape[2];
  const int input_width = input_shape[3];
  // filter
  const Shape& filter_shape = update->shape();
  const int filter_height = filter_shape[2];
  const int filter_width = filter_shape[3];
  // output
  const Shape& output_shape = output->shape();
  const int output_channel = output_shape[1];
  const int output_height = output_shape[2];
  const int output_width = output_shape[3];

  int batch_input_offset;
  int batch_output_offset;
  const int input_batch_offset = input_channel * input_height * input_width;
  const int output_batch_offset = output_channel * output_height * output_width;
  const int dim_left = output_channel;
  const int dim_right = input_channel * filter_height * filter_width;
  const int dim_common = output_height * output_width;

  #ifdef BLITZ_PERFORMANCE
  time_point<system_clock> start, end;
  duration<double> gemm_time = duration<double>::zero();
  duration<double> unpack_time = duration<double>::zero();
  double average_gemm_time = 0.0;
  double average_unpack_time = 0.0;
  #endif  // BLITZ_PERFORMANCE

  #pragma omp parallel private(batch_input_offset, batch_output_offset)
  {
    int tid = omp_get_thread_num();

    #ifdef BLITZ_PERFORMANCE
      #pragma omp for private(start, end)
    #else
      #pragma omp for
    #endif
    for (int batch_index = 0; batch_index < batch_size; ++batch_index) {
      batch_input_offset = batch_index * input_batch_offset;
      batch_output_offset = batch_index * output_batch_offset;
      #ifdef BLITZ_PERFORMANCE
      start = system_clock::now();
      #endif  // BLITZ_PERFORMANCE
      // unpack
      // (input_channel) *
      // (input_width * input_height)
      // to
      // (input_channel * filter_height * filter_width)
      // (output_width * output_height)
      Unpack2DFunc(input->Slice(batch_input_offset),
        input_channel, input_height, input_width,
        filter_height, filter_width, output_height, output_width,
        padding_height, padding_width, stride_height, stride_width,
        (*unpack_batch)[tid]->data());
      #ifdef BLITZ_PERFORMANCE
      end = system_clock::now();
      #pragma omp critical
      unpack_time += end - start;
      #endif  // BLITZ_PERFORMANCE

      #ifdef BLITZ_PERFORMANCE
      start = system_clock::now();
      #endif  // BLITZ_PERFORMANCE
      // gemm generate
      // (output_channel) *
      // (input_channel * filter_height * filter_width)
      BlitzCPUGemm(false, true, dim_left, dim_right, dim_common,
        const_cast<CPUTensor<DType>*>(output)->Slice(batch_output_offset),
        (*unpack_batch)[tid]->data(), (*update_batch)[tid]->data(),
        static_cast<DType>(1), static_cast<DType>(1));
      #ifdef BLITZ_PERFORMANCE
      end = system_clock::now();
      #pragma omp critical
      gemm_time += end - start;
      #endif  // BLITZ_PERFORMANCE
    }

    for (size_t i = 0; i < update->size(); ++i) {
      #pragma omp atomic
      (*update)[i] += (*(*update_batch)[tid])[i];
    }

    #ifdef BLITZ_PERFORMANCE
    if (tid == 0) {
      average_unpack_time = unpack_time.count() /
        omp_get_num_threads();
      average_gemm_time = gemm_time.count() /
        omp_get_num_threads();
    }
    #endif
  }
  #ifdef BLITZ_PERFORMANCE
  LOG(INFO) << "Backward convolution filter average gemm: " <<
    average_gemm_time;
  LOG(INFO) << "Backward convolution filter average unpack: " <<
    average_unpack_time;
  #endif  // BLITZ_PERFORMANCE
}

// naive parallel
template<typename DType>
void Backend<CPUTensor, DType>::Convolution2DForwardFunc(
  const CPUTensor<DType>* input, const CPUTensor<DType>* filter,
  const int stride_height, const int stride_width,
  CPUTensor<DType>* output) {
}

#endif  // SRC_BACKEND_CPU_BACKEND_CONV_INL_H_

