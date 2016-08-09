#ifndef SRC_BACKEND_CPU_BACKEND_CONV_INL_H_
#define SRC_BACKEND_CPU_BACKEND_CONV_INL_H_

#include <immintrin.h>
#include "backend/cpu_backend.h"

namespace blitz {

template<typename DType>
inline void Backend<CPUTensor, DType>::Convolution2DForwardFunc(
  const CPUTensor<DType>* input, const CPUTensor<DType>* weight,
  const int padding_height, const int padding_width,
  const int stride_height, const int stride_width,
  CPUTensor<DType>* unpack, CPUTensor<DType>* output) {
  // shape decode
  // input
  const Shape& input_shape = input->shape();
  int batch_size = input_shape[0];
  int input_channel = input_shape[1];
  int input_height = input_shape[2];
  int input_width = input_shape[3];
  // filter
  const Shape& filter_shape = weight->shape();
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
  std::chrono::time_point<std::chrono::system_clock> start, end;
  std::chrono::duration<double> gemm_time =
    std::chrono::duration<double>::zero();
  std::chrono::duration<double> unpack_time =
    std::chrono::duration<double>::zero();
#endif  // BLITZ_PERFORMANCE

  for (int batch_index = 0; batch_index < batch_size; ++batch_index) {
#ifdef BLITZ_PERFORMANCE
    start = std::chrono::system_clock::now();
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
    end = std::chrono::system_clock::now();
    unpack_time += end - start;
#endif

#ifdef BLITZ_PERFORMANCE
    start = std::chrono::system_clock::now();
#endif
    // gemm generate
    // (output_channel) * (output_height * output_width)
    BlitzCPUGemm(false, false, dim_left, dim_right, dim_common,
        const_cast<CPUTensor<DType>*>(weight)->data(),
        unpack->data(), output->Slice(batch_output_offset),
        static_cast<DType>(1), static_cast<DType>(0));
#ifdef BLITZ_PERFORMANCE
    end = std::chrono::system_clock::now();
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
inline void Backend<CPUTensor, DType>::Convolution2DBackwardFunc(
  const CPUTensor<DType>* output, const CPUTensor<DType>* weight,
  const int padding_height, const int padding_width,
  const int stride_height, const int stride_width,
  CPUTensor<DType>* pack, CPUTensor<DType>* input) {
  // shape decode
  // input
  const Shape& input_shape = input->shape();
  int batch_size = input_shape[0];
  int input_channel = input_shape[1];
  int input_height = input_shape[2];
  int input_width = input_shape[3];
  // filter
  const Shape& filter_shape = weight->shape();
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
  std::chrono::time_point<std::chrono::system_clock> start, end;
  std::chrono::duration<double> gemm_time = std::chrono::duration<double>::zero();
  std::chrono::duration<double> pack_time = std::chrono::duration<double>::zero();
  #endif  // BLITZ_PERFORMANCE

  for (int batch_index = 0; batch_index < batch_size; ++batch_index) {
    #ifdef BLITZ_PERFORMANCE
    start = std::chrono::system_clock::now();
    #endif
    // gemm generate
    // (output_width * output_height) *
    // (input_channel * filter_height * filter_width)
    BlitzCPUGemm(true, false, dim_left, dim_right, dim_common,
    const_cast<CPUTensor<DType>*>(weight)->data(),
    const_cast<CPUTensor<DType>*>(output)->Slice(batch_output_offset),
    pack->data(), static_cast<DType>(1), static_cast<DType>(0));
    #ifdef BLITZ_PERFORMANCE
    end = std::chrono::system_clock::now();
    gemm_time += end - start;
    #endif

    #ifdef BLITZ_PERFORMANCE
    start = std::chrono::system_clock::now();
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
    end = std::chrono::system_clock::now();
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
inline void Backend<CPUTensor, DType>::Convolution2DUpdateFunc(
  const CPUTensor<DType>* input, const CPUTensor<DType>* output,
  const int padding_height, const int padding_width,
  const int stride_height, const int stride_width,
  CPUTensor<DType>* unpack, CPUTensor<DType>* update) {
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
  update->Fill(0);
  #ifdef BLITZ_PERFORMANCE  // only valid for a single thread
  std::chrono::time_point<std::chrono::system_clock> start, end;
  std::chrono::duration<double> gemm_time = std::chrono::duration<double>::zero();
  std::chrono::duration<double> unpack_time = std::chrono::duration<double>::zero();
  #endif  // BLITZ_PERFORMANCE

  for (int batch_index = 0; batch_index < batch_size; ++batch_index) {
    #ifdef BLITZ_PERFORMANCE
    start = std::chrono::system_clock::now();
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
    end = std::chrono::system_clock::now();
    unpack_time += end - start;
    #endif

    #ifdef BLITZ_PERFORMANCE
    start = std::chrono::system_clock::now();
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
    end = std::chrono::system_clock::now();
    gemm_time += end - start;
    #endif
  }
  #ifdef BLITZ_PERFORMANCE
  LOG(INFO) << "Backward convolution weight gemm: " << gemm_time.count();
  LOG(INFO) << "Backward convolution weight unpack: " << unpack_time.count();
  #endif  // BLITZ_PERFORMANCE
}

// batch parallel
template<typename DType>
inline void Backend<CPUTensor, DType>::Convolution2DForwardFunc(
  const CPUTensor<DType>* input, const CPUTensor<DType>* weight,
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
  const Shape& filter_shape = weight->shape();
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
  std::chrono::time_point<std::chrono::system_clock> start, end;
  std::chrono::duration<double> gemm_time = std::chrono::duration<double>::zero();
  std::chrono::duration<double> unpack_time = std::chrono::duration<double>::zero();
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
      start = std::chrono::system_clock::now();
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
      end = std::chrono::system_clock::now();
      #pragma omp critical
      unpack_time += end - start;
      #endif  // BLITZ_PERFORMANCE

      #ifdef BLITZ_PERFORMANCE
      start = std::chrono::system_clock::now();
      #endif  // BLITZ_PERFORMANCE
      // gemm generate
      // (output_channel) * (output_height * output_width)
      BlitzCPUGemm(false, false, dim_left, dim_right, dim_common,
      const_cast<CPUTensor<DType>*>(weight)->data(),
      (*unpack_batch)[tid]->data(),
      output->Slice(batch_output_offset),
      static_cast<DType>(1), static_cast<DType>(0));

      #ifdef BLITZ_PERFORMANCE
      end = std::chrono::system_clock::now();
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
inline void Backend<CPUTensor, DType>::Convolution2DBackwardFunc(
  const CPUTensor<DType>* output, const CPUTensor<DType>* weight,
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
  const Shape& filter_shape = weight->shape();
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
  std::chrono::time_point<std::chrono::system_clock> start, end;
  std::chrono::duration<double> gemm_time = std::chrono::duration<double>::zero();
  std::chrono::duration<double> pack_time = std::chrono::duration<double>::zero();
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
      start = std::chrono::system_clock::now();
      #endif  // BLITZ_PERFORMANCE
      // gemm generate
      // (output_width * output_height) *
      // (input_channel * filter_height * filter_width)
      BlitzCPUGemm(true, false, dim_left, dim_right, dim_common,
      const_cast<CPUTensor<DType>*>(weight)->data(),
      const_cast<CPUTensor<DType>*>(output)->Slice(batch_output_offset),
      (*pack_batch)[tid]->data(),
      static_cast<DType>(1), static_cast<DType>(0));
      #ifdef BLITZ_PERFORMANCE
      end = std::chrono::system_clock::now();
      #pragma omp critical
      gemm_time += end - start;
      #endif  // BLITZ_PERFORMANCE

      #ifdef BLITZ_PERFORMANCE
      start = std::chrono::system_clock::now();
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
      end = std::chrono::system_clock::now();
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
inline void Backend<CPUTensor, DType>::Convolution2DUpdateFunc(
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
  update->Fill(0);

  #ifdef BLITZ_PERFORMANCE
  std::chrono::time_point<std::chrono::system_clock> start, end;
  std::chrono::duration<double> gemm_time = std::chrono::duration<double>::zero();
  std::chrono::duration<double> unpack_time = std::chrono::duration<double>::zero();
  double average_gemm_time = 0.0;
  double average_unpack_time = 0.0;
  #endif  // BLITZ_PERFORMANCE

  #pragma omp parallel private(batch_input_offset, batch_output_offset)
  {
    int tid = omp_get_thread_num();
    (*update_batch)[tid]->Fill(0);

    #ifdef BLITZ_PERFORMANCE
      #pragma omp for private(start, end)
    #else
      #pragma omp for
    #endif
    for (int batch_index = 0; batch_index < batch_size; ++batch_index) {
      batch_input_offset = batch_index * input_batch_offset;
      batch_output_offset = batch_index * output_batch_offset;
      #ifdef BLITZ_PERFORMANCE
      start = std::chrono::system_clock::now();
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
      end = std::chrono::system_clock::now();
      #pragma omp critical
      unpack_time += end - start;
      #endif  // BLITZ_PERFORMANCE

      #ifdef BLITZ_PERFORMANCE
      start = std::chrono::system_clock::now();
      #endif  // BLITZ_PERFORMANCE
      // gemm generate
      // (output_channel) *
      // (input_channel * filter_height * filter_width)
      BlitzCPUGemm(false, true, dim_left, dim_right, dim_common,
        const_cast<CPUTensor<DType>*>(output)->Slice(batch_output_offset),
        (*unpack_batch)[tid]->data(), (*update_batch)[tid]->data(),
        static_cast<DType>(1), static_cast<DType>(1));
      #ifdef BLITZ_PERFORMANCE
      end = std::chrono::system_clock::now();
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
  LOG(INFO) << "Backward convolution weight average gemm: " << average_gemm_time;
  LOG(INFO) << "Backward convolution weight average unpack: " << average_unpack_time;
  #endif  // BLITZ_PERFORMANCE
}

// naive parallel
template<typename DType>
inline void Backend<CPUTensor, DType>::Convolution2DForwardFunc(
  const CPUTensor<DType>* input, const CPUTensor<DType>* weight,
  const int stride_height, const int stride_width,
  CPUTensor<DType>* output) {
  // shape decode
  // input
  const Shape& input_shape = input->shape();
  const int batch_size = input_shape[0];
  const int input_channel = input_shape[1];
  const int input_height = input_shape[2];
  const int input_width = input_shape[3];
  // filter
  const Shape& filter_shape = weight->shape();
  const int filter_height = filter_shape[2];
  const int filter_width = filter_shape[3];
  // output
  const Shape& output_shape = output->shape();
  const int output_channel = output_shape[1];
  const int output_height = output_shape[2];
  const int output_width = output_shape[3];

  const int output_channel_offset = output_height * output_width;
  const int output_batch_offset = output_channel * output_channel_offset;
  const int input_channel_offset = input_height * input_width;
  const int input_batch_offset = input_channel * input_channel_offset;
  const int filter_input_channel_offset =
    filter_height * filter_width;
  const int filter_output_channel_offset =
    input_channel * filter_height * filter_width;

  #ifdef BLITZ_PERFORMANCE  // only for single thread
  std::chrono::time_point<std::chrono::system_clock> total_start, total_end;
  std::chrono::duration<double> total_time = std::chrono::duration<double>::zero();
  total_start = std::chrono::system_clock::now();
  #endif  // BLITZ_PERFORMANCE

  const int BB = 8;
  const int BB2 = 2 * BB;
  const int BB3 = 3 * BB;
  const int OCB = 4;
  const int ICB = 4;
  const int OWB = 4;
  const int FWB = 4;
  const int BB_output_channel_offset = BB * output_channel_offset;
  const int BB_input_channel_offset = BB * input_channel_offset;
  #pragma omp parallel for
  for (int i = 0; i < batch_size; i += BB) {
    DType* output_slice = output->Slice(output_batch_offset * i);
    const DType* input_slice = input->Slice(input_batch_offset * i);
    __m256 output_reg00, output_reg01, output_reg02, output_reg03,
    output_reg10, output_reg11, output_reg12, output_reg13,
    output_reg20, output_reg21, output_reg22, output_reg23,
    output_reg30, output_reg31, output_reg32, output_reg33,
    filter_reg00, filter_reg01, filter_reg02, filter_reg03,
    filter_reg10, filter_reg11, filter_reg12, filter_reg13,
    filter_reg20, filter_reg21, filter_reg22, filter_reg23,
    filter_reg30, filter_reg31, filter_reg32, filter_reg33,
    input_reg00, input_reg01, input_reg02, input_reg03,
    input_reg10, input_reg11, input_reg12, input_reg13,
    input_reg20, input_reg21, input_reg22, input_reg23,
    input_reg30, input_reg31, input_reg32, input_reg33;
    for (int output_channel_index = 0; output_channel_index < output_channel;
        output_channel_index += OCB) {
      int input_channel_index = 0;
      if (input_channel >= ICB) {
        for (; input_channel_index < input_channel - ICB; input_channel_index += ICB) {
          DType* p_output = output_slice +
            BB * (output_channel_index * output_channel_offset); // p_output
          const DType* p_input = input_slice +
            BB * (input_channel_index * input_channel_offset); // p_input
          const DType* filter_slice = weight->Slice(
              input_channel_index * filter_input_channel_offset +
              output_channel_index * filter_output_channel_offset);  // filter_slice
          for (int output_height_index = 0; output_height_index < output_height;
              ++output_height_index) {
            int output_width_index = 0;
            if (output_width >= OWB) {
              for (; output_width_index < output_width - OWB; output_width_index += OWB) {
                output_reg00 = _mm256_load_ps(p_output);
                output_reg10 = _mm256_load_ps(p_output + BB);
                output_reg20 = _mm256_load_ps(p_output + BB2);
                output_reg30 = _mm256_load_ps(p_output + BB3);
                output_reg01 = _mm256_load_ps(p_output + BB_output_channel_offset);
                output_reg11 = _mm256_load_ps(p_output + BB_output_channel_offset + BB);
                output_reg21 = _mm256_load_ps(p_output + BB_output_channel_offset + BB2);
                output_reg31 = _mm256_load_ps(p_output + BB_output_channel_offset + BB3);
                output_reg02 = _mm256_load_ps(p_output + 2 * BB_output_channel_offset);
                output_reg12 = _mm256_load_ps(p_output + 2 * BB_output_channel_offset + BB);
                output_reg22 = _mm256_load_ps(p_output + 2 * BB_output_channel_offset + BB2);
                output_reg32 = _mm256_load_ps(p_output + 2 * BB_output_channel_offset + BB3);
                output_reg03 = _mm256_load_ps(p_output + 3 * BB_output_channel_offset);
                output_reg13 = _mm256_load_ps(p_output + 3 * BB_output_channel_offset + BB);
                output_reg23 = _mm256_load_ps(p_output + 3 * BB_output_channel_offset + BB2);
                output_reg33 = _mm256_load_ps(p_output + 3 * BB_output_channel_offset + BB3);
                const DType* p_input_slice1 = p_input +
                  (output_height_index * stride_height * input_width +
                   output_width_index * stride_width) * BB;
                const DType* p_input_slice2 = p_input_slice1 + stride_width * BB;
                const DType* p_input_slice3 = p_input_slice2 + stride_width * BB;
                const DType* p_input_slice4 = p_input_slice3 + stride_width * BB;
                const DType* p_filter_slice = filter_slice;
                for (int filter_height_index = 0; filter_height_index < filter_height;
                    ++filter_height_index) {
                  int filter_width_index = 0;
                  for (; filter_width_index < filter_width; filter_width_index++) {
                    filter_reg00 = _mm256_broadcast_ss(p_filter_slice);
                    filter_reg10 = _mm256_broadcast_ss(p_filter_slice + filter_input_channel_offset);
                    filter_reg20 = _mm256_broadcast_ss(p_filter_slice + 2 * filter_input_channel_offset);
                    filter_reg30 = _mm256_broadcast_ss(p_filter_slice + 3 * filter_input_channel_offset);
                    filter_reg01 = _mm256_broadcast_ss(p_filter_slice + filter_output_channel_offset);
                    filter_reg11 = _mm256_broadcast_ss(p_filter_slice + filter_output_channel_offset + filter_input_channel_offset);
                    filter_reg21 = _mm256_broadcast_ss(p_filter_slice + filter_output_channel_offset + 2 * filter_input_channel_offset);
                    filter_reg31 = _mm256_broadcast_ss(p_filter_slice + filter_output_channel_offset + 3 * filter_input_channel_offset);
                    filter_reg02 = _mm256_broadcast_ss(p_filter_slice + 2 * filter_output_channel_offset);
                    filter_reg12 = _mm256_broadcast_ss(p_filter_slice + 2 * filter_output_channel_offset + filter_input_channel_offset);
                    filter_reg22 = _mm256_broadcast_ss(p_filter_slice + 2 * filter_output_channel_offset + 2 * filter_input_channel_offset);
                    filter_reg32 = _mm256_broadcast_ss(p_filter_slice + 2 * filter_output_channel_offset + 3 * filter_input_channel_offset);
                    filter_reg03 = _mm256_broadcast_ss(p_filter_slice + 3 * filter_output_channel_offset);
                    filter_reg13 = _mm256_broadcast_ss(p_filter_slice + 3 * filter_output_channel_offset + filter_input_channel_offset);
                    filter_reg23 = _mm256_broadcast_ss(p_filter_slice + 3 * filter_output_channel_offset + 2 * filter_input_channel_offset);
                    filter_reg33 = _mm256_broadcast_ss(p_filter_slice + 3 * filter_output_channel_offset + 3 * filter_input_channel_offset);
                    input_reg00 = _mm256_load_ps(p_input_slice1);
                    input_reg01 = _mm256_load_ps(p_input_slice1 + BB_input_channel_offset);
                    input_reg02 = _mm256_load_ps(p_input_slice1 + 2 * BB_input_channel_offset);
                    input_reg03 = _mm256_load_ps(p_input_slice1 + 3 * BB_input_channel_offset);
                    input_reg10 = _mm256_load_ps(p_input_slice2);
                    input_reg11 = _mm256_load_ps(p_input_slice2 + BB_input_channel_offset);
                    input_reg12 = _mm256_load_ps(p_input_slice2 + 2 * BB_input_channel_offset);
                    input_reg13 = _mm256_load_ps(p_input_slice2 + 3 * BB_input_channel_offset);
                    input_reg20 = _mm256_load_ps(p_input_slice3);
                    input_reg21 = _mm256_load_ps(p_input_slice3 + BB_input_channel_offset);
                    input_reg22 = _mm256_load_ps(p_input_slice3 + 2 * BB_input_channel_offset);
                    input_reg23 = _mm256_load_ps(p_input_slice3 + 3 * BB_input_channel_offset);
                    input_reg30 = _mm256_load_ps(p_input_slice4);
                    input_reg31 = _mm256_load_ps(p_input_slice4 + BB * input_channel_offset);
                    input_reg32 = _mm256_load_ps(p_input_slice4 + 2 * BB * input_channel_offset);
                    input_reg33 = _mm256_load_ps(p_input_slice4 + 3 * BB * input_channel_offset);

                    //first row
                    output_reg00 = _mm256_add_ps(output_reg00,
                        _mm256_add_ps(
                          _mm256_add_ps(
                            _mm256_mul_ps(input_reg00, filter_reg00),
                            _mm256_mul_ps(input_reg01, filter_reg10)),
                          _mm256_add_ps(
                            _mm256_mul_ps(input_reg02, filter_reg20),
                            _mm256_mul_ps(input_reg03, filter_reg30))));
                    output_reg01 = _mm256_add_ps(output_reg01,
                        _mm256_add_ps(
                          _mm256_add_ps(
                            _mm256_mul_ps(input_reg00, filter_reg01),
                            _mm256_mul_ps(input_reg01, filter_reg11)),
                          _mm256_add_ps(
                            _mm256_mul_ps(input_reg02, filter_reg21),
                            _mm256_mul_ps(input_reg03, filter_reg31))));
                    output_reg02 = _mm256_add_ps(output_reg02,
                        _mm256_add_ps(
                          _mm256_add_ps(
                            _mm256_mul_ps(input_reg00, filter_reg02),
                            _mm256_mul_ps(input_reg01, filter_reg12)),
                          _mm256_add_ps(
                            _mm256_mul_ps(input_reg02, filter_reg22),
                            _mm256_mul_ps(input_reg03, filter_reg32))));
                    output_reg03 = _mm256_add_ps(output_reg03,
                        _mm256_add_ps(
                          _mm256_add_ps(
                            _mm256_mul_ps(input_reg00, filter_reg03),
                            _mm256_mul_ps(input_reg01, filter_reg13)),
                          _mm256_add_ps(
                            _mm256_mul_ps(input_reg02, filter_reg23),
                            _mm256_mul_ps(input_reg03, filter_reg33))));

                    //second row
                    output_reg10 = _mm256_add_ps(output_reg10,
                        _mm256_add_ps(
                          _mm256_add_ps(
                            _mm256_mul_ps(input_reg10, filter_reg00),
                            _mm256_mul_ps(input_reg11, filter_reg10)),
                          _mm256_add_ps(
                            _mm256_mul_ps(input_reg12, filter_reg20),
                            _mm256_mul_ps(input_reg13, filter_reg30))));
                    output_reg11 = _mm256_add_ps(output_reg11,
                        _mm256_add_ps(
                          _mm256_add_ps(
                            _mm256_mul_ps(input_reg10, filter_reg01),
                            _mm256_mul_ps(input_reg11, filter_reg11)),
                          _mm256_add_ps(
                            _mm256_mul_ps(input_reg12, filter_reg21),
                            _mm256_mul_ps(input_reg13, filter_reg31))));
                    output_reg12 = _mm256_add_ps(output_reg12,
                        _mm256_add_ps(
                          _mm256_add_ps(
                            _mm256_mul_ps(input_reg10, filter_reg02),
                            _mm256_mul_ps(input_reg11, filter_reg12)),
                          _mm256_add_ps(
                            _mm256_mul_ps(input_reg12, filter_reg22),
                            _mm256_mul_ps(input_reg13, filter_reg32))));
                    output_reg13 = _mm256_add_ps(output_reg13,
                        _mm256_add_ps(
                          _mm256_add_ps(
                            _mm256_mul_ps(input_reg10, filter_reg03),
                            _mm256_mul_ps(input_reg11, filter_reg13)),
                          _mm256_add_ps(
                            _mm256_mul_ps(input_reg12, filter_reg23),
                            _mm256_mul_ps(input_reg13, filter_reg33))));

                    //third row
                    output_reg20 = _mm256_add_ps(output_reg20,
                        _mm256_add_ps(
                          _mm256_add_ps(
                            _mm256_mul_ps(input_reg20, filter_reg00),
                            _mm256_mul_ps(input_reg21, filter_reg10)),
                          _mm256_add_ps(
                            _mm256_mul_ps(input_reg22, filter_reg20),
                            _mm256_mul_ps(input_reg23, filter_reg30))));
                    output_reg21 = _mm256_add_ps(output_reg21,
                        _mm256_add_ps(
                          _mm256_add_ps(
                            _mm256_mul_ps(input_reg20, filter_reg01),
                            _mm256_mul_ps(input_reg21, filter_reg11)),
                          _mm256_add_ps(
                            _mm256_mul_ps(input_reg22, filter_reg21),
                            _mm256_mul_ps(input_reg23, filter_reg31))));
                    output_reg22 = _mm256_add_ps(output_reg22,
                        _mm256_add_ps(
                          _mm256_add_ps(
                            _mm256_mul_ps(input_reg20, filter_reg02),
                            _mm256_mul_ps(input_reg21, filter_reg12)),
                          _mm256_add_ps(
                            _mm256_mul_ps(input_reg22, filter_reg22),
                            _mm256_mul_ps(input_reg23, filter_reg32))));
                    output_reg23 = _mm256_add_ps(output_reg23,
                        _mm256_add_ps(
                          _mm256_add_ps(
                            _mm256_mul_ps(input_reg20, filter_reg03),
                            _mm256_mul_ps(input_reg21, filter_reg13)),
                          _mm256_add_ps(
                            _mm256_mul_ps(input_reg22, filter_reg23),
                            _mm256_mul_ps(input_reg23, filter_reg33))));

                    //fourth row
                    output_reg30 = _mm256_add_ps(output_reg30,
                        _mm256_add_ps(
                          _mm256_add_ps(
                            _mm256_mul_ps(input_reg30, filter_reg00),
                            _mm256_mul_ps(input_reg31, filter_reg10)),
                          _mm256_add_ps(
                            _mm256_mul_ps(input_reg32, filter_reg20),
                            _mm256_mul_ps(input_reg33, filter_reg30))));
                    output_reg31 = _mm256_add_ps(output_reg31,
                        _mm256_add_ps(
                          _mm256_add_ps(
                            _mm256_mul_ps(input_reg30, filter_reg01),
                            _mm256_mul_ps(input_reg31, filter_reg11)),
                          _mm256_add_ps(
                            _mm256_mul_ps(input_reg32, filter_reg21),
                            _mm256_mul_ps(input_reg33, filter_reg31))));
                    output_reg32 = _mm256_add_ps(output_reg32,
                        _mm256_add_ps(
                          _mm256_add_ps(
                            _mm256_mul_ps(input_reg30, filter_reg02),
                            _mm256_mul_ps(input_reg31, filter_reg12)),
                          _mm256_add_ps(
                            _mm256_mul_ps(input_reg32, filter_reg22),
                            _mm256_mul_ps(input_reg33, filter_reg32))));
                    output_reg33 = _mm256_add_ps(output_reg33,
                        _mm256_add_ps(
                          _mm256_add_ps(
                            _mm256_mul_ps(input_reg30, filter_reg03),
                            _mm256_mul_ps(input_reg31, filter_reg13)),
                          _mm256_add_ps(
                            _mm256_mul_ps(input_reg32, filter_reg23),
                            _mm256_mul_ps(input_reg33, filter_reg33))));

                    p_input_slice1 += BB;
                    p_input_slice2 += BB;
                    p_input_slice3 += BB;
                    p_input_slice4 += BB;
                    p_filter_slice += 1;
                  }
                  p_input_slice1 += (input_width - filter_width) * BB;
                  p_input_slice2 += (input_width - filter_width) * BB;
                  p_input_slice3 += (input_width - filter_width) * BB;
                  p_input_slice4 += (input_width - filter_width) * BB;
                }
                _mm256_store_ps(p_output, output_reg00);
                _mm256_store_ps(p_output + BB, output_reg10);
                _mm256_store_ps(p_output + BB2, output_reg20);
                _mm256_store_ps(p_output + BB3, output_reg30);
                _mm256_store_ps(p_output + BB_output_channel_offset, output_reg01);
                _mm256_store_ps(p_output + BB_output_channel_offset + BB, output_reg11);
                _mm256_store_ps(p_output + BB_output_channel_offset + BB2, output_reg21);
                _mm256_store_ps(p_output + BB_output_channel_offset + BB3, output_reg31);
                _mm256_store_ps(p_output + 2 * BB_output_channel_offset, output_reg02);
                _mm256_store_ps(p_output + 2 * BB_output_channel_offset + BB, output_reg12);
                _mm256_store_ps(p_output + 2 * BB_output_channel_offset + BB2, output_reg22);
                _mm256_store_ps(p_output + 2 * BB_output_channel_offset + BB3, output_reg32);
                _mm256_store_ps(p_output + 3 * BB_output_channel_offset, output_reg03);
                _mm256_store_ps(p_output + 3 * BB_output_channel_offset + BB, output_reg13);
                _mm256_store_ps(p_output + 3 * BB_output_channel_offset + BB2, output_reg23);
                _mm256_store_ps(p_output + 3 * BB_output_channel_offset + BB3, output_reg33);
                p_output += OWB * BB;
              }
            }
            for (; output_width_index < output_width; ++output_width_index) {
              output_reg00 = _mm256_load_ps(p_output);
              output_reg01 = _mm256_load_ps(p_output + BB_output_channel_offset);
              output_reg02 = _mm256_load_ps(p_output + 2 * BB_output_channel_offset);
              output_reg03 = _mm256_load_ps(p_output + 3 * BB_output_channel_offset);
              const DType* p_input_slice1 = p_input +
                BB * (output_height_index * stride_height * input_width +
                    output_width_index * stride_width);
              const DType* p_filter_slice = filter_slice;
              for (int filter_height_index = 0; filter_height_index < filter_height;
                  ++filter_height_index) {
                int filter_width_index = 0;
                for (; filter_width_index < filter_width; filter_width_index++) {
                  filter_reg00 = _mm256_broadcast_ss(p_filter_slice);
                  filter_reg10 = _mm256_broadcast_ss(p_filter_slice + filter_input_channel_offset);
                  filter_reg20 = _mm256_broadcast_ss(p_filter_slice + 2 * filter_input_channel_offset);
                  filter_reg30 = _mm256_broadcast_ss(p_filter_slice + 3 * filter_input_channel_offset);
                  filter_reg01 = _mm256_broadcast_ss(p_filter_slice + filter_output_channel_offset);
                  filter_reg11 = _mm256_broadcast_ss(p_filter_slice + filter_output_channel_offset + filter_input_channel_offset);
                  filter_reg21 = _mm256_broadcast_ss(p_filter_slice + filter_output_channel_offset + 2 * filter_input_channel_offset);
                  filter_reg31 = _mm256_broadcast_ss(p_filter_slice + filter_output_channel_offset + 3 * filter_input_channel_offset);
                  filter_reg02 = _mm256_broadcast_ss(p_filter_slice + 2 * filter_output_channel_offset);
                  filter_reg12 = _mm256_broadcast_ss(p_filter_slice + 2 * filter_output_channel_offset + filter_input_channel_offset);
                  filter_reg22 = _mm256_broadcast_ss(p_filter_slice + 2 * filter_output_channel_offset + 2 * filter_input_channel_offset);
                  filter_reg32 = _mm256_broadcast_ss(p_filter_slice + 2 * filter_output_channel_offset + 3 * filter_input_channel_offset);
                  filter_reg03 = _mm256_broadcast_ss(p_filter_slice + 3 * filter_output_channel_offset);
                  filter_reg13 = _mm256_broadcast_ss(p_filter_slice + 3 * filter_output_channel_offset + filter_input_channel_offset);
                  filter_reg23 = _mm256_broadcast_ss(p_filter_slice + 3 * filter_output_channel_offset + 2 * filter_input_channel_offset);
                  filter_reg33 = _mm256_broadcast_ss(p_filter_slice + 3 * filter_output_channel_offset + 3 * filter_input_channel_offset);
                  input_reg00 = _mm256_load_ps(p_input_slice1);
                  input_reg01 = _mm256_load_ps(p_input_slice1 + BB_input_channel_offset);
                  input_reg02 = _mm256_load_ps(p_input_slice1 + 2 * BB_input_channel_offset);
                  input_reg03 = _mm256_load_ps(p_input_slice1 + 3 * BB_input_channel_offset);

                  //first row
                  output_reg00 = _mm256_add_ps(output_reg00,
                      _mm256_add_ps(
                        _mm256_add_ps(
                          _mm256_mul_ps(input_reg00, filter_reg00),
                          _mm256_mul_ps(input_reg01, filter_reg10)),
                        _mm256_add_ps(
                          _mm256_mul_ps(input_reg02, filter_reg20),
                          _mm256_mul_ps(input_reg03, filter_reg30))));
                  output_reg01 = _mm256_add_ps(output_reg01,
                      _mm256_add_ps(
                        _mm256_add_ps(
                          _mm256_mul_ps(input_reg00, filter_reg01),
                          _mm256_mul_ps(input_reg01, filter_reg11)),
                        _mm256_add_ps(
                          _mm256_mul_ps(input_reg02, filter_reg21),
                          _mm256_mul_ps(input_reg03, filter_reg31))));
                  output_reg02 = _mm256_add_ps(output_reg02,
                      _mm256_add_ps(
                        _mm256_add_ps(
                          _mm256_mul_ps(input_reg00, filter_reg02),
                          _mm256_mul_ps(input_reg01, filter_reg12)),
                        _mm256_add_ps(
                          _mm256_mul_ps(input_reg02, filter_reg22),
                          _mm256_mul_ps(input_reg03, filter_reg32))));
                  output_reg03 = _mm256_add_ps(output_reg03,
                      _mm256_add_ps(
                        _mm256_add_ps(
                          _mm256_mul_ps(input_reg00, filter_reg03),
                          _mm256_mul_ps(input_reg01, filter_reg13)),
                        _mm256_add_ps(
                          _mm256_mul_ps(input_reg02, filter_reg23),
                          _mm256_mul_ps(input_reg03, filter_reg33))));

                  p_input_slice1 += BB;
                  p_filter_slice += 1;
                }
                p_input_slice1 += (input_width - filter_width) * BB;
              }
              _mm256_store_ps(p_output, output_reg00);
              _mm256_store_ps(p_output + BB_output_channel_offset, output_reg01);
              _mm256_store_ps(p_output + 2 * BB_output_channel_offset, output_reg02);
              _mm256_store_ps(p_output + 3 * BB_output_channel_offset, output_reg03);
              p_output += BB;
            }
          }
        }
      }
      for (; input_channel_index < input_channel; input_channel_index++) {
        DType* p_output = output_slice +
          BB * (output_channel_index * output_channel_offset); // p_output
        const DType* p_input = input_slice +
          BB * (input_channel_index * input_channel_offset); // p_input
        const DType* filter_slice = weight->Slice(
          input_channel_index * filter_input_channel_offset +
          output_channel_index * filter_output_channel_offset);  // filter_slice
        for (int output_height_index = 0; output_height_index < output_height;
            ++output_height_index) {
          int output_width_index = 0;
          if (output_width >= OWB) {
            for (; output_width_index < output_width - OWB; output_width_index += OWB) {
              output_reg00 = _mm256_load_ps(p_output);
              output_reg10 = _mm256_load_ps(p_output + BB);
              output_reg20 = _mm256_load_ps(p_output + BB2);
              output_reg30 = _mm256_load_ps(p_output + BB3);
              output_reg01 = _mm256_load_ps(p_output + BB_output_channel_offset);
              output_reg11 = _mm256_load_ps(p_output + BB_output_channel_offset + BB);
              output_reg21 = _mm256_load_ps(p_output + BB_output_channel_offset + BB2);
              output_reg31 = _mm256_load_ps(p_output + BB_output_channel_offset + BB3);
              output_reg02 = _mm256_load_ps(p_output + 2 * BB_output_channel_offset);
              output_reg12 = _mm256_load_ps(p_output + 2 * BB_output_channel_offset + BB);
              output_reg22 = _mm256_load_ps(p_output + 2 * BB_output_channel_offset + BB2);
              output_reg32 = _mm256_load_ps(p_output + 2 * BB_output_channel_offset + BB3);
              output_reg03 = _mm256_load_ps(p_output + 3 * BB_output_channel_offset);
              output_reg13 = _mm256_load_ps(p_output + 3 * BB_output_channel_offset + BB);
              output_reg23 = _mm256_load_ps(p_output + 3 * BB_output_channel_offset + BB2);
              output_reg33 = _mm256_load_ps(p_output + 3 * BB_output_channel_offset + BB3);
              const DType* p_input_slice1 = p_input +
                (output_height_index * stride_height * input_width +
                 output_width_index * stride_width) * BB;
              const DType* p_input_slice2 = p_input_slice1 + stride_width * BB;
              const DType* p_input_slice3 = p_input_slice2 + stride_width * BB;
              const DType* p_input_slice4 = p_input_slice3 + stride_width * BB;
              const DType* p_filter_slice = filter_slice;
              for (int filter_height_index = 0; filter_height_index < filter_height;
                ++filter_height_index) {
                int filter_width_index = 0;
                if (filter_width >= FWB) {
                  for (; filter_width_index < filter_width - FWB; filter_width_index += FWB) {
                    filter_reg00 = _mm256_broadcast_ss(p_filter_slice);
                    filter_reg01 = _mm256_broadcast_ss(p_filter_slice + filter_output_channel_offset);
                    filter_reg02 = _mm256_broadcast_ss(p_filter_slice + 2 * filter_output_channel_offset);
                    filter_reg03 = _mm256_broadcast_ss(p_filter_slice + 3 * filter_output_channel_offset);
                    filter_reg10 = _mm256_broadcast_ss(p_filter_slice + 1);
                    filter_reg11 = _mm256_broadcast_ss(p_filter_slice + 1 + filter_output_channel_offset);
                    filter_reg12 = _mm256_broadcast_ss(p_filter_slice + 1 + 2 * filter_output_channel_offset);
                    filter_reg13 = _mm256_broadcast_ss(p_filter_slice + 1 + 3 * filter_output_channel_offset);
                    filter_reg20 = _mm256_broadcast_ss(p_filter_slice + 2);
                    filter_reg21 = _mm256_broadcast_ss(p_filter_slice + 2 + filter_output_channel_offset);
                    filter_reg22 = _mm256_broadcast_ss(p_filter_slice + 2 + 2 * filter_output_channel_offset);
                    filter_reg23 = _mm256_broadcast_ss(p_filter_slice + 2 + 3 * filter_output_channel_offset);
                    filter_reg30 = _mm256_broadcast_ss(p_filter_slice + 3);
                    filter_reg31 = _mm256_broadcast_ss(p_filter_slice + 3 + filter_output_channel_offset);
                    filter_reg32 = _mm256_broadcast_ss(p_filter_slice + 3 + 2 * filter_output_channel_offset);
                    filter_reg33 = _mm256_broadcast_ss(p_filter_slice + 3 + 3 * filter_output_channel_offset);
                    input_reg00 = _mm256_load_ps(p_input_slice1);
                    input_reg01 = _mm256_load_ps(p_input_slice1 + BB);
                    input_reg02 = _mm256_load_ps(p_input_slice1 + BB2);
                    input_reg03 = _mm256_load_ps(p_input_slice1 + BB3);
                    input_reg10 = _mm256_load_ps(p_input_slice2);
                    input_reg11 = _mm256_load_ps(p_input_slice2 + BB);
                    input_reg12 = _mm256_load_ps(p_input_slice2 + BB2);
                    input_reg13 = _mm256_load_ps(p_input_slice2 + BB3);
                    input_reg20 = _mm256_load_ps(p_input_slice3);
                    input_reg21 = _mm256_load_ps(p_input_slice3 + BB);
                    input_reg22 = _mm256_load_ps(p_input_slice3 + BB2);
                    input_reg23 = _mm256_load_ps(p_input_slice3 + BB3);
                    input_reg30 = _mm256_load_ps(p_input_slice4);
                    input_reg31 = _mm256_load_ps(p_input_slice4 + BB);
                    input_reg32 = _mm256_load_ps(p_input_slice4 + BB2);
                    input_reg33 = _mm256_load_ps(p_input_slice4 + BB3);

                    //first row
                    output_reg00 = _mm256_add_ps(output_reg00,
                        _mm256_add_ps(
                          _mm256_add_ps(
                            _mm256_mul_ps(input_reg00, filter_reg00),
                            _mm256_mul_ps(input_reg01, filter_reg10)),
                          _mm256_add_ps(
                            _mm256_mul_ps(input_reg02, filter_reg20),
                            _mm256_mul_ps(input_reg03, filter_reg30))));
                    output_reg01 = _mm256_add_ps(output_reg01,
                        _mm256_add_ps(
                          _mm256_add_ps(
                            _mm256_mul_ps(input_reg00, filter_reg01),
                            _mm256_mul_ps(input_reg01, filter_reg11)),
                          _mm256_add_ps(
                            _mm256_mul_ps(input_reg02, filter_reg21),
                            _mm256_mul_ps(input_reg03, filter_reg31))));
                    output_reg02 = _mm256_add_ps(output_reg02,
                        _mm256_add_ps(
                          _mm256_add_ps(
                            _mm256_mul_ps(input_reg00, filter_reg02),
                            _mm256_mul_ps(input_reg01, filter_reg12)),
                          _mm256_add_ps(
                            _mm256_mul_ps(input_reg02, filter_reg22),
                            _mm256_mul_ps(input_reg03, filter_reg32))));
                    output_reg03 = _mm256_add_ps(output_reg03,
                        _mm256_add_ps(
                          _mm256_add_ps(
                            _mm256_mul_ps(input_reg00, filter_reg03),
                            _mm256_mul_ps(input_reg01, filter_reg13)),
                          _mm256_add_ps(
                            _mm256_mul_ps(input_reg02, filter_reg23),
                            _mm256_mul_ps(input_reg03, filter_reg33))));

                    //second row
                    output_reg10 = _mm256_add_ps(output_reg10,
                        _mm256_add_ps(
                          _mm256_add_ps(
                            _mm256_mul_ps(input_reg10, filter_reg00),
                            _mm256_mul_ps(input_reg11, filter_reg10)),
                          _mm256_add_ps(
                            _mm256_mul_ps(input_reg12, filter_reg20),
                            _mm256_mul_ps(input_reg13, filter_reg30))));
                    output_reg11 = _mm256_add_ps(output_reg11,
                        _mm256_add_ps(
                          _mm256_add_ps(
                            _mm256_mul_ps(input_reg10, filter_reg01),
                            _mm256_mul_ps(input_reg11, filter_reg11)),
                          _mm256_add_ps(
                            _mm256_mul_ps(input_reg12, filter_reg21),
                            _mm256_mul_ps(input_reg13, filter_reg31))));
                    output_reg12 = _mm256_add_ps(output_reg12,
                        _mm256_add_ps(
                          _mm256_add_ps(
                            _mm256_mul_ps(input_reg10, filter_reg02),
                            _mm256_mul_ps(input_reg11, filter_reg12)),
                          _mm256_add_ps(
                            _mm256_mul_ps(input_reg12, filter_reg22),
                            _mm256_mul_ps(input_reg13, filter_reg32))));
                    output_reg13 = _mm256_add_ps(output_reg13,
                        _mm256_add_ps(
                          _mm256_add_ps(
                            _mm256_mul_ps(input_reg10, filter_reg03),
                            _mm256_mul_ps(input_reg11, filter_reg13)),
                          _mm256_add_ps(
                            _mm256_mul_ps(input_reg12, filter_reg23),
                            _mm256_mul_ps(input_reg13, filter_reg33))));

                    //third row
                    output_reg20 = _mm256_add_ps(output_reg20,
                        _mm256_add_ps(
                          _mm256_add_ps(
                            _mm256_mul_ps(input_reg20, filter_reg00),
                            _mm256_mul_ps(input_reg21, filter_reg10)),
                          _mm256_add_ps(
                            _mm256_mul_ps(input_reg22, filter_reg20),
                            _mm256_mul_ps(input_reg23, filter_reg30))));
                    output_reg21 = _mm256_add_ps(output_reg21,
                        _mm256_add_ps(
                          _mm256_add_ps(
                            _mm256_mul_ps(input_reg20, filter_reg01),
                            _mm256_mul_ps(input_reg21, filter_reg11)),
                          _mm256_add_ps(
                            _mm256_mul_ps(input_reg22, filter_reg21),
                            _mm256_mul_ps(input_reg23, filter_reg31))));
                    output_reg22 = _mm256_add_ps(output_reg22,
                        _mm256_add_ps(
                          _mm256_add_ps(
                            _mm256_mul_ps(input_reg20, filter_reg02),
                            _mm256_mul_ps(input_reg21, filter_reg12)),
                          _mm256_add_ps(
                            _mm256_mul_ps(input_reg22, filter_reg22),
                            _mm256_mul_ps(input_reg23, filter_reg32))));
                    output_reg23 = _mm256_add_ps(output_reg23,
                        _mm256_add_ps(
                          _mm256_add_ps(
                            _mm256_mul_ps(input_reg20, filter_reg03),
                            _mm256_mul_ps(input_reg21, filter_reg13)),
                          _mm256_add_ps(
                            _mm256_mul_ps(input_reg22, filter_reg23),
                            _mm256_mul_ps(input_reg23, filter_reg33))));

                    //fourth row
                    output_reg30 = _mm256_add_ps(output_reg30,
                        _mm256_add_ps(
                          _mm256_add_ps(
                            _mm256_mul_ps(input_reg30, filter_reg00),
                            _mm256_mul_ps(input_reg31, filter_reg10)),
                          _mm256_add_ps(
                            _mm256_mul_ps(input_reg32, filter_reg20),
                            _mm256_mul_ps(input_reg33, filter_reg30))));
                    output_reg31 = _mm256_add_ps(output_reg31,
                        _mm256_add_ps(
                          _mm256_add_ps(
                            _mm256_mul_ps(input_reg30, filter_reg01),
                            _mm256_mul_ps(input_reg31, filter_reg11)),
                          _mm256_add_ps(
                            _mm256_mul_ps(input_reg32, filter_reg21),
                            _mm256_mul_ps(input_reg33, filter_reg31))));
                    output_reg32 = _mm256_add_ps(output_reg32,
                        _mm256_add_ps(
                          _mm256_add_ps(
                            _mm256_mul_ps(input_reg30, filter_reg02),
                            _mm256_mul_ps(input_reg31, filter_reg12)),
                          _mm256_add_ps(
                            _mm256_mul_ps(input_reg32, filter_reg22),
                            _mm256_mul_ps(input_reg33, filter_reg32))));
                    output_reg33 = _mm256_add_ps(output_reg33,
                        _mm256_add_ps(
                          _mm256_add_ps(
                            _mm256_mul_ps(input_reg30, filter_reg03),
                            _mm256_mul_ps(input_reg31, filter_reg13)),
                          _mm256_add_ps(
                            _mm256_mul_ps(input_reg32, filter_reg23),
                            _mm256_mul_ps(input_reg33, filter_reg33))));


                    p_input_slice1 += FWB * BB;
                    p_input_slice2 += FWB * BB;
                    p_input_slice3 += FWB * BB;
                    p_input_slice4 += FWB * BB;
                    p_filter_slice += FWB;
                  }
                }
                for (; filter_width_index < filter_width; filter_width_index++) {
                  filter_reg00 = _mm256_broadcast_ss(p_filter_slice);
                  filter_reg01 = _mm256_broadcast_ss(p_filter_slice + filter_output_channel_offset);
                  filter_reg02 = _mm256_broadcast_ss(p_filter_slice + 2 * filter_output_channel_offset);
                  filter_reg03 = _mm256_broadcast_ss(p_filter_slice + 3 * filter_output_channel_offset);
                  input_reg00 = _mm256_load_ps(p_input_slice1);
                  input_reg10 = _mm256_load_ps(p_input_slice2);
                  input_reg20 = _mm256_load_ps(p_input_slice3);
                  input_reg30 = _mm256_load_ps(p_input_slice4);
                  
                  //first row
                  output_reg00 = _mm256_add_ps(output_reg00,
                    _mm256_mul_ps(input_reg00, filter_reg00));
                  output_reg01 = _mm256_add_ps(output_reg01,
                    _mm256_mul_ps(input_reg00, filter_reg01));
                  output_reg02 = _mm256_add_ps(output_reg02,
                    _mm256_mul_ps(input_reg00, filter_reg02));
                  output_reg03 = _mm256_add_ps(output_reg03,
                    _mm256_mul_ps(input_reg00, filter_reg03));

                  //first row
                  output_reg10 = _mm256_add_ps(output_reg10,
                    _mm256_mul_ps(input_reg10, filter_reg00));
                  output_reg11 = _mm256_add_ps(output_reg11,
                    _mm256_mul_ps(input_reg10, filter_reg01));
                  output_reg12 = _mm256_add_ps(output_reg12,
                    _mm256_mul_ps(input_reg10, filter_reg02));
                  output_reg13 = _mm256_add_ps(output_reg13,
                    _mm256_mul_ps(input_reg10, filter_reg03));

                  //first row
                  output_reg20 = _mm256_add_ps(output_reg20,
                    _mm256_mul_ps(input_reg20, filter_reg00));
                  output_reg21 = _mm256_add_ps(output_reg21,
                    _mm256_mul_ps(input_reg20, filter_reg01));
                  output_reg22 = _mm256_add_ps(output_reg22,
                    _mm256_mul_ps(input_reg20, filter_reg02));
                  output_reg23 = _mm256_add_ps(output_reg23,
                    _mm256_mul_ps(input_reg20, filter_reg03));

                  //first row
                  output_reg30 = _mm256_add_ps(output_reg30,
                    _mm256_mul_ps(input_reg30, filter_reg00));
                  output_reg31 = _mm256_add_ps(output_reg31,
                    _mm256_mul_ps(input_reg30, filter_reg01));
                  output_reg32 = _mm256_add_ps(output_reg32,
                    _mm256_mul_ps(input_reg30, filter_reg02));
                  output_reg33 = _mm256_add_ps(output_reg33,
                    _mm256_mul_ps(input_reg30, filter_reg03));

                  p_input_slice1 += BB;
                  p_input_slice2 += BB;
                  p_input_slice3 += BB;
                  p_input_slice4 += BB;
                  p_filter_slice += 1;
                }
                p_input_slice1 += (input_width - filter_width) * BB;
                p_input_slice2 += (input_width - filter_width) * BB;
                p_input_slice3 += (input_width - filter_width) * BB;
                p_input_slice4 += (input_width - filter_width) * BB;
              }

              _mm256_store_ps(p_output, output_reg00);
              _mm256_store_ps(p_output + BB, output_reg10);
              _mm256_store_ps(p_output + BB2, output_reg20);
              _mm256_store_ps(p_output + BB3, output_reg30);
              _mm256_store_ps(p_output + BB_output_channel_offset, output_reg01);
              _mm256_store_ps(p_output + BB_output_channel_offset + BB, output_reg11);
              _mm256_store_ps(p_output + BB_output_channel_offset + BB2, output_reg21);
              _mm256_store_ps(p_output + BB_output_channel_offset + BB3, output_reg31);
              _mm256_store_ps(p_output + 2 * BB_output_channel_offset, output_reg02);
              _mm256_store_ps(p_output + 2 * BB_output_channel_offset + BB, output_reg12);
              _mm256_store_ps(p_output + 2 * BB_output_channel_offset + BB2, output_reg22);
              _mm256_store_ps(p_output + 2 * BB_output_channel_offset + BB3, output_reg32);
              _mm256_store_ps(p_output + 3 * BB_output_channel_offset, output_reg03);
              _mm256_store_ps(p_output + 3 * BB_output_channel_offset + BB, output_reg13);
              _mm256_store_ps(p_output + 3 * BB_output_channel_offset + BB2, output_reg23);
              _mm256_store_ps(p_output + 3 * BB_output_channel_offset + BB3, output_reg33);
              p_output += OWB * BB;
            }
          }
          for (; output_width_index < output_width; ++output_width_index) {
            output_reg00 = _mm256_load_ps(p_output);
            output_reg01 = _mm256_load_ps(p_output + BB_output_channel_offset);
            output_reg02 = _mm256_load_ps(p_output + 2 * BB_output_channel_offset);
            output_reg03 = _mm256_load_ps(p_output + 3 * BB_output_channel_offset);
            const DType* p_input_slice1 = p_input +
              BB * (output_height_index * stride_height * input_width +
              output_width_index * stride_width);
            const DType* p_filter_slice = filter_slice;
            for (int filter_height_index = 0; filter_height_index < filter_height;
                ++filter_height_index) {
              int filter_width_index = 0;
              if (filter_width >= FWB) {
                for (; filter_width_index < filter_width - FWB; filter_width_index += FWB) {
                  filter_reg00 = _mm256_broadcast_ss(p_filter_slice);
                  filter_reg10 = _mm256_broadcast_ss(p_filter_slice + 1);
                  filter_reg20 = _mm256_broadcast_ss(p_filter_slice + 2);
                  filter_reg30 = _mm256_broadcast_ss(p_filter_slice + 3);
                  filter_reg01 = _mm256_broadcast_ss(p_filter_slice + filter_output_channel_offset);
                  filter_reg11 = _mm256_broadcast_ss(p_filter_slice + 1 + filter_output_channel_offset);
                  filter_reg21 = _mm256_broadcast_ss(p_filter_slice + 2 + filter_output_channel_offset);
                  filter_reg31 = _mm256_broadcast_ss(p_filter_slice + 3 + filter_output_channel_offset);
                  filter_reg02 = _mm256_broadcast_ss(p_filter_slice + 2 * filter_output_channel_offset);
                  filter_reg12 = _mm256_broadcast_ss(p_filter_slice + 1 + 2 * filter_output_channel_offset);
                  filter_reg22 = _mm256_broadcast_ss(p_filter_slice + 2 + 2 * filter_output_channel_offset);
                  filter_reg32 = _mm256_broadcast_ss(p_filter_slice + 3 + 2 * filter_output_channel_offset);
                  filter_reg03 = _mm256_broadcast_ss(p_filter_slice + 3 * filter_output_channel_offset);
                  filter_reg13 = _mm256_broadcast_ss(p_filter_slice + 1 + 3 * filter_output_channel_offset);
                  filter_reg23 = _mm256_broadcast_ss(p_filter_slice + 2 + 3 * filter_output_channel_offset);
                  filter_reg33 = _mm256_broadcast_ss(p_filter_slice + 3 + 3 * filter_output_channel_offset);
                  input_reg00 = _mm256_load_ps(p_input_slice1);
                  input_reg01 = _mm256_load_ps(p_input_slice1 + BB);
                  input_reg02 = _mm256_load_ps(p_input_slice1 + BB2);
                  input_reg03 = _mm256_load_ps(p_input_slice1 + BB3);

                  //first row
                  output_reg00 = _mm256_add_ps(output_reg00,
                      _mm256_add_ps(
                        _mm256_add_ps(
                          _mm256_mul_ps(input_reg00, filter_reg00),
                          _mm256_mul_ps(input_reg01, filter_reg10)),
                        _mm256_add_ps(
                          _mm256_mul_ps(input_reg02, filter_reg20),
                          _mm256_mul_ps(input_reg03, filter_reg30))));
                  output_reg01 = _mm256_add_ps(output_reg01,
                      _mm256_add_ps(
                        _mm256_add_ps(
                          _mm256_mul_ps(input_reg00, filter_reg01),
                          _mm256_mul_ps(input_reg01, filter_reg11)),
                        _mm256_add_ps(
                          _mm256_mul_ps(input_reg02, filter_reg21),
                          _mm256_mul_ps(input_reg03, filter_reg31))));
                  output_reg02 = _mm256_add_ps(output_reg02,
                      _mm256_add_ps(
                        _mm256_add_ps(
                          _mm256_mul_ps(input_reg00, filter_reg02),
                          _mm256_mul_ps(input_reg01, filter_reg12)),
                        _mm256_add_ps(
                          _mm256_mul_ps(input_reg02, filter_reg22),
                          _mm256_mul_ps(input_reg03, filter_reg32))));
                  output_reg03 = _mm256_add_ps(output_reg03,
                      _mm256_add_ps(
                        _mm256_add_ps(
                          _mm256_mul_ps(input_reg00, filter_reg03),
                          _mm256_mul_ps(input_reg01, filter_reg13)),
                        _mm256_add_ps(
                          _mm256_mul_ps(input_reg02, filter_reg23),
                          _mm256_mul_ps(input_reg03, filter_reg33))));

                  p_input_slice1 += FWB * BB;
                  p_filter_slice += FWB;
                }
              }
              for (; filter_width_index < filter_width; filter_width_index++) {
                filter_reg00 = _mm256_broadcast_ss(p_filter_slice);
                filter_reg01 = _mm256_broadcast_ss(p_filter_slice + filter_output_channel_offset);
                filter_reg02 = _mm256_broadcast_ss(p_filter_slice + 2 * filter_output_channel_offset);
                filter_reg03 = _mm256_broadcast_ss(p_filter_slice + 3 * filter_output_channel_offset);
                input_reg00 = _mm256_load_ps(p_input_slice1);
                
                //first row
                output_reg00 = _mm256_add_ps(output_reg00,
                  _mm256_mul_ps(input_reg00, filter_reg00));
                output_reg01 = _mm256_add_ps(output_reg01,
                  _mm256_mul_ps(input_reg00, filter_reg01));
                output_reg02 = _mm256_add_ps(output_reg02,
                  _mm256_mul_ps(input_reg00, filter_reg02));
                output_reg03 = _mm256_add_ps(output_reg03,
                  _mm256_mul_ps(input_reg00, filter_reg03));

                p_input_slice1 += BB;
                p_filter_slice += 1;
              }
              p_input_slice1 += (input_width - filter_width) * BB;
            }
            _mm256_store_ps(p_output, output_reg00);
            _mm256_store_ps(p_output + BB_output_channel_offset, output_reg01);
            _mm256_store_ps(p_output + 2 * BB_output_channel_offset, output_reg02);
            _mm256_store_ps(p_output + 3 * BB_output_channel_offset, output_reg03);
            p_output += BB;
          }
        }
      }
    }
  }

  #ifdef BLITZ_PERFORMANCE
  total_end = std::chrono::system_clock::now();
  total_time = total_end - total_start;
  LOG(INFO) << "Forward convolution total: " << total_time.count();
  #endif  // BLITZ_PERFORMANCE
}

}  // namespace blitz

#endif  // SRC_BACKEND_CPU_BACKEND_CONV_INL_H_

