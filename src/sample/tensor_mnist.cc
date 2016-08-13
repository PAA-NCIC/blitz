#include <omp.h>
#include <iostream>

#include "util/common.h"
#include "backend/backends.h"

using namespace blitz;

void forward_conv(
  CPUTensor<float>& input, CPUTensor<float>& filter, 
  vector<shared_ptr<CPUTensor<float> > >& unpacks, CPUTensor<float>& output,
  const bool barrier) {
  // padding height, width
  const int padding_height = 0;
  const int padding_width = 0;
  const int stride_height = 0;
  const int stride_width = 0;
  if (barrier == true) {
    Backend<CPUTensor, float>::Convolution2DForwardFunc(&input, &filter,
      padding_height, padding_width, stride_height, stride_width,
      &unpacks, &output);
  } else {
    // shape decode
    // input
    const Shape& input_shape = input.shape();
    int batch_size = input_shape[0];
    int input_channel = input_shape[1];
    int input_height = input_shape[2];
    int input_width = input_shape[3];
    // filter
    const Shape& filter_shape = filter.shape();
    int filter_height = filter_shape[2];
    int filter_width = filter_shape[3];
    // output
    const Shape& output_shape = output.shape();
    int output_channel = output_shape[1];
    int output_height = output_shape[2];
    int output_width = output_shape[3];

    int batch_input_offset = 0;
    int batch_output_offset = 0;
    int dim_left = output_channel;
    int dim_right = output_height * output_width;
    int dim_common = input_channel * filter_height * filter_width;

    int num_threads = omp_get_num_threads();
    int tid = omp_get_thread_num();
    int thread_idx_start = batch_size / num_threads * tid;
    int thread_idx_end = batch_size / num_threads * (tid + 1);

    for (int batch_index = thread_idx_start; batch_index < thread_idx_end; ++batch_index) {
      // unpack
      // (input_channel) *
      // (input_width * input_height)
      // to
      // (input_channel * filter_height * filter_width)
      // (output_width * output_height)
      Backend<CPUTensor, float>::Unpack2DFunc(
        input.Slice(batch_input_offset),
        input_channel, input_height, input_width,
        filter_height, filter_width, output_height, output_width,
        padding_height, padding_width,
        stride_height, stride_width, unpacks[tid]->data());

      // gemm generate
      // (output_channel) * (output_height * output_width)
      BlitzCPUGemm(false, false, dim_left, dim_right, dim_common,
        filter.data(), unpacks[tid]->data(), output.Slice(batch_output_offset),
        static_cast<float>(1), static_cast<float>(0));

      batch_input_offset += input_channel * input_height * input_width;
      batch_output_offset += output_channel * output_height * output_width;
    }
  }
}

void forward_pool(
  CPUTensor<float>& input, CPUTensor<int>& max_index,
  CPUTensor<float>& output, const bool barrier) {
  const int stride_height = 2;
  const int stride_width = 2;
  const int filter_height = 2;
  const int filter_width = 2;
  if (barrier == true) {
    Backend<CPUTensor, float>::MaxPooling2DForwardFunc(
      &input, filter_height, filter_width,
      stride_width, stride_height, &max_index, &output);
  } else {
    // shape decode
    // input
    const Shape& input_shape = input.shape();
    int batch_size = input_shape[0];
    int input_channel = input_shape[1];
    int input_height = input_shape[2];
    int input_width = input_shape[3];
    // output
    const Shape& output_shape = output.shape();
    int output_channel = output_shape[1];
    int output_height = output_shape[2];
    int output_width = output_shape[3];

    CHECK_EQ(input_channel, output_channel);

    const int input_single_offset = input_channel * input_height * input_width;
    const int input_channel_offset = input_height * input_width;
    const int output_single_offset = output_channel * output_height * output_width;
    const int output_channel_offset = output_height * output_width;
    int channel_input_offset;
    int channel_output_offset;
    int batch_input_offset;
    int batch_output_offset;

    int num_threads = omp_get_num_threads();
    int tid = omp_get_thread_num();
    int thread_idx_start = batch_size / num_threads * tid;
    int thread_idx_end = batch_size / num_threads * (tid + 1);

    for (int batch_index = thread_idx_start; batch_index < thread_idx_end; ++batch_index) {
      batch_input_offset = batch_index * input_single_offset;
      batch_output_offset = batch_index * output_single_offset;
      for (int channel_index = 0; channel_index < input_channel; ++channel_index) {
        channel_input_offset = channel_index * input_channel_offset;
        channel_output_offset = channel_index * output_channel_offset;
        const float* input_slice = input.Slice(batch_input_offset +
          channel_input_offset);
        float* output_slice = output.Slice(batch_output_offset +
          channel_output_offset);
        int* max_index_slice = max_index.Slice(batch_output_offset +
          channel_output_offset);
        for (int output_height_index = 0; output_height_index < output_height;
          ++output_height_index) {
          for (int output_width_index = 0; output_width_index < output_width;
            ++output_width_index) {
            const int height_start = output_height_index * stride_height;
            const int width_start = output_width_index * stride_width;
            const int height_end = height_start + filter_height;
            const int width_end = width_start + filter_width;
            const int pool_index = output_height_index * output_width +
              output_width_index;
            int max_index_tmp = height_start * input_width + width_start;
            for (int h = height_start; h < height_end; ++h) {
              for (int w = width_start; w < width_end; ++w) {
                const int index = h * input_width + w;
                if (input_slice[index] > input_slice[max_index_tmp]) {
                  max_index_tmp = index;
                }
              }
            }
            output_slice[pool_index] = input_slice[max_index_tmp];
            max_index_slice[pool_index] = max_index_tmp;
          }
        }
      }
    }
  }
}

void forward_gemm(
  CPUTensor<float>& input, CPUTensor<float>& weight,
  CPUTensor<float>& output, const bool barrier) {
  if (barrier == true) {
    Backend<CPUTensor, float>::MatrixDotFunc(
      &input, &weight, false, false, 1.0f, 0.0f,
      &output);
  } else {
    const int dim_left = input.shape()[0];
    const int dim_common = input.shape()[1];
    const int dim_right = output.shape()[1];

    int num_threads = omp_get_num_threads();
    int tid = omp_get_thread_num();
    int input_start = (dim_left / num_threads * tid) * dim_common;
    int output_start = (dim_common / num_threads * tid) * dim_right;

    BlitzCPUGemm(false, false, dim_left, dim_right, dim_common,
        input.Slice(input_start), weight.data(), output.Slice(output_start),
        static_cast<float>(1), static_cast<float>(0));
  }
}

void forward_mnist(const int& iteration, const int& thread) {
  std::cout << "configs " << std::endl;

  Shape input_shape(4);
  // batch_size
  input_shape[0] = 128;
  // input channel
  input_shape[1] = 1;
  // input height
  input_shape[2] = 28;
  // input width
  input_shape[3] = 28;

  ///////////////////////
  // convolution1
  // unpack_shape1
  Shape unpack_shape1(2);
  // ic * ih * iw
  unpack_shape1[0] = 1 * 28 * 28;
  // oh * ow
  unpack_shape1[1] = 24 * 24;

  // filter_shape1
  Shape filter_shape1(4);
  // output channel
  filter_shape1[0] = 16;
  // input channel
  filter_shape1[1] = 1;
  // input height
  filter_shape1[2] = 5;
  // input width
  filter_shape1[3] = 5;
  
  // output_shape1
  Shape output_shape1(4);
  // batch_size
  output_shape1[0] = 128;
  // output channel
  output_shape1[1] = 16;
  // output height
  output_shape1[2] = 24;
  // output width
  output_shape1[3] = 24;
  
  ///////////////////////
  // pooling1
  // max_index_shape1
  Shape max_index_shape1(4);
  // batch_size
  max_index_shape1[0] = 128;
  // output channel
  max_index_shape1[1] = 16;
  // output height
  max_index_shape1[2] = 12;
  // output width
  max_index_shape1[3] = 12;

  // output_shape1
  Shape output_shape2(4);
  // batch_size
  output_shape2[0] = 128;
  // output channel
  output_shape2[1] = 16;
  // output height
  output_shape2[2] = 12;
  // output width
  output_shape2[3] = 12;

  ///////////////////////
  // convolution2
  // unpack_shape2
  Shape unpack_shape2(2);
  // ic * ih * iw
  unpack_shape2[0] = 16 * 12 * 12;
  // oh * ow
  unpack_shape2[1] = 8 * 8;

  // filter_shape2
  Shape filter_shape2(4);
  // output channel
  filter_shape2[0] = 32;
  // input channel
  filter_shape2[1] = 16;
  // input height
  filter_shape2[2] = 5;
  // input width
  filter_shape2[3] = 5;
  
  // output_shape3
  Shape output_shape3(4);
  // batch_size
  output_shape3[0] = 128;
  // output channel
  output_shape3[1] = 32;
  // output height
  output_shape3[2] = 8;
  // output width
  output_shape3[3] = 8;

  ///////////////////////
  // pooling2
  // max_index_shape2
  Shape max_index_shape2(4);
  // batch_size
  max_index_shape2[0] = 128;
  // output channel
  max_index_shape2[1] = 32;
  // output height
  max_index_shape2[2] = 4;
  // output width
  max_index_shape2[3] = 4;

  // output_shape3
  Shape output_shape4(4);
  // batch_size
  output_shape4[0] = 128;
  // output channel
  output_shape4[1] = 32;
  // output height
  output_shape4[2] = 4;
  // output width
  output_shape4[3] = 4;

  ///////////////////////
  // Affine1
  // weight_shape1
  Shape weight_shape1(2);
  // num_input
  weight_shape1[0] = 512;
  // num_output
  weight_shape1[1] = 500;

  // output_shape5
  Shape output_shape5(2);
  // batch_size
  output_shape5[0] = 128;
  // num_output
  output_shape5[1] = 500;

  ///////////////////////
  // Affine2
  // weight_shape1
  Shape weight_shape2(2);
  // num_input
  weight_shape2[0] = 500;
  // num_output
  weight_shape2[1] = 10;

  // output_shape6
  Shape output_shape6(2);
  // batch_size
  output_shape6[0] = 128;
  // num_output
  output_shape6[1] = 10;

  std::cout << "allocate " << std::endl;

  CPUTensor<float> input(input_shape);
  Backend<CPUTensor, float>::NormalDistributionFunc(0.0, 1.0, &input);
  // conv1
  CPUTensor<float> filter1(filter_shape1);
  CPUTensor<float> output1(output_shape1);
  Backend<CPUTensor, float>::NormalDistributionFunc(0.0, 1.0, &filter1);
  std::vector<shared_ptr<CPUTensor<float> > > unpacks1;
  for (int i = 0; i < thread; ++i) {
    unpacks1[i] = make_shared<CPUTensor<float> >(unpack_shape1);
  }
  // pool1
  CPUTensor<float> output2(output_shape2);
  CPUTensor<int> max_index1(max_index_shape1);
  // conv2
  CPUTensor<float> filter2(filter_shape2);
  CPUTensor<float> output3(output_shape3);
  Backend<CPUTensor, float>::NormalDistributionFunc(0.0, 1.0, &filter2);
  std::vector<shared_ptr<CPUTensor<float> > > unpacks2;
  for (int i = 0; i < thread; ++i) {
    unpacks2[i] = make_shared<CPUTensor<float> >(unpack_shape2);
  }
  // pool2
  CPUTensor<float> output4(output_shape4);
  CPUTensor<int> max_index2(max_index_shape2);
  // affine1
  CPUTensor<float> weight1(weight_shape1);
  CPUTensor<float> output5(output_shape5);
  Backend<CPUTensor, float>::NormalDistributionFunc(0.0, 1.0, &weight1);
  // affine2
  CPUTensor<float> weight2(weight_shape2);
  CPUTensor<float> output6(output_shape6);
  Backend<CPUTensor, float>::NormalDistributionFunc(0.0, 1.0, &weight2);

  std::cout << "start forward " << std::endl;

  for (int i = 0; i < iteration; ++i) {
      forward_conv(input, filter1, unpacks1, output1, true);
      forward_pool(output1, max_index1, output2, true);
      forward_conv(output2, filter2, unpacks2, output3, true);
      forward_pool(output3, max_index2, output4, true);
      forward_gemm(output4, weight1, output5, true);
      forward_gemm(output5, weight2, output6, true);
  }
  //for (int i = 0; i < iteration; ++i) {
  //  forward_conv(input, filter1, unpack1, output1, false);
  //  forward_pool(output1, max_index1, output2, false);
  //  forward_conv(output2, filter2, unpack2, output3, false);
  //  forward_pool(output3, max_index2, output4, false);
  //  forward_gemm(output4, weight1, output5, false);
  //  forward_gemm(output5, weight2, output6, false);
  //}
}

int main() {
  const int iteration = 1;
  const int thread = 16;
  forward_mnist(iteration, thread);
  return 0;
}
