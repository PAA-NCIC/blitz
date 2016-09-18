#include <omp.h>
#include <iostream>
#include <algorithm>

#include "util/common.h"
#include "util/blitz_cpu_function.h"
#include "backend/backends.h"

using namespace blitz;

void inline forward_rectlin(
  CPUTensor<float>& input, CPUTensor<float>& output, int tid, int num_threads,
  const bool barrier) {
  if (barrier == true) {
    Backend<CPUTensor, float>::RectlinApplyFunc(&input, 0.0, &output);
  } else {
    int batch_start = input.size() / num_threads * tid;
    int batch_end = input.size() / num_threads * (tid + 1);
    for (size_t i = batch_start; i < batch_end; ++i) {
      output[i] = std::max(input[i], float(0)) +
        float(0) * std::min(input[i], float(0));
    }
  }
}

void inline forward_logistic(
  CPUTensor<float>& input, CPUTensor<float>& output, int tid, int num_threads,
  const bool barrier) {
  if (barrier == true) {
    Backend<CPUTensor, float>::LogisticApplyFunc(&input, &output);
  } else {
    int batch_start = input.size() / num_threads * tid;
    int batch_end = input.size() / num_threads * (tid + 1);
    for (size_t i = batch_start; i < batch_end; ++i) {
      output[i] = 1 / (exp(-input[i]) + 1);
    }
  }
}

void inline forward_conv(
  CPUTensor<float>& input, CPUTensor<float>& filter, 
  vector<shared_ptr<CPUTensor<float> > >& unpacks, CPUTensor<float>& output,
  int tid, int num_threads, const bool barrier) {
  // padding height, width
  const int padding_height = 0;
  const int padding_width = 0;
  const int stride_height = 1;
  const int stride_width = 1;
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

    int dim_left = output_channel;
    int dim_right = output_height * output_width;
    int dim_common = input_channel * filter_height * filter_width;

    int thread_idx_start = batch_size / num_threads * tid;
    int thread_idx_end = batch_size / num_threads * (tid + 1);

    int batch_input_offset = thread_idx_start * input_channel * input_height * input_width;
    int batch_output_offset = thread_idx_start * output_channel * output_height * output_width;

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

void inline forward_pool(
  CPUTensor<float>& input, CPUTensor<size_t>& max_index,
  CPUTensor<float>& output, int tid, int num_threads, const bool barrier) {
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
        size_t* max_index_slice = max_index.Slice(batch_output_offset +
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

void inline forward_gemm(
  CPUTensor<float>& input, CPUTensor<float>& weight,
  CPUTensor<float>& output, int tid, int num_threads,
  const bool barrier) {
  if (barrier == true) {
    Backend<CPUTensor, float>::MatrixDotFunc(
      &input, &weight, false, false, 1.0f, 0.0f,
      &output);
  } else {
    const int dim_left = input.shape()[0];
    const int dim_common = weight.shape()[0];
    const int dim_right = weight.shape()[1];

    //std::cout << "dim_left " << dim_left << std::endl;
    //std::cout << "dim_common " << dim_common << std::endl;
    //std::cout << "dim_right " << dim_right << std::endl;

    int input_start = (dim_left / num_threads * tid) * dim_common;
    int output_start = (dim_left / num_threads * tid) * dim_right;

    BlitzCPUGemm(false, false, dim_left / num_threads, dim_right, dim_common,
        input.Slice(input_start), weight.data(), output.Slice(output_start),
        static_cast<float>(1), static_cast<float>(0));
  }
}

void forward_mnist(const int batch_size, const int iteration, const int thread) {
  std::cout << "configs " << std::endl;

  // mnist
  Shape input_shape(4);
  // batch_size
  input_shape[0] = batch_size;
  // input channel
  input_shape[1] = 3;
  // input height
  input_shape[2] = 64;
  // input width
  input_shape[3] = 64;

  ///////////////////////
  // convolution1
  // unpack_shape1
  Shape unpack_shape1(2);
  // ic * ih * iw
  unpack_shape1[0] = 3 * 64 * 64;
  // oh * ow
  unpack_shape1[1] = 60 * 60;

  // filter_shape1
  Shape filter_shape1(4);
  // output channel
  filter_shape1[0] = 64;
  // input channel
  filter_shape1[1] = 3;
  // filter height
  filter_shape1[2] = 5;
  // filter width
  filter_shape1[3] = 5;
  
  // output_shape1
  Shape output_shape1(4);
  // batch_size
  output_shape1[0] = batch_size;
  // output channel
  output_shape1[1] = 64;
  // output height
  output_shape1[2] = 60;
  // output width
  output_shape1[3] = 60;
  
  ///////////////////////
  // pooling1
  // max_index_shape1
  Shape max_index_shape1(4);
  // batch_size
  max_index_shape1[0] = batch_size;
  // output channel
  max_index_shape1[1] = 64;
  // output height
  max_index_shape1[2] = 30;
  // output width
  max_index_shape1[3] = 30;

  // output_shape1
  Shape output_shape2(4);
  // batch_size
  output_shape2[0] = batch_size;
  // output channel
  output_shape2[1] = 64;
  // output height
  output_shape2[2] = 30;
  // output width
  output_shape2[3] = 30;

  ///////////////////////
  // convolution2
  // unpack_shape2
  Shape unpack_shape2(2);
  // ic * ih * iw
  unpack_shape2[0] = 64 * 30 * 30;
  // oh * ow
  unpack_shape2[1] = 26 * 26;

  // filter_shape2
  Shape filter_shape2(4);
  // output channel
  filter_shape2[0] = 192;
  // input channel
  filter_shape2[1] = 64;
  // filter height
  filter_shape2[2] = 5;
  // filter width
  filter_shape2[3] = 5;
  
  // output_shape3
  Shape output_shape3(4);
  // batch_size
  output_shape3[0] = batch_size;
  // output channel
  output_shape3[1] = 192;
  // output height
  output_shape3[2] = 26;
  // output width
  output_shape3[3] = 26;

  ///////////////////////
  // pooling2
  // max_index_shape2
  Shape max_index_shape2(4);
  // batch_size
  max_index_shape2[0] = batch_size;
  // output channel
  max_index_shape2[1] = 192;
  // output height
  max_index_shape2[2] = 13;
  // output width
  max_index_shape2[3] = 13;

  // output_shape3
  Shape output_shape4(4);
  // batch_size
  output_shape4[0] = batch_size;
  // output channel
  output_shape4[1] = 192;
  // output height
  output_shape4[2] = 13;
  // output width
  output_shape4[3] = 13;

  ///////////////////////
  // Affine1
  // weight_shape1
  Shape weight_shape1(2);
  // num_input
  weight_shape1[0] = 800;
  // num_output
  weight_shape1[1] = 500;

  // output_shape5
  Shape output_shape5(2);
  // batch_size
  output_shape5[0] = batch_size;
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
  output_shape6[0] = batch_size;
  // num_output
  output_shape6[1] = 10 * 10;

  std::cout << "allocate " << std::endl;

  std::vector<shared_ptr<CPUTensor<float> > > inputs(iteration);
  for (int i = 0; i < iteration; ++i) {
    inputs[i] = make_shared<CPUTensor<float> >(input_shape);
    Backend<CPUTensor, float>::NormalDistributionFunc(0.0, 1.0, inputs[i].get());
  }
  // conv1
  CPUTensor<float> filter1(filter_shape1);
  CPUTensor<float> output1(output_shape1);
  Backend<CPUTensor, float>::NormalDistributionFunc(0.0, 1.0, &filter1);
  std::vector<shared_ptr<CPUTensor<float> > > unpacks1(thread);
  for (int i = 0; i < thread; ++i) {
    unpacks1[i] = make_shared<CPUTensor<float> >(unpack_shape1);
  }
  // pool1
  CPUTensor<float> output2(output_shape2);
  CPUTensor<size_t> max_index1(max_index_shape1);
  // conv2
  CPUTensor<float> filter2(filter_shape2);
  CPUTensor<float> output3(output_shape3);
  Backend<CPUTensor, float>::NormalDistributionFunc(0.0, 1.0, &filter2);
  std::vector<shared_ptr<CPUTensor<float> > > unpacks2(thread);
  for (int i = 0; i < thread; ++i) {
    unpacks2[i] = make_shared<CPUTensor<float> >(unpack_shape2);
  }
  // pool2
  CPUTensor<float> output4(output_shape4);
  CPUTensor<size_t> max_index2(max_index_shape2);
  // affine1
  CPUTensor<float> weight1(weight_shape1);
  CPUTensor<float> output5(output_shape5);
  Backend<CPUTensor, float>::NormalDistributionFunc(0.0, 1.0, &weight1);
  // affine2
  CPUTensor<float> weight2(weight_shape2);
  CPUTensor<float> output6(output_shape6);
  Backend<CPUTensor, float>::NormalDistributionFunc(0.0, 1.0, &weight2);

  std::cout << "start forward " << std::endl;
  timeval t1, t2;
  double elapsed_time = 0.0f;
  gettimeofday(&t1, NULL);

  //for (int i = 0; i < iteration; ++i) {
  //    shared_ptr<CPUTensor<float> > input = inputs[i];
  //    forward_conv(*input, filter1, unpacks1, output1, 0, 0, true);
  //    forward_rectlin(output1, output1, 0, 0, true);
  //    forward_pool(output1, max_index1, output2, 0, 0, true);
  //    forward_conv(output2, filter2, unpacks2, output3, 0, 0, true);
  //    forward_rectlin(output2, output2, 0, 0, true);
  //    forward_pool(output3, max_index2, output4, 0, 0, true);
  //    //forward_gemm(output4, weight1, output5, 0, 0, true);
  //    //forward_gemm(output5, weight2, output6, 0, 0, true);
  //}

  //CPUTensor<float> output_compare(output_shape6);
  //Backend<CPUTensor, float>::HostCopyToFunc(output6.data(), output6.size(),
  //  output_compare.data());

  #pragma omp parallel
  {
    int num_threads = omp_get_num_threads();
    int tid = omp_get_thread_num();
    for (int i = 0; i < iteration; ++i) {
      shared_ptr<CPUTensor<float> > input = inputs[i];
      //std::cout << " conv1 " << std::endl;
      forward_conv(*input, filter1, unpacks1, output1, tid, num_threads, false);
      forward_rectlin(output1, output1, tid, num_threads, false);
      ////std::cout << " pool1 " << std::endl;
      forward_pool(output1, max_index1, output2, tid, num_threads, false);
      //std::cout << " conv2 " << std::endl;
      forward_conv(output2, filter2, unpacks2, output3, tid, num_threads, false);
      forward_rectlin(output2, output2, tid, num_threads, false);
      ////std::cout << " pool2 " << std::endl;
      forward_pool(output3, max_index2, output4, tid, num_threads, false);
      //////std::cout << " affine1 " << std::endl;
      //forward_gemm(output4, weight1, output5, tid, num_threads, false);
      //////std::cout << " affine2 " << std::endl;
      //forward_gemm(output5, weight2, output6, tid, num_threads, false);
    }
  }

  //for (int i = 0; i < output6.size(); ++i) {
  //  if (output6[i] != output_compare[i]) {
  //    std::cout << "here " << output6[i] << " " << output_compare[i] << std::endl;
  //  }
  //}

  gettimeofday(&t2, NULL);
  elapsed_time = (t2.tv_sec - t1.tv_sec) * 1000.0; 
  elapsed_time += (t2.tv_usec - t1.tv_usec) / 1000.0;
  elapsed_time /= 1000.0;
  std::cout << "total time: " << elapsed_time << std::endl;
}

int main() {
  const int iteration = 10;
  const int thread = 16;
  const int batch_size = 16 * 8;
  forward_mnist(batch_size, iteration, thread);
  return 0;
}
