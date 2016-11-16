#ifndef SRC_BACKENDS_GPU_BACKEND_CONV_INL_H_
#define SRC_BACKENDS_GPU_BACKEND_CONV_INL_H_

template<typename DType>
void Backend<GPUTensor, DType>::Convolution2DForwardFunc(
  const GPUTensor<DType>* input,
  const GPUTensor<DType>* filter,
  GPUTensor<DType>* output,
  GPUTensor<DType>* workspace, 
  size_t padding_height,
  size_t padding_width,
  size_t stride_height,
  size_t stride_width,
  BLITZ_ALGORITHM algorithm) {
  // shape decode
  // input
  const Shape& input_shape = input->shape();
  const size_t batch_size = input_shape[0];
  const size_t input_channel = input_shape[1];
  const size_t input_height = input_shape[2];
  const size_t input_width = input_shape[3];
  // filter
  const Shape& filter_shape = filter->shape();
  const size_t filter_height = filter_shape[2];
  const size_t filter_width = filter_shape[3];
  // output
  const Shape& output_shape = output->shape();
  const size_t output_channel = output_shape[1];
  const size_t output_height = output_shape[2];
  const size_t output_width = output_shape[3];
  // offset
  size_t input_batch_offset = 0;
  size_t output_batch_offset = 0;
  const size_t input_batch_size = input_channel * input_height * input_width;
  const size_t output_batch_size = output_channel * output_height * output_width;
  // dims
  const size_t dim_left = output_channel;
  const size_t dim_right = output_height * output_width;
  const size_t dim_common = input_channel * filter_height * filter_width;
  #ifdef BLITZ_PERFORMANCE  // only valid for a single thread
  cudaEvent_t start, stop;
  float elapsed_time = 0;
  float gemm_time = 0;
  float unpack_time = 0;
  #endif  // BLITZ_PERFORMANCE
	switch (algorithm) {
		case BLITZ_CONVOLUTION_SASS_DIRECT:
		{
			workspace->Fill(0);
			// transpose Input
			BlitzGPUTrans(const_cast<DType*>(input->data()), 
				workspace->data(),
				batch_size,
				input_channel * input_height * input_width);
			// transpose Weight
			BlitzGPUTrans(const_cast<DType*>(filter->data()), 
				workspace->Slice(input->size() + output->size()),
				output_channel,
				input_channel * filter_height * filter_width);
			#ifdef BLITZ_PERFORMANCE
			BLITZ_GPU_TIMER_START(elapsed_time, start, stop);
			#endif
			// direct GEMM
			BlitzSassConvolution2D(
				workspace->data(),
				workspace->Slice(input->size()),
				workspace->Slice(input->size() + output->size()),
				batch_size,
				input_channel,
				input_height, input_width,
				filter_height, filter_width,
				output_channel,
				output_height, output_width,
				stride_height, stride_width,
				padding_height, padding_width,
				"forward");
			#ifdef BLITZ_PERFORMANCE
			BLITZ_GPU_TIMER_END(elapsed_time, start, stop);
			double computations = 2 * (double)batch_size * (double)filter_shape.size() * (double)output_height * (double)output_width;
			BLITZ_GPU_TIMER_INFO(computations, elapsed_time);
			#endif
			// transpose Output
			BlitzGPUTrans(const_cast<DType*>(workspace->Slice(input->size())), 
				output->data(),
				output_channel * output_height * output_width,
				batch_size);
			break;
		}
		case BLITZ_CONVOLUTION_BLAS_GEMM:
		case BLITZ_CONVOLUTION_SASS_GEMM:
			for (size_t batch_index = 0; batch_index < batch_size; ++batch_index) {
				input_batch_offset = batch_index * input_batch_size;
				output_batch_offset = batch_index *  output_batch_size;
				#ifdef BLITZ_PERFORMANCE
				cudaEventRecord(start);
				#endif
				// unpack
				// (input_channel) *
				// (input_width * input_height)
				// to
				// (output_width * output_height)
				// (input_channel * filter_height * filter_width)
				Unpack2DFunc(input->Slice(input_batch_offset),
					workspace->data(),
					input_channel, input_height, input_width,
					filter_height, filter_width,
					output_height, output_width,
					padding_height, padding_width,
					stride_height, stride_width);
				#ifdef BLITZ_PERFORMANCE
				cudaEventRecord(stop);
				cudaEventSynchronize(stop);
				cudaEventElapsedTime(&elapsed_time, start, stop);
				unpack_time += elapsed_time / 1000;
				cudaEventRecord(start);
				#endif
				// gemm generate
				// (output_channel) * (output_height * output_width)
				if (algorithm == BLITZ_CONVOLUTION_BLAS_GEMM) {
					BlitzGPUGemm(const_cast<GPUTensor<DType>*>(filter)->data(),
						workspace->data(),
						output->Slice(output_batch_offset),
						false, true,
						static_cast<DType>(1), static_cast<DType>(0),
						dim_left, dim_right, dim_common);
				} else if (algorithm == BLITZ_CONVOLUTION_SASS_GEMM) {
					BlitzSassGemm(const_cast<GPUTensor<DType>*>(filter)->data(),
						workspace->data(),
						output->Slice(output_batch_offset),
						false, true,
						static_cast<DType>(1), static_cast<DType>(0),
						dim_left, dim_right, dim_common);
				}
				#ifdef BLITZ_PERFORMANCE
				cudaEventRecord(stop);
				cudaEventSynchronize(stop);
				cudaEventElapsedTime(&elapsed_time, start, stop);
				gemm_time += elapsed_time / 1000;
				#endif
			}
			break;
		default:
			LOG(FATAL) << "Unsupported algorithm type: " << algorithm;
			break;
	}
  #ifdef BLITZ_PERFORMANCE
  LOG(INFO) << "Forward convolution compute: " << gemm_time;
  LOG(INFO) << "Forward convolution transform: " << unpack_time;
  #endif  // BLITZ_PERFORMANCE
}

template<typename DType>
void Backend<GPUTensor, DType>::Convolution2DBackwardFunc(
  const GPUTensor<DType>* output,
  const GPUTensor<DType>* filter,
  GPUTensor<DType>* input,
  GPUTensor<DType>* workspace,
  size_t padding_height,
  size_t padding_width,
  size_t stride_height,
  size_t stride_width,
  BLITZ_ALGORITHM algorithm) {
  // shape decode
  // input
  const Shape& input_shape = input->shape();
  const size_t batch_size = input_shape[0];
  const size_t input_channel = input_shape[1];
  const size_t input_height = input_shape[2];
  const size_t input_width = input_shape[3];
  // filter
  const Shape& filter_shape = filter->shape();
  const size_t filter_height = filter_shape[2];
  const size_t filter_width = filter_shape[3];
  // output
  const Shape& output_shape = output->shape();
  const size_t output_channel = output_shape[1];
  const size_t output_height = output_shape[2];
  const size_t output_width = output_shape[3];
  // offset
  size_t input_batch_offset = 0;
  size_t output_batch_offset = 0;
  const size_t input_batch_size = input_channel * input_height * input_width;
  const size_t output_batch_size = output_channel * output_height * output_width;
  // dims
  const size_t dim_left = output_height * output_width;
  const size_t dim_right = input_channel * filter_height * filter_width;
  const size_t dim_common = output_channel;
  // init
  input->Fill(0);
  #ifdef BLITZ_PERFORMANCE
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  float gemm_time = 0;
  float elapsed_time = 0;
  float pack_time = 0;
  #endif  // BLITZ_PERFORMANCE
	switch (algorithm) {
		case BLITZ_CONVOLUTION_SASS_DIRECT:
			workspace->Fill(0);
			// transpose output
			BlitzGPUTrans(const_cast<DType*>(output->data()), 
				workspace->Slice(input->size()),
				batch_size,
				output_channel * output_height * output_width);
			if (input_channel % 64 != 0) {
				// direct GEMM
				BlitzSassConvolution2D(
					workspace->data(),
					const_cast<DType*>(workspace->Slice(input->size())),
					const_cast<DType*>(filter->data()),
					batch_size,
					input_channel,
					input_height, input_width,
					filter_height, filter_width,
					output_channel,
					output_height, output_width,
					stride_height, stride_width,
					padding_height, padding_width,
					"backward");
			} else {
				// transpose filter
				BlitzGPUTrans(const_cast<DType*>(filter->data()), 
					workspace->Slice(input->size() + output->size()),
					output_channel,
					input_channel * filter_height * filter_width);
				// direct GEMM
				BlitzSassConvolution2D(
					workspace->data(),
					const_cast<DType*>(workspace->Slice(input->size())),
					const_cast<DType*>(workspace->Slice(input->size() + output->size())),
					batch_size,
					input_channel,
					input_height, input_width,
					filter_height, filter_width,
					output_channel,
					output_height, output_width,
					stride_height, stride_width,
					padding_height, padding_width,
					"backward");
			}
			// transpose input
			BlitzGPUTrans(const_cast<DType*>(workspace->data()), 
				input->data(), 
				input_channel * input_height * input_width,
				batch_size);
			break;
		case BLITZ_CONVOLUTION_BLAS_GEMM:
		case BLITZ_CONVOLUTION_SASS_GEMM:
			for (size_t batch_index = 0; batch_index < batch_size; ++batch_index) {
				input_batch_offset = batch_index * input_batch_size;
				output_batch_offset = batch_index * output_batch_size;
				#ifdef BLITZ_PERFORMANCE
				cudaEventRecord(start);
				#endif
				// gemm generate
				// (output_width * output_height) *
				// (input_channel * filter_height * filter_width)
				if (algorithm == BLITZ_CONVOLUTION_BLAS_GEMM) {
					BlitzGPUGemm(const_cast<GPUTensor<DType>*>(output)->Slice(output_batch_offset),
						const_cast<GPUTensor<DType>*>(filter)->data(),
						workspace->data(),
						true, false,
						static_cast<DType>(1), static_cast<DType>(0),
						dim_left, dim_right, dim_common);
				} else if (algorithm == BLITZ_CONVOLUTION_SASS_GEMM) {
					BlitzSassGemm(const_cast<GPUTensor<DType>*>(output)->Slice(output_batch_offset),
						const_cast<GPUTensor<DType>*>(filter)->data(),
						workspace->data(),
						true, false,
						static_cast<DType>(1), static_cast<DType>(0),
						dim_left, dim_right, dim_common);
				}
				#ifdef BLITZ_PERFORMANCE
				cudaEventRecord(stop);
				cudaEventSynchronize(stop);
				cudaEventElapsedTime(&elapsed_time, start, stop);
				gemm_time += elapsed_time / 1000.0;
				cudaEventRecord(start);
				#endif
				// pack
				// (output_width * output_height)
				// (input_channel * filter_height * filter_width)
				// to
				// (input_channel) *
				// (input_height * input_width)
				Pack2DFunc(workspace->data(),
					input->Slice(input_batch_offset),
					input_channel, input_height, input_width,
					filter_height, filter_width,
					output_height, output_width,
					padding_height, padding_width,
					stride_height, stride_width);
				#ifdef BLITZ_PERFORMANCE
				cudaEventRecord(stop);
				cudaEventSynchronize(stop);
				cudaEventElapsedTime(&elapsed_time, start, stop);
				pack_time += elapsed_time / 1000.0;
				#endif
			}
			break;
		default:
			LOG(FATAL) << "Unsupported algorithm type: " << algorithm;
			break;
	}
  #ifdef BLITZ_PERFORMANCE
  LOG(INFO) << "Backward convolution compute: " << gemm_time;
  LOG(INFO) << "Backward convolution transform: " << pack_time;
  #endif  // BLITZ_PERFORMANCE
}

template<typename DType>
void Backend<GPUTensor, DType>::Convolution2DUpdateFunc(
  const GPUTensor<DType>* input,
  const GPUTensor<DType>* output,
  GPUTensor<DType>* update,
  GPUTensor<DType>* workspace, 
  size_t padding_height,
  size_t padding_width,
  size_t stride_height,
  size_t stride_width,
  BLITZ_ALGORITHM algorithm) {
  // shape decode
  // input
  const Shape& input_shape = input->shape();
  const size_t batch_size = input_shape[0];
  const size_t input_channel = input_shape[1];
  const size_t input_height = input_shape[2];
  const size_t input_width = input_shape[3];
  // filter
  const Shape& filter_shape = update->shape();
  const size_t filter_height = filter_shape[2];
  const size_t filter_width = filter_shape[3];
  // output
  const Shape& output_shape = output->shape();
  const size_t output_channel = output_shape[1];
  const size_t output_height = output_shape[2];
  const size_t output_width = output_shape[3];
  // offset
  size_t input_batch_offset = 0;
  size_t output_batch_offset = 0;
  const size_t input_batch_size = input_channel * input_height * input_width;
  const size_t output_batch_size = output_channel * output_height * output_width;
  // dims
  const size_t dim_left = output_channel;
  const size_t dim_right = input_channel * filter_height * filter_width;
  const size_t dim_common = output_height * output_width;
  #ifdef BLITZ_PERFORMANCE  // only valid for a single thread
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  float gemm_time = 0;
  float elapsed_time = 0;
  float unpack_time = 0;
  #endif  // BLITZ_PERFORMANCE
	switch (algorithm) {
		case BLITZ_CONVOLUTION_SASS_DIRECT:
			workspace->Fill(0);
			// transpose input
			BlitzGPUTrans(const_cast<DType*>(input->data()), 
				workspace->data(), 
				batch_size,
				input_channel * input_height * input_width);
			// transpose output
			BlitzGPUTrans(const_cast<DType*>(output->data()), 
				workspace->Slice(input->size()), 
				batch_size,
				output_channel * output_height * output_width);
			BlitzSassConvolution2D(
				const_cast<DType*>(workspace->data()),
				const_cast<DType*>(workspace->Slice(input->size())),
				workspace->Slice(input->size() + output->size()),
				batch_size,
				input_channel,
				input_height, input_width,
				filter_height, filter_width,
				output_channel,
				output_height, output_width,
				stride_height, stride_width,
				padding_height, padding_width,
				"update");
			// transpose update
			BlitzGPUTrans(
				const_cast<DType*>(workspace->Slice(input->size() + output->size())),
				update->data(),
				input_channel * filter_height * filter_width,
				output_channel);
			break;
		case BLITZ_CONVOLUTION_SASS_GEMM:
		case BLITZ_CONVOLUTION_BLAS_GEMM:
			for (size_t batch_index = 0; batch_index < batch_size; ++batch_index) {
				input_batch_offset = batch_index * input_batch_size;
				output_batch_offset = batch_index * output_batch_size;
				#ifdef BLITZ_PERFORMANCE
				cudaEventRecord(start);
				#endif
				// unpack
				// (input_channel) *
				// (input_width * input_height)
				// to
				// (output_width * output_height)
				// (input_channel * filter_height * filter_width)
				Unpack2DFunc(input->Slice(input_batch_offset),
					workspace->data(),
					input_channel, input_height, input_width,
					filter_height, filter_width,
					output_height, output_width,
					padding_height, padding_width,
					stride_height, stride_width);
				#ifdef BLITZ_PERFORMANCE
				cudaEventRecord(stop);
				cudaEventSynchronize(stop);
				cudaEventElapsedTime(&elapsed_time, start, stop);
				unpack_time += elapsed_time / 1000;
				cudaEventRecord(start);
				#endif
				// gemm generate
				// (output_channel) *
				// (input_channel * filter_height * filter_width)
				if (algorithm == BLITZ_CONVOLUTION_BLAS_GEMM) {
					BlitzGPUGemm(const_cast<GPUTensor<DType>*>(output)->Slice(output_batch_offset),
						workspace->data(),
						update->data(),
						false, false,
						static_cast<DType>(1), static_cast<DType>(1),
						dim_left, dim_right, dim_common);
				} else if (algorithm == BLITZ_CONVOLUTION_SASS_GEMM) {
					BlitzSassGemm(const_cast<GPUTensor<DType>*>(output)->Slice(output_batch_offset),
						workspace->data(),
						update->data(),
						false, false,
						static_cast<DType>(1), static_cast<DType>(1),
						dim_left, dim_right, dim_common);
				}
				#ifdef BLITZ_PERFORMANCE
				cudaEventRecord(stop);
				cudaEventSynchronize(stop);
				cudaEventElapsedTime(&elapsed_time, start, stop);
				gemm_time += elapsed_time / 1000;
				#endif
			}
			break;
		default:
			LOG(FATAL) << "Unsupported algorithm type: " << algorithm;
			break;
	}
  #ifdef BLITZ_PERFORMANCE
  LOG(INFO) << "Backward convolution filter gemm: " << gemm_time;
  LOG(INFO) << "Backward convolution filter unpack: " << unpack_time;
  #endif  // BLITZ_PERFORMANCE
}

#endif  // SRC_BACKENDS_GPU_BACKEND_CONV_INL_H_
