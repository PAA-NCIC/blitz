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
  cudaEvent_t start, stop;
  float elapsed_time = 0;
  float compute_time = 0;
  float transform_time = 0;
	switch (algorithm) {
		case BLITZ_CONVOLUTION_SASS_DIRECT: {
			BLITZ_GPU_TIMER_START(elapsed_time, start, stop);
			workspace->Fill(0);
			// transpose Input
			BlitzGPUTrans(const_cast<DType*>(input->data()), 
				workspace->data(),
				NIN, CHW);
			// transpose Weight
			BlitzGPUTrans(const_cast<DType*>(filter->data()), 
				workspace->Slice(input->size() + output->size()),
				K, CRS);
			BLITZ_GPU_TIMER_END(elapsed_time, start, stop);
			transform_time += elapsed_time;
			BLITZ_GPU_TIMER_START(elapsed_time, start, stop);
			// direct GEMM
			BlitzSassConvolution2D(
				workspace->data(),
				workspace->Slice(input->size()),
				workspace->Slice(input->size() + output->size()),
				NIN,
				C, H, W,
				R, S,
				K, P, Q,
				stride_height, stride_width,
				padding_height, padding_width,
				"forward");
			BLITZ_GPU_TIMER_END(elapsed_time, start, stop);
			compute_time = elapsed_time;
			BLITZ_GPU_TIMER_START(elapsed_time, start, stop);
			// transpose Output
			BlitzGPUTrans(const_cast<DType*>(workspace->Slice(input->size())), 
				output->data(),
				KPQ, NIN);
			BLITZ_GPU_TIMER_END(elapsed_time, start, stop);
			transform_time += elapsed_time;
			break;
		}
		case BLITZ_CONVOLUTION_BLAS_GEMM:
		case BLITZ_CONVOLUTION_SASS_GEMM: {
			for (size_t n = 0; n < NIN; ++n) {
				nCHW = n * CHW;
				nKPQ = n * KPQ;
				BLITZ_GPU_TIMER_START(elapsed_time, start, stop);
				// unpack
				// (input_channel) *
				// (input_width * input_height)
				// to
				// (output_width * output_height)
				// (input_channel * filter_height * filter_width)
				Unpack2DFunc(input->Slice(nCHW),
					workspace->data(),
					C, H, W,
					R, S,
					P, Q,
					padding_height, padding_width,
					stride_height, stride_width);
				BLITZ_GPU_TIMER_END(elapsed_time, start, stop);
				transform_time += elapsed_time;
				BLITZ_GPU_TIMER_START(elapsed_time, start, stop);
				// gemm generate
				// (output_channel) * (output_height * output_width)
				if (algorithm == BLITZ_CONVOLUTION_BLAS_GEMM) {
					BlitzGPUGemm(const_cast<GPUTensor<DType>*>(filter)->data(),
						workspace->data(),
						output->Slice(nKPQ),
						false, true,
						static_cast<DType>(1), static_cast<DType>(0),
						K, PQ, CRS);
				} else if (algorithm == BLITZ_CONVOLUTION_SASS_GEMM) {
					BlitzSassGemm(const_cast<GPUTensor<DType>*>(filter)->data(),
						workspace->data(),
						output->Slice(nKPQ),
						false, true,
						static_cast<DType>(1), static_cast<DType>(0),
						K, PQ, CRS);
				}
				BLITZ_GPU_TIMER_END(elapsed_time, start, stop);
				compute_time += elapsed_time;
			}
			break;
		}		
		default:
			LOG(FATAL) << "Unsupported algorithm type: " << algorithm;
			break;
	}
  #ifdef BLITZ_PERFORMANCE
	double computations = static_cast<double>(KPQ) * static_cast<double>(CRS) * static_cast<double>(2 * NIN);
  LOG(INFO) << "Forward convolution compute: " << compute_time;
  LOG(INFO) << "Forward convolution transform: " << transform_time;
  LOG(INFO) << "Forward convolution compute gflops: " << computations / (compute_time * 1e9);
  LOG(INFO) << "Forward convolution total gflops: " << computations / ((transform_time + compute_time) * 1e9);
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
  // init
  input->Fill(0);
	// time counter
  cudaEvent_t start, stop;
  float elapsed_time = 0;
  float compute_time = 0;
  float transform_time = 0;
	switch (algorithm) {
		case BLITZ_CONVOLUTION_SASS_DIRECT: {
			BLITZ_GPU_TIMER_START(elapsed_time, start, stop);
			workspace->Fill(0);
			// transpose output
			BlitzGPUTrans(const_cast<DType*>(output->data()), 
				workspace->Slice(input->size()),
				NIN, KPQ);
			if (C % 64 != 0) {
				BLITZ_GPU_TIMER_END(elapsed_time, start, stop);
				transform_time = elapsed_time;
				BLITZ_GPU_TIMER_START(elapsed_time, start, stop);
				// direct GEMM
				BlitzSassConvolution2D(
					workspace->data(),
					const_cast<DType*>(workspace->Slice(input->size())),
					const_cast<DType*>(filter->data()),
					NIN,
					C, H, W,
					R, S,
					K, P, Q,
					stride_height, stride_width,
					padding_height, padding_width,
					"backward");
				BLITZ_GPU_TIMER_END(elapsed_time, start, stop);
				compute_time = elapsed_time;
			} else {
				// transpose filter
				BlitzGPUTrans(const_cast<DType*>(filter->data()), 
					workspace->Slice(input->size() + output->size()),
					K, CRS);
				BLITZ_GPU_TIMER_END(elapsed_time, start, stop);
				transform_time = elapsed_time;
				BLITZ_GPU_TIMER_START(elapsed_time, start, stop);
				// direct GEMM
				BlitzSassConvolution2D(
					workspace->data(),
					const_cast<DType*>(workspace->Slice(input->size())),
					const_cast<DType*>(workspace->Slice(input->size() + output->size())),
					NIN,
					C, H, W,
					R, S,
					K, P, Q,
					stride_height, stride_width,
					padding_height, padding_width,
					"backward");
				BLITZ_GPU_TIMER_END(elapsed_time, start, stop);
				compute_time = elapsed_time;
			}
			BLITZ_GPU_TIMER_START(elapsed_time, start, stop);
			// transpose input
			BlitzGPUTrans(const_cast<DType*>(workspace->data()), 
				input->data(), 
				CRS, NIN);
			BLITZ_GPU_TIMER_END(elapsed_time, start, stop);
			transform_time += elapsed_time;
			break;
		}
		case BLITZ_CONVOLUTION_SASS_GEMM:
		case BLITZ_CONVOLUTION_BLAS_GEMM: {
			for (size_t n = 0; n < NIN; ++n) {
				nCHW = n * CHW;
				nKPQ = n * KPQ;
				BLITZ_GPU_TIMER_START(elapsed_time, start, stop);
				// gemm generate
				// (output_width * output_height) *
				// (input_channel * filter_height * filter_width)
				if (algorithm == BLITZ_CONVOLUTION_BLAS_GEMM) {
					BlitzGPUGemm(const_cast<GPUTensor<DType>*>(output)->Slice(nKPQ),
						const_cast<GPUTensor<DType>*>(filter)->data(),
						workspace->data(),
						true, false,
						static_cast<DType>(1), static_cast<DType>(0),
						PQ, CRS, K);
				} else if (algorithm == BLITZ_CONVOLUTION_SASS_GEMM) {
					BlitzSassGemm(const_cast<GPUTensor<DType>*>(output)->Slice(nKPQ),
						const_cast<GPUTensor<DType>*>(filter)->data(),
						workspace->data(),
						true, false,
						static_cast<DType>(1), static_cast<DType>(0),
						PQ, CRS, K);
				}
				BLITZ_GPU_TIMER_END(elapsed_time, start, stop);
				compute_time += elapsed_time;
				BLITZ_GPU_TIMER_START(elapsed_time, start, stop);
				// pack
				// (output_width * output_height)
				// (input_channel * filter_height * filter_width)
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
				BLITZ_GPU_TIMER_END(elapsed_time, start, stop);
				transform_time += elapsed_time;
			}
			break;
		}
		default:
			LOG(FATAL) << "Unsupported algorithm type: " << algorithm;
			break;
	}
  #ifdef BLITZ_PERFORMANCE
  LOG(INFO) << "Backward convolution compute: " << compute_time;
  LOG(INFO) << "Backward convolution transform: " << transform_time;
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
	size_t NIN, C, H, W;
	size_t KF, CF, R, S;
	size_t NOUT, K, P, Q;
	Blitz2DBuffer(input->data_layout(), input->shape_ptr(), &NIN, &C, &H, &W);
	Blitz2DFilter(update->data_layout(), update->shape_ptr(), &KF, &CF, &R, &S);
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
  cudaEvent_t start, stop;
  float elapsed_time = 0;
  float compute_time = 0;
  float transform_time = 0;
	switch (algorithm) {
		case BLITZ_CONVOLUTION_SASS_DIRECT: {
			BLITZ_GPU_TIMER_START(elapsed_time, start, stop);
			workspace->Fill(0);
			// transpose input
			BlitzGPUTrans(const_cast<DType*>(input->data()), 
				workspace->data(), 
				NIN, CRS);
			// transpose output
			BlitzGPUTrans(const_cast<DType*>(output->data()), 
				workspace->Slice(input->size()), 
				NIN, KPQ);
			BLITZ_GPU_TIMER_END(elapsed_time, start, stop);
			transform_time = elapsed_time;
			BLITZ_GPU_TIMER_START(elapsed_time, start, stop);
			BlitzSassConvolution2D(
				const_cast<DType*>(workspace->data()),
				const_cast<DType*>(workspace->Slice(input->size())),
				workspace->Slice(input->size() + output->size()),
				NIN,
				C, H, W,
				R, S,
				K, P, Q,
				stride_height, stride_width,
				padding_height, padding_width,
				"update");
			BLITZ_GPU_TIMER_END(elapsed_time, start, stop);
			compute_time = elapsed_time;
			BLITZ_GPU_TIMER_START(elapsed_time, start, stop);
			// transpose update
			BlitzGPUTrans(
				const_cast<DType*>(workspace->Slice(input->size() + output->size())),
				update->data(),
				CRS, K);
			BLITZ_GPU_TIMER_END(elapsed_time, start, stop);
			transform_time += elapsed_time;
			break;
		}
		case BLITZ_CONVOLUTION_SASS_GEMM:
		case BLITZ_CONVOLUTION_BLAS_GEMM: {
			for (size_t n = 0; n < NIN; ++n) {
				nCHW = n * CHW;
				nKPQ = n * KPQ;
				BLITZ_GPU_TIMER_START(elapsed_time, start, stop);
				// unpack
				// (input_channel) *
				// (input_width * input_height)
				// to
				// (output_width * output_height)
				// (input_channel * filter_height * filter_width)
				Unpack2DFunc(input->Slice(nCHW),
					workspace->data(),
					C, H, W,
					R, S,
					P, Q,
					padding_height, padding_width,
					stride_height, stride_width);
				BLITZ_GPU_TIMER_END(elapsed_time, start, stop);
				transform_time += elapsed_time;
				BLITZ_GPU_TIMER_START(elapsed_time, start, stop);
				// gemm generate
				// (output_channel) *
				// (input_channel * filter_height * filter_width)
				if (algorithm == BLITZ_CONVOLUTION_BLAS_GEMM) {
					BlitzGPUGemm(const_cast<GPUTensor<DType>*>(output)->Slice(nKPQ),
						workspace->data(),
						update->data(),
						false, false,
						static_cast<DType>(1), static_cast<DType>(1),
						K, CRS, PQ);
				} else if (algorithm == BLITZ_CONVOLUTION_SASS_GEMM) {
					BlitzSassGemm(const_cast<GPUTensor<DType>*>(output)->Slice(nKPQ),
						workspace->data(),
						update->data(),
						false, false,
						static_cast<DType>(1), static_cast<DType>(1),
						K, CRS, PQ);
				}
				BLITZ_GPU_TIMER_END(elapsed_time, start, stop);
				compute_time += elapsed_time;
				BLITZ_GPU_TIMER_START(elapsed_time, start, stop);
			}
			break;
		}
		default:
			LOG(FATAL) << "Unsupported algorithm type: " << algorithm;
			break;
	}
  #ifdef BLITZ_PERFORMANCE
  LOG(INFO) << "Backward convolution filter gemm: " << compute_time;
  LOG(INFO) << "Backward convolution filter unpack: " << transform_time;
  #endif  // BLITZ_PERFORMANCE
}

#endif  // SRC_BACKENDS_GPU_BACKEND_CONV_INL_H_
