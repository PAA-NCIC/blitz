#ifndef SRC_BACKENDS_MIC_BACKEND_POOL_INL_H_
#define SRC_BACKENDS_MIC_BACKEND_POOL_INL_H_

template<typename DType>
void MaxPoolingForwardNCHWImpl(
	const DType* I,
	DType* O,
	size_t* max_index,
	size_t N,
	size_t C, size_t H, size_t W,
	size_t K, size_t P, size_t Q,
	size_t R, size_t S,
	size_t str_h, size_t str_w) {
  // offset
  const size_t HW = H * W;
  const size_t CHW = C * HW;
  const size_t PQ = P * Q;
  const size_t KPQ = K * PQ;
  #pragma omp parallel for
  for (size_t n = 0; n < N; ++n) {
    for (size_t c = 0; c < C; ++c) {
      const DType* input_slice = I + n * CHW + c * HW;
      DType* output_slice = O + n * KPQ + c * PQ;
      size_t* max_index_slice = max_index + n * KPQ + c * PQ;
      for (size_t oh = 0; oh < P; ++oh) {
        for (size_t ow = 0; ow < Q; ++ow) {
          size_t hs = oh * str_h;
          size_t ws = ow * str_w;
          size_t he = hs + R;
          size_t we = ws + S;
          size_t pool_index = oh * Q + ow;
          max_index_slice[pool_index] = hs * W + ws;
          for (size_t h = hs; h < he; ++h) {
            for (size_t w = ws; w < we; ++w) {
              size_t index = h * W + w;
              if (input_slice[index] > input_slice[max_index_slice[pool_index]]) {
                max_index_slice[pool_index] = index;
              }
            }
          }
          output_slice[pool_index] = input_slice[max_index_slice[pool_index]];
        }
      }
    }
  }
}

template<typename DType>
void MaxPoolingForwardNHWCImpl(
	const DType* I,
	DType* O,
	size_t* max_index,
	size_t N,
	size_t C, size_t H, size_t W,
	size_t K, size_t P, size_t Q,
	size_t R, size_t S,
	size_t str_h, size_t str_w) {
  const size_t HWC = H * W * C;
  const size_t PQK = P * Q * K;
  #pragma omp parallel for
  for (size_t n = 0; n < N; ++n) {
		const DType* input_slice = I + n * HWC;
		DType* output_slice = O + n * PQK;
		size_t* max_index_slice = max_index + n * PQK;
		for (size_t oh = 0; oh < P; ++oh) {
			for (size_t ow = 0; ow < Q; ++ow) {
				const size_t hs = oh * str_h;
				const size_t ws = ow * str_w;
				const size_t he = hs + R;
				const size_t we = ws + S;
				const size_t pool_index = (oh * Q + ow) * C;
				for (size_t c = 0; c < C; ++c) {
					max_index_slice[pool_index + c] = (hs * W + ws) * C + c;
				}
				for (size_t h = hs; h < he; ++h) {
					for (size_t w = ws; w < we; ++w) {
						for (size_t c = 0; c < C; ++c) {
							size_t index = (h * W + w) * C + c;
							if (input_slice[index] > input_slice[max_index_slice[pool_index + c]]) {
								max_index_slice[pool_index + c] = index;
							}
						}
					}
				}
				for (size_t c = 0; c < C; ++c) {
					output_slice[pool_index + c] = input_slice[max_index_slice[pool_index + c]];
				}
			}
		}
  }
}

template<typename DType>
void Backend<MICTensor, DType>::MaxPooling2DForwardFunc(
  const MICTensor<DType>* input,
  MICTensor<DType>* output,
  MICTensor<size_t>* max_index, 
  size_t filter_height,
  size_t filter_width,
  size_t stride_width,
  size_t stride_height) {
	// shape init
	size_t IN, C, H, W;
	size_t ON, K, P, Q;
  // shape decode
  CHECK_EQ(input->data_layout(), output->data_layout());
	BlitzPooling2DShape(input->data_layout(), input->shape_ptr(), &IN, &C, &H, &W);
	BlitzPooling2DShape(output->data_layout(), output->shape_ptr(), &ON, &K, &P, &Q);
  CHECK_EQ(IN, ON);
  CHECK_EQ(C, K);
	switch (input->data_layout()) {
		case BLITZ_POOLING_BUFFER_NCHW:
			MaxPoolingForwardNCHWImpl(
				input->data(),
				output->data(),
				max_index->data(),
				IN,
				C, H, W,
				K, P, Q,
				filter_height, filter_width,
				stride_height, stride_width);
			break;
		case BLITZ_POOLING_BUFFER_NHWC:
			MaxPoolingForwardNHWCImpl(
				input->data(),
				output->data(),
				max_index->data(),
				IN,
				C, H, W,
				K, P, Q,
				filter_height, filter_width,
				stride_height, stride_width);
			break;
		default:
			LOG(FATAL) << "Blitz not support pooling format: " << input->data_layout(); 
			break;
	}	
}

template<typename DType>
void MaxPoolingBackwardNCHWImpl(
	const DType* O,
	DType* I,
	const size_t* max_index,
	size_t N,
	size_t C, size_t H, size_t W,
	size_t K, size_t P, size_t Q) {
	const size_t HW = H * W;
  const size_t CHW = C * HW;
	const size_t PQ = P * Q;
  const size_t KPQ = K * PQ;
  #pragma omp parallel for
	for (size_t n = 0; n < N; ++n) {
    for (size_t c = 0; c < C; ++c) {
      DType* input_slice = I + n * CHW + c * HW;
      const DType* output_slice = O + n * KPQ + c * PQ;
      const size_t* max_index_slice = max_index + n * KPQ + c * PQ;
      for (size_t oh = 0; oh < P; ++oh) {
        for (size_t ow = 0; ow < Q; ++ow) {
          input_slice[max_index_slice[oh * Q + ow]] = output_slice[oh * Q + ow];
        }
      }
    }
  }
}

template<typename DType>
void MaxPoolingBackwardNHWCImpl(
	const DType* O,
	DType* I,
	const size_t* max_index,
	size_t N,
	size_t C, size_t H, size_t W,
	size_t K, size_t P, size_t Q) {
  const size_t CHW = C * H * W;
  const size_t KPQ = K * P * Q;
  #pragma omp parallel for
	for (size_t n = 0; n < N; ++n) {
		DType* input_slice = I + n * CHW;
		const DType* output_slice = O + n * KPQ;
		const size_t* max_index_slice = max_index + n * KPQ;
		for (size_t oh = 0; oh < P; ++oh) {
			for (size_t ow = 0; ow < Q; ++ow) {
				for (size_t c = 0; c < C; ++c) {
					input_slice[max_index_slice[(oh * Q + ow) * C + c]] = output_slice[(oh * Q + ow) * C + c];
				}
			}
		}
	}
}

template<typename DType>
void Backend<MICTensor, DType>::MaxPooling2DBackwardFunc(
  const MICTensor<DType>* output,
  MICTensor<DType>* input,
  const MICTensor<size_t>* max_index,
  size_t filter_height,
  size_t filter_width,
  size_t stride_height,
  size_t stride_width) {
	// shape init
	size_t IN, C, H, W;
	size_t ON, K, P, Q;
  // shape decode
  CHECK_EQ(input->data_layout(), output->data_layout());
	BlitzPooling2DShape(input->data_layout(), input->shape_ptr(), &IN, &C, &H, &W);
	BlitzPooling2DShape(output->data_layout(), output->shape_ptr(), &ON, &K, &P, &Q);
  CHECK_EQ(IN, ON);
  CHECK_EQ(C, K);
  // set zero
  input->Fill(0);
  // no padding
	switch (input->data_layout()) {
		case BLITZ_POOLING_BUFFER_NCHW:
			MaxPoolingBackwardNCHWImpl(
				output->data(),
				input->data(),
				max_index->data(),
				IN,
				C, H, W,
				K, P, Q);
			break;
		case BLITZ_POOLING_BUFFER_NHWC:
			MaxPoolingBackwardNHWCImpl(
				output->data(),
				input->data(),
				max_index->data(),
				IN,
				C, H, W,
				K, P, Q);
			break;
		default:
			LOG(FATAL) << "Blitz not support pooling format: " << input->data_layout();
			break;
	}
}

#endif  // SRC_BACKENDS_MIC_BACKEND_POOL_INL_H_
