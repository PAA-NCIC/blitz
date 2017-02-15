#Convolution

This document shows how to use blitz convolution apis. With input parameters, only few setups are needed to run convolution in either forward or backward phase.

### CPU Example

A more comphrensive example is in `samples/cpu/convolutoin/convolution.cc`.

```C++
// set up data layouts
Shape input_shape(4, BLITZ_BUFFER_NCHW);
Shape filter_shape(4, BLITZ_FILTER_KCRS);
Shape output_shape(4, BLITZ_BUFFER_NCHW);

// init shapes
input_shape[0] = N; input_shape[1] = C; input_shape[2] = H; input_shape[3] = W;
output_shape[0] = N; output_shape[1] = K; output_shape[2] = P; output_shape[3] = Q;
filter_shape[0] = K; filter_shape[1] = C; filter_shape[2] = R; filter_shape[3] = S;

// init tensors
CPUTensor<float> input(input_shape);
CPUTensor<float> filter(filter_shape);
CPUTensor<float> output(output_shape);

// init context
ConvolutionContext<CPUTensor, float> context(input_shape, filter_shape, pad_h, pad_w, str_h, str_w);

// choose an algorithm
context.InitAlgorithmForUser(BLITZ_CONVOLUTION_NAIVE_DIRECT);

// run convolutions
Backend<CPUTensor, float>::Convolution2DForwardFunc(&input_cpu, &output_cpu, &filter_cpu, &context); 
Backend<CPUTensor, float>::Convolution2DBackwardFunc(&input_cpu, &output_cpu, &filter_cpu, &context); 
Backend<CPUTensor, float>::Convolution2DUpdateFunc(&input_cpu, &output_cpu, &filter_cpu, &context); 
```

### Data Layout

**CPU Backend**

Two layouts and four combinations are supported.

- input tensor: BLITZ_BUFFER_NCHW, filter tensor: BLITZ_FILTER_KCRS, output tensor: BLITZ_BUFFER_NHWC *or* BLITZ_BUFFER_NCHW
- input tensor: BLITZ_BUFFER_NHWC, filter tensor: BLITZ_FILTER_RSCK, output tensor: BLITZ_BUFFER_NHWC *or* BLITZ_BUFFER_NCHW

### Algorithm

**CPU Backend**

Three algorithms are supported, and extra memory is needed for some of them.

- BLITZ_CONVOLUTION_BLAS_GEMM (explicit GEMM)
- BLITZ_CONVOLUTION_BLAS_GEMM_BATCH (batch GEMM)
- BLITZ_CONVOLUTION_NAIVE_DIRECT (direct convolution without any dependency)

### Tuning

Three methods are supported for choosing an algorithm.

- InitAlgorithmForMemory (return an algorithm with the least extra memory).
- InitAlgorithmForSpeed (return an algorithm with the fast speed).
- InitAlgorithmForUser (specify an algorithm by user).
