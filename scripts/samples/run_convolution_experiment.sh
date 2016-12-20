#!/bin/bash
#phase N C H W R S K P Q pad_h pad_w str_h str_w iterations

#N
#./samples/gpu/convolution/convolution forward convolution_sass_direct 128 128 12 12 4 4 128 9 9 0 0 1 1 2
#./samples/gpu/convolution/convolution forward convolution_sass_direct 256 128 12 12 4 4 128 9 9 0 0 1 1 2
#./samples/gpu/convolution/convolution forward convolution_sass_direct 384 128 12 12 4 4 128 9 9 0 0 1 1 2
#./samples/gpu/convolution/convolution forward convolution_sass_direct 512 128 12 12 4 4 128 9 9 0 0 1 1 2
#./samples/gpu/convolution/convolution forward convolution_sass_direct 640 128 12 12 4 4 128 9 9 0 0 1 1 2
#./samples/gpu/convolution/convolution forward convolution_sass_direct 768 128 12 12 4 4 128 9 9 0 0 1 1 2
#./samples/gpu/convolution/convolution forward convolution_sass_direct 896 128 12 12 4 4 128 9 9 0 0 1 1 2
#./samples/gpu/convolution/convolution forward convolution_sass_direct 1024 128 12 12 4 4 128 9 9 0 0 1 1 2

#C
#./samples/gpu/convolution/convolution forward convolution_sass_direct 128 32 12 12 4 4 128 9 9 0 0 1 1 2
#./samples/gpu/convolution/convolution forward convolution_sass_direct 128 64 12 12 4 4 128 9 9 0 0 1 1 2
#./samples/gpu/convolution/convolution forward convolution_sass_direct 128 96 12 12 4 4 128 9 9 0 0 1 1 2
#./samples/gpu/convolution/convolution forward convolution_sass_direct 128 128 12 12 4 4 128 9 9 0 0 1 1 2
#./samples/gpu/convolution/convolution forward convolution_sass_direct 128 160 12 12 4 4 128 9 9 0 0 1 1 2
#./samples/gpu/convolution/convolution forward convolution_sass_direct 128 224 12 12 4 4 128 9 9 0 0 1 1 2
#./samples/gpu/convolution/convolution forward convolution_sass_direct 128 256 12 12 4 4 128 9 9 0 0 1 1 2
#./samples/gpu/convolution/convolution forward convolution_sass_direct 128 288 12 12 4 4 128 9 9 0 0 1 1 2

#H
#./samples/gpu/convolution/convolution forward convolution_sass_direct 128 128 12 12 4 4 128 9 9 0 0 1 1 2
#./samples/gpu/convolution/convolution forward convolution_sass_direct 128 128 16 16 4 4 128 13 13 0 0 1 1 2
#./samples/gpu/convolution/convolution forward convolution_sass_direct 128 128 20 20 4 4 128 17 17 0 0 1 1 2
#./samples/gpu/convolution/convolution forward convolution_sass_direct 128 128 24 24 4 4 128 21 21 0 0 1 1 2
#./samples/gpu/convolution/convolution forward convolution_sass_direct 128 128 28 28 4 4 128 25 25 0 0 1 1 2
#./samples/gpu/convolution/convolution forward convolution_sass_direct 128 128 32 32 4 4 128 29 29 0 0 1 1 2
#./samples/gpu/convolution/convolution forward convolution_sass_direct 128 128 36 36 4 4 128 33 33 0 0 1 1 2

#filter size
#./samples/gpu/convolution/convolution forward convolution_sass_direct 128 128 12 12 12 12 128 1 1 0 0 1 1 2
#./samples/gpu/convolution/convolution forward convolution_sass_direct 128 128 12 12 11 11 128 2 2 0 0 1 1 2
#./samples/gpu/convolution/convolution forward convolution_sass_direct 128 128 12 12 10 10 128 3 3 0 0 1 1 2
#./samples/gpu/convolution/convolution forward convolution_sass_direct 128 128 12 12 9 9 128 4 4 0 0 1 1 2
#./samples/gpu/convolution/convolution forward convolution_sass_direct 128 128 12 12 8 8 128 5 5 0 0 1 1 2
#./samples/gpu/convolution/convolution forward convolution_sass_direct 128 128 12 12 7 7 128 6 6 0 0 1 1 2
#./samples/gpu/convolution/convolution forward convolution_sass_direct 128 128 12 12 6 6 128 7 7 0 0 1 1 2
#./samples/gpu/convolution/convolution forward convolution_sass_direct 128 128 12 12 5 5 128 8 8 0 0 1 1 2
#./samples/gpu/convolution/convolution forward convolution_sass_direct 128 128 12 12 4 4 128 9 9 0 0 1 1 2

#K
#./samples/gpu/convolution/convolution forward convolution_sass_direct 128 128 12 12 4 4 128 9 9 0 0 1 1 2
#./samples/gpu/convolution/convolution forward convolution_sass_direct 128 128 12 12 4 4 256 9 9 0 0 1 1 2
#./samples/gpu/convolution/convolution forward convolution_sass_direct 128 128 12 12 4 4 384 9 9 0 0 1 1 2
#./samples/gpu/convolution/convolution forward convolution_sass_direct 128 128 12 12 4 4 512 9 9 0 0 1 1 2
#./samples/gpu/convolution/convolution forward convolution_sass_direct 128 128 12 12 4 4 640 9 9 0 0 1 1 2
#./samples/gpu/convolution/convolution forward convolution_sass_direct 128 128 12 12 4 4 768 9 9 0 0 1 1 2
#./samples/gpu/convolution/convolution forward convolution_sass_direct 128 128 12 12 4 4 896 9 9 0 0 1 1 2
#./samples/gpu/convolution/convolution forward convolution_sass_direct 128 128 12 12 4 4 1024 9 9 0 0 1 1 2

#stride
./samples/gpu/convolution/convolution forward convolution_sass_direct 128 128 12 12 4 4 128 9 9 0 0 1 1 2
./samples/gpu/convolution/convolution forward convolution_sass_direct 128 128 12 12 4 4 128 5 5 0 0 2 2 2
./samples/gpu/convolution/convolution forward convolution_sass_direct 128 128 12 12 4 4 128 3 3 0 0 3 3 2
./samples/gpu/convolution/convolution forward convolution_sass_direct 128 128 12 12 4 4 128 3 3 0 0 4 4 2
./samples/gpu/convolution/convolution forward convolution_sass_direct 128 128 12 12 4 4 128 2 2 0 0 5 5 2
./samples/gpu/convolution/convolution forward convolution_sass_direct 128 128 12 12 4 4 128 2 2 0 0 6 6 2

#pad
#./samples/gpu/convolution/convolution forward convolution_sass_direct 128 128 12 12 8 8 128 5 5 0 0 1 1 2
#./samples/gpu/convolution/convolution forward convolution_sass_direct 128 128 12 12 8 8 128 7 7 1 1 1 1 2
#./samples/gpu/convolution/convolution forward convolution_sass_direct 128 128 12 12 8 8 128 9 9 2 2 1 1 2
#./samples/gpu/convolution/convolution forward convolution_sass_direct 128 128 12 12 8 8 128 11 11 3 3 1 1 2
#./samples/gpu/convolution/convolution forward convolution_sass_direct 128 128 12 12 8 8 128 13 13 4 4 1 1 2
#./samples/gpu/convolution/convolution forward convolution_sass_direct 128 128 12 12 8 8 128 15 15 5 5 1 1 2
#./samples/gpu/convolution/convolution forward convolution_sass_direct 128 128 12 12 8 8 128 17 17 6 6 1 1 2

#cudnn
#N
#./samples/gpu/convolution/cudnn forward gemm_pre 128 256 24 24 4 4 256 21 21 0 0 1 1 2
#./samples/gpu/convolution/cudnn forward gemm_pre 256 256 24 24 4 4 256 21 21 0 0 1 1 2
#./samples/gpu/convolution/cudnn forward gemm_pre 384 256 24 24 4 4 256 21 21 0 0 1 1 2
#./samples/gpu/convolution/cudnn forward gemm_pre 512 256 24 24 4 4 256 21 21 0 0 1 1 2
#./samples/gpu/convolution/cudnn forward gemm_pre 640 256 24 24 4 4 256 21 21 0 0 1 1 2
#./samples/gpu/convolution/cudnn forward gemm_pre 768 256 24 24 4 4 256 21 21 0 0 1 1 2
#./samples/gpu/convolution/cudnn forward gemm_pre 896 256 24 24 4 4 256 21 21 0 0 1 1 2
#./samples/gpu/convolution/cudnn forward gemm_pre 1024 256 24 24 4 4 256 21 21 0 0 1 1 2

#C
#./samples/gpu/convolution/cudnn forward gemm_pre 128 128 24 24 4 4 256 21 21 0 0 1 1 2
#./samples/gpu/convolution/cudnn forward gemm_pre 128 256 24 24 4 4 256 21 21 0 0 1 1 2
#./samples/gpu/convolution/cudnn forward gemm_pre 128 384 24 24 4 4 256 21 21 0 0 1 1 2
#./samples/gpu/convolution/cudnn forward gemm_pre 128 512 24 24 4 4 256 21 21 0 0 1 1 2
#./samples/gpu/convolution/cudnn forward gemm_pre 128 640 24 24 4 4 256 21 21 0 0 1 1 2
#./samples/gpu/convolution/cudnn forward gemm_pre 128 768 24 24 4 4 256 21 21 0 0 1 1 2
#./samples/gpu/convolution/cudnn forward gemm_pre 128 896 24 24 4 4 256 21 21 0 0 1 1 2
#./samples/gpu/convolution/cudnn forward gemm_pre 128 1024 24 24 4 4 256 21 21 0 0 1 1 2

#H
#./samples/gpu/convolution/cudnn forward gemm_pre 128 256 12 12 4 4 256 9 9 0 0 1 1 2
#./samples/gpu/convolution/cudnn forward gemm_pre 128 256 24 24 4 4 256 21 21 0 0 1 1 2
#./samples/gpu/convolution/cudnn forward gemm_pre 128 256 36 36 4 4 256 33 33 0 0 1 1 2
#./samples/gpu/convolution/cudnn forward gemm_pre 128 256 48 48 4 4 256 45 45 0 0 1 1 2
#./samples/gpu/convolution/cudnn forward gemm_pre 128 256 60 60 4 4 256 57 57 0 0 1 1 2
#./samples/gpu/convolution/cudnn forward gemm_pre 128 256 72 72 4 4 256 69 69 0 0 1 1 2

#filter size
#./samples/gpu/convolution/cudnn forward gemm_pre 128 256 24 24 16 16 256 9 9 0 0 1 1 2
#./samples/gpu/convolution/cudnn forward gemm_pre 128 256 24 24 14 14 256 11 11 0 0 1 1 2
#./samples/gpu/convolution/cudnn forward gemm_pre 128 256 24 24 12 12 256 13 13 0 0 1 1 2
#./samples/gpu/convolution/cudnn forward gemm_pre 128 256 24 24 10 10 256 15 15 0 0 1 1 2
#./samples/gpu/convolution/cudnn forward gemm_pre 128 256 24 24 8 8 256 17 17 0 0 1 1 2
#./samples/gpu/convolution/cudnn forward gemm_pre 128 256 24 24 6 6 256 19 19 0 0 1 1 2
#./samples/gpu/convolution/cudnn forward gemm_pre 128 256 24 24 4 4 256 21 21 0 0 1 1 2
#./samples/gpu/convolution/cudnn forward gemm_pre 128 256 24 24 2 2 256 23 23 0 0 1 1 2

#K
#./samples/gpu/convolution/cudnn forward gemm_pre 128 256 24 24 4 4 128 21 21 0 0 1 1 2
#./samples/gpu/convolution/cudnn forward gemm_pre 128 256 24 24 4 4 256 21 21 0 0 1 1 2
#./samples/gpu/convolution/cudnn forward gemm_pre 128 256 24 24 4 4 384 21 21 0 0 1 1 2
#./samples/gpu/convolution/cudnn forward gemm_pre 128 256 24 24 4 4 512 21 21 0 0 1 1 2
#./samples/gpu/convolution/cudnn forward gemm_pre 128 256 24 24 4 4 640 21 21 0 0 1 1 2
#./samples/gpu/convolution/cudnn forward gemm_pre 128 256 24 24 4 4 768 21 21 0 0 1 1 2
#./samples/gpu/convolution/cudnn forward gemm_pre 128 256 24 24 4 4 896 21 21 0 0 1 1 2
#./samples/gpu/convolution/cudnn forward gemm_pre 128 256 24 24 4 4 1024 21 21 0 0 1 1 2

#stride
#./samples/gpu/convolution/cudnn forward gemm_pre 128 256 24 24 4 4 128 21 21 0 0 1 1 2
#./samples/gpu/convolution/cudnn forward gemm_pre 128 256 24 24 4 4 128 11 11 0 0 2 2 2
#./samples/gpu/convolution/cudnn forward gemm_pre 128 256 24 24 4 4 128 7 7 0 0 3 3 2
#./samples/gpu/convolution/cudnn forward gemm_pre 128 256 24 24 4 4 128 6 6 0 0 4 4 2
#./samples/gpu/convolution/cudnn forward gemm_pre 128 256 24 24 4 4 128 5 5 0 0 5 5 2
#./samples/gpu/convolution/cudnn forward gemm_pre 128 256 24 24 4 4 128 4 4 0 0 6 6 2

#pad
#./samples/gpu/convolution/cudnn forward gemm_implicit 128 256 24 24 8 8 128 17 17 0 0 1 1 2
#./samples/gpu/convolution/cudnn forward gemm_implicit 128 256 24 24 8 8 128 19 19 1 1 1 1 2
#./samples/gpu/convolution/cudnn forward gemm_implicit 128 256 24 24 8 8 128 21 21 2 2 1 1 2
#./samples/gpu/convolution/cudnn forward gemm_implicit 128 256 24 24 8 8 128 23 23 3 3 1 1 2
#./samples/gpu/convolution/cudnn forward gemm_implicit 128 256 24 24 8 8 128 25 25 4 4 1 1 2
#./samples/gpu/convolution/cudnn forward gemm_implicit 128 256 24 24 8 8 128 27 27 5 5 1 1 2
#./samples/gpu/convolution/cudnn forward gemm_implicit 128 256 24 24 8 8 128 29 29 6 6 1 1 2
#./samples/gpu/convolution/cudnn forward gemm_implicit 128 256 24 24 8 8 128 31 31 7 7 1 1 2
