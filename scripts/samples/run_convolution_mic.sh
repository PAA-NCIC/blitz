#!/bin/bash
#phase N C H W R S K P Q pad_h pad_w str_h str_w iterations

#MNIST
#forward
#./samples/mic/convolution/convolution forward convolution_xsmm_direct 128 1 28 28 5 5 16 24 24 0 0 1 1 2
#./samples/mic/convolution/convolution forward convolution_xsmm_direct 128 16 24 24 5 5 32 20 20 0 0 1 1 2

#Alexnet
##forward:
#./samples/mic/convolution/convolution forward convolution_xsmm_direct 128 3 224 224 11 11 64 55 55 3 3 4 4 2
#./samples/mic/convolution/convolution forward convolution_xsmm_direct 128 64 27 27 5 5 192 27 27 2 2 1 1 2
#./samples/mic/convolution/convolution forward convolution_xsmm_direct 128 192 13 13 3 3 384 13 13 1 1 1 1 2 
#./samples/mic/convolution/convolution forward convolution_xsmm_direct 128 384 13 13 3 3 256 13 13 1 1 1 1 2
#./samples/mic/convolution/convolution forward convolution_xsmm_direct 128 256 13 13 3 3 256 13 13 1 1 1 1 2

##backward:
#./samples/mic/convolution/convolution backward convolution_xsmm_direct 128 3 224 224 11 11 64 55 55 3 3 4 4 1
#./samples/mic/convolution/convolution backward convolution_xsmm_direct 128 64 27 27 5 5 192 27 27 2 2 1 1 1
#./samples/mic/convolution/convolution backward convolution_xsmm_direct 128 192 13 13 3 3 384 13 13 1 1 1 1 1
#./samples/mic/convolution/convolution backward convolution_xsmm_direct 128 384 13 13 3 3 256 13 13 1 1 1 1 1
#./samples/mic/convolution/convolution backward convolution_xsmm_direct 128 256 13 13 3 3 256 13 13 1 1 1 1 1
##update:
#./samples/mic/convolution/convolution update convolution_xsmm_direct 128 3 224 224 11 11 64 55 55 3 3 4 4 1
#./samples/mic/convolution/convolution update convolution_xsmm_direct 128 64 27 27 5 5 192 27 27 2 2 1 1 1
#./samples/mic/convolution/convolution update convolution_xsmm_direct 128 192 13 13 3 3 384 13 13 1 1 1 1 1
#./samples/mic/convolution/convolution update convolution_xsmm_direct 128 384 13 13 3 3 256 13 13 1 1 1 1 1
#./samples/mic/convolution/convolution update convolution_xsmm_direct 128 256 13 13 3 3 256 13 13 1 1 1 1 1

#Vgg-a
##forward:
#./samples/mic/convolution/convolution forward convolution_xsmm_direct 128 3 224 224 3 3 64 224 224 1 1 1 1 2
#./samples/mic/convolution/convolution forward convolution_xsmm_direct 128 64 112 112 3 3 128 112 112 1 1 1 1 2
#./samples/mic/convolution/convolution forward convolution_xsmm_direct 128 128 56 56 3 3 256 56 56 1 1 1 1 2 
#./samples/mic/convolution/convolution forward convolution_xsmm_direct 128 256 56 56 3 3 256 56 56 1 1 1 1 2
#./samples/mic/convolution/convolution forward convolution_xsmm_direct 128 256 28 28 3 3 512 28 28 1 1 1 1 2
#./samples/mic/convolution/convolution forward convolution_xsmm_direct 128 512 28 28 3 3 512 28 28 1 1 1 1 2
#./samples/mic/convolution/convolution forward convolution_xsmm_direct 128 512 14 14 3 3 512 14 14 1 1 1 1 2
#./samples/mic/convolution/convolution forward convolution_xsmm_direct 128 512 14 14 3 3 512 14 14 1 1 1 1 2

#Overfeat
#phase iter H W N C K R S pad str P Q 
##forward:
./samples/mic/convolution/convolution forward convolution_xsmm_direct 1 231 231 1 3    96   11 11 0 4 56 56
./samples/mic/convolution/convolution forward convolution_xsmm_direct 1 28  28  1 96   256  5  5  0 1 24 24
./samples/mic/convolution/convolution forward convolution_xsmm_direct 1 14  14  1 256  512  3  3  0 1 12 12
./samples/mic/convolution/convolution forward convolution_xsmm_direct 1 14  14  1 512  1024 3  3  0 1 12 12
./samples/mic/convolution/convolution forward convolution_xsmm_direct 1 14  14  1 1024 1024 3  3  0 1 12 12
##backward:
./samples/mic/convolution/convolution backward convolution_xsmm_direct 1 231 231 1 3    96   11 11 0 4 56 56
./samples/mic/convolution/convolution backward convolution_xsmm_direct 1 28  28  1 96   256  5  5  0 1 24 24
./samples/mic/convolution/convolution backward convolution_xsmm_direct 1 14  14  1 256  512  3  3  0 1 12 12
./samples/mic/convolution/convolution backward convolution_xsmm_direct 1 14  14  1 512  1024 3  3  0 1 12 12
./samples/mic/convolution/convolution backward convolution_xsmm_direct 1 14  14  1 1024 1024 3  3  0 1 12 12
##update:
./samples/mic/convolution/convolution update convolution_xsmm_direct 1 231 231 1 3    96   11 11 0 4 56 56
./samples/mic/convolution/convolution update convolution_xsmm_direct 1 28  28  1 96   256  5  5  0 1 24 24
./samples/mic/convolution/convolution update convolution_xsmm_direct 1 14  14  1 256  512  3  3  0 1 12 12
./samples/mic/convolution/convolution update convolution_xsmm_direct 1 14  14  1 512  1024 3  3  0 1 12 12
./samples/mic/convolution/convolution update convolution_xsmm_direct 1 14  14  1 1024 1024 3  3  0 1 12 12

