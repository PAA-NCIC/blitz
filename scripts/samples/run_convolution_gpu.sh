#!/bin/bash
#phase N C H W R S K P Q pad_h pad_w str_h str_w

#MNIST
#forward
#./samples/gpu/convolution/convolution forward convolution_sass_direct 128 1 28 28 5 5 16 24 24 0 0 1 1 2
#./samples/gpu/convolution/convolution forward convolution_sass_direct 128 16 24 24 5 5 32 20 20 0 0 1 1 2

#Alexnet
##forward:
#./samples/gpu/convolution/convolution forward convolution_sass_direct 128 3 224 224 11 11 64 55 55 3 3 4 4 2
#./samples/gpu/convolution/convolution forward convolution_sass_direct 128 64 27 27 5 5 192 27 27 2 2 1 1 2
#./samples/gpu/convolution/convolution forward convolution_sass_direct 128 192 13 13 3 3 384 13 13 1 1 1 1 2 
#./samples/gpu/convolution/convolution forward convolution_sass_direct 128 384 13 13 3 3 256 13 13 1 1 1 1 2
#./samples/gpu/convolution/convolution forward convolution_sass_direct 128 256 13 13 3 3 256 13 13 1 1 1 1 2
##backward:
#./samples/gpu/convolution/convolution backward convolution_sass_direct 128 3 224 224 11 11 64 55 55 3 3 4 4 1
#./samples/gpu/convolution/convolution backward convolution_sass_direct 128 64 27 27 5 5 192 27 27 2 2 1 1 1
#./samples/gpu/convolution/convolution backward convolution_sass_direct 128 192 13 13 3 3 384 13 13 1 1 1 1 1
#./samples/gpu/convolution/convolution backward convolution_sass_direct 128 384 13 13 3 3 256 13 13 1 1 1 1 1
#./samples/gpu/convolution/convolution backward convolution_sass_direct 128 256 13 13 3 3 256 13 13 1 1 1 1 1
##update:
#./samples/gpu/convolution/convolution update convolution_sass_direct 128 3 224 224 11 11 64 55 55 3 3 4 4 1
#./samples/gpu/convolution/convolution update convolution_sass_direct 128 64 27 27 5 5 192 27 27 2 2 1 1 1
#./samples/gpu/convolution/convolution update convolution_sass_direct 128 192 13 13 3 3 384 13 13 1 1 1 1 1
#./samples/gpu/convolution/convolution update convolution_sass_direct 128 384 13 13 3 3 256 13 13 1 1 1 1 1
#./samples/gpu/convolution/convolution update convolution_sass_direct 128 256 13 13 3 3 256 13 13 1 1 1 1 1

#Vgg-a
##forward:
#./samples/gpu/convolution/convolution forward convolution_sass_direct 128 3 224 224 3 3 64 224 224 1 1 1 1 2
#./samples/gpu/convolution/convolution forward convolution_sass_direct 128 64 112 112 3 3 128 112 112 1 1 1 1 2
#./samples/gpu/convolution/convolution forward convolution_sass_direct 128 128 56 56 3 3 256 56 56 1 1 1 1 2 
#./samples/gpu/convolution/convolution forward convolution_sass_direct 128 256 56 56 3 3 256 56 56 1 1 1 1 2
#./samples/gpu/convolution/convolution forward convolution_sass_direct 128 256 28 28 3 3 512 28 28 1 1 1 1 2
#./samples/gpu/convolution/convolution forward convolution_sass_direct 128 512 28 28 3 3 512 28 28 1 1 1 1 2
#./samples/gpu/convolution/convolution forward convolution_sass_direct 128 512 14 14 3 3 512 14 14 1 1 1 1 2
#./samples/gpu/convolution/convolution forward convolution_sass_direct 128 512 14 14 3 3 512 14 14 1 1 1 1 2

#Overleaf
##forward:
#./samples/gpu/convolution/convolution forward convolution_sass_direct 128 3 231 231 11 11 96 56 56 0 0 4 4 2
#./samples/gpu/convolution/convolution forward convolution_sass_direct 128 96 28 28 5 5 256 24 24 0 0 1 1 2
#./samples/gpu/convolution/convolution forward convolution_sass_direct 128 256 12 12 3 3 512 12 12 1 1 1 1 2 
#./samples/gpu/convolution/convolution forward convolution_sass_direct 128 512 12 12 3 3 1024 12 12 1 1 1 1 2
./samples/gpu/convolution/convolution forward convolution_sass_direct 128 1024 12 12 3 3 1024 12 12 1 1 1 1 2
