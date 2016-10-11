#!/bin/bash
#phase N C H W R S K P Q pad_h pad_w str_h str_w
#MNIST first
#./samples/gpu/convolution/convolution_implicit_asm forward 64 1 28 28 5 5 64 24 24 0 0 1 1
#Alexnet
./samples/gpu/convolution/convolution_implicit_asm forward 128 3 224 224 11 11 64 55 55 3 3 4 4
./samples/gpu/convolution/convolution_implicit_asm forward 128 64 55 55 5 5 192 27 27 2 2 1 1
./samples/gpu/convolution/convolution_implicit_asm forward 128 192 27 27 3 3 384 13 13 1 1 1 1
./samples/gpu/convolution/convolution_implicit_asm forward 128 384 13 13 3 3 256 13 13 1 1 1 1
./samples/gpu/convolution/convolution_implicit_asm forward 128 256 13 13 3 3 256 13 13 1 1 1 1
