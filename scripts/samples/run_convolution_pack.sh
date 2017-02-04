#!/bin/bash

#phase input_layout output_layout filter_layout C H W R S K P Q pad_h pad_w str_h str_w iterations

#MNIST
#./samples/cpu/convolution/pack unpack nchw nchw kcrs 1 28 28 5 5 16 24 24 0 0 5 5 10
#./samples/cpu/convolution/pack unpack nchw nchw kcrs 16 24 24 5 5 32 20 20 0 0 5 5 10

#Alexnet
##forward:
./samples/cpu/convolution/pack unpack nhwc nchw kcrs 64 27 27 5 5 192 27 27 2 2 1 1 8
./samples/cpu/convolution/pack unpack nhwc nchw kcrs 192 13 13 3 3 384 13 13 1 1 1 1 8 
./samples/cpu/convolution/pack unpack nhwc nchw kcrs 384 13 13 3 3 256 13 13 1 1 1 1 8
./samples/cpu/convolution/pack unpack nhwc nchw kcrs 256 13 13 3 3 256 13 13 1 1 1 1 8

#./samples/cpu/convolution/pack unpack nhwc nchw kcrs 3 224 224 11 11 64 55 55 3 3 4 4 8
#./samples/cpu/convolution/pack unpack nhwc nchw kcrs 64 27 27 5 5 192 27 27 2 2 1 1 8
#./samples/cpu/convolution/pack unpack nhwc nchw kcrs 192 13 13 3 3 384 13 13 1 1 1 1 8 
#./samples/cpu/convolution/pack unpack nhwc nchw kcrs 384 13 13 3 3 256 13 13 1 1 1 1 8
#./samples/cpu/convolution/pack unpack nhwc nchw kcrs 256 13 13 3 3 256 13 13 1 1 1 1 8

