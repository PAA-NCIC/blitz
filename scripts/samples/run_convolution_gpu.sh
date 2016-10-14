#!/bin/bash
#phase N C H W R S K P Q pad_h pad_w str_h str_w

#MNIST first
#./samples/gpu/convolution/convolution forward asm_implicit 64 1 28 28 5 5 64 24 24 0 0 1 1

#Alexnet
#forward:
#./samples/gpu/convolution/convolution forward asm_implicit 128 3 224 224 11 11 64 55 55 3 3 4 4 
#./samples/gpu/convolution/convolution forward asm_implicit 128 64 27 27 5 5 192 27 27 2 2 1 1
#./samples/gpu/convolution/convolution forward asm_implicit 128 192 13 13 3 3 384 13 13 1 1 1 1 
#./samples/gpu/convolution/convolution forward asm_implicit 128 384 13 13 3 3 256 13 13 1 1 1 1 
#./samples/gpu/convolution/convolution forward asm_implicit 128 256 13 13 3 3 256 13 13 1 1 1 1 
#backward:
#./samples/gpu/convolution/convolution backward asm_implicit 128 3 224 224 11 11 64 55 55 3 3 4 4 
#./samples/gpu/convolution/convolution backward asm_implicit 128 64 27 27 5 5 192 27 27 2 2 1 1
#./samples/gpu/convolution/convolution backward asm_implicit 128 192 13 13 3 3 384 13 13 1 1 1 1 
#./samples/gpu/convolution/convolution backward asm_implicit 128 384 13 13 3 3 256 13 13 1 1 1 1 
#./samples/gpu/convolution/convolution backward asm_implicit 128 256 13 13 3 3 256 13 13 1 1 1 1 
#backward:
#./samples/gpu/convolution/convolution update asm_implicit 128 3 224 224 11 11 64 55 55 3 3 4 4 
./samples/gpu/convolution/convolution update asm_implicit 128 64 27 27 5 5 192 27 27 2 2 1 1
./samples/gpu/convolution/convolution update asm_implicit 128 192 13 13 3 3 384 13 13 1 1 1 1 
./samples/gpu/convolution/convolution update asm_implicit 128 384 13 13 3 3 256 13 13 1 1 1 1 
./samples/gpu/convolution/convolution update asm_implicit 128 256 13 13 3 3 256 13 13 1 1 1 1 
