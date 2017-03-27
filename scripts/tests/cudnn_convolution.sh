#!/bin/bash
#phase N C H W R S K P Q pad_h pad_w str_h str_w iterations

PHASE=(forward)
ALG=(gemm_pre)
BATCH_SIZE=128
ITERS=2

for((i=0;i<${#PHASE[@]};i++))
do
for((j=0;j<${#ALG[@]};j++))
do
#ALEXNET
./samples/gpu/cudnn/convolution ${PHASE[$i]} ${ALG[$j]} ${BATCH_SIZE} 3 224 224 11 11 64 55 55 3 3 4 4 ${ITERS}
./samples/gpu/cudnn/convolution ${PHASE[$i]} ${ALG[$j]} ${BATCH_SIZE} 64 27 27 5 5 192 27 27 2 2 1 1 ${ITERS}
./samples/gpu/cudnn/convolution ${PHASE[$i]} ${ALG[$j]} ${BATCH_SIZE} 192 13 13 3 3 384 13 13 1 1 1 1 ${ITERS} 
./samples/gpu/cudnn/convolution ${PHASE[$i]} ${ALG[$j]} ${BATCH_SIZE} 384 13 13 3 3 256 13 13 1 1 1 1 ${ITERS}
./samples/gpu/cudnn/convolution ${PHASE[$i]} ${ALG[$j]} ${BATCH_SIZE} 256 13 13 3 3 256 13 13 1 1 1 1 ${ITERS}
#VGG
./samples/gpu/cudnn/convolution ${PHASE[$i]} ${ALG[$j]} ${BATCH_SIZE} 3 224 224 3 3 64 224 224 1 1 1 1 ${ITERS}
./samples/gpu/cudnn/convolution ${PHASE[$i]} ${ALG[$j]} ${BATCH_SIZE} 64 112 112 3 3 128 112 112 1 1 1 1 ${ITERS}
./samples/gpu/cudnn/convolution ${PHASE[$i]} ${ALG[$j]} ${BATCH_SIZE} 128 56 56 3 3 256 56 56 1 1 1 1 ${ITERS} 
./samples/gpu/cudnn/convolution ${PHASE[$i]} ${ALG[$j]} ${BATCH_SIZE} 256 56 56 3 3 256 56 56 1 1 1 1 ${ITERS}
./samples/gpu/cudnn/convolution ${PHASE[$i]} ${ALG[$j]} ${BATCH_SIZE} 256 28 28 3 3 512 28 28 1 1 1 1 ${ITERS}
./samples/gpu/cudnn/convolution ${PHASE[$i]} ${ALG[$j]} ${BATCH_SIZE} 512 28 28 3 3 512 28 28 1 1 1 1 ${ITERS}
./samples/gpu/cudnn/convolution ${PHASE[$i]} ${ALG[$j]} ${BATCH_SIZE} 512 14 14 3 3 512 14 14 1 1 1 1 ${ITERS}
./samples/gpu/cudnn/convolution ${PHASE[$i]} ${ALG[$j]} ${BATCH_SIZE} 512 14 14 3 3 512 14 14 1 1 1 1 ${ITERS} 
#OVERFEAT
./samples/gpu/cudnn/convolution ${PHASE[$i]} ${ALG[$j]} ${BATCH_SIZE} 3 231 231 11 11 96 56 56 0 0 4 4 ${ITERS}
./samples/gpu/cudnn/convolution ${PHASE[$i]} ${ALG[$j]} ${BATCH_SIZE} 96 28 28 5 5 256 24 24 0 0 1 1 ${ITERS}
./samples/gpu/cudnn/convolution ${PHASE[$i]} ${ALG[$j]} ${BATCH_SIZE} 256 12 12 3 3 512 12 12 1 1 1 1 ${ITERS} 
./samples/gpu/cudnn/convolution ${PHASE[$i]} ${ALG[$j]} ${BATCH_SIZE} 512 12 12 3 3 1024 12 12 1 1 1 1 ${ITERS}
./samples/gpu/cudnn/convolution ${PHASE[$i]} ${ALG[$j]} ${BATCH_SIZE} 1024 12 12 3 3 1024 12 12 1 1 1 1 ${ITERS}
done
done

