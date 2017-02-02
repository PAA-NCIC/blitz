#!/bin/bash
#phase N C H W R S K P Q pad_h pad_w str_h str_w iterations

PHASE=(forward backward update)
ALG=(convolution_blas_gemm convolution_blas_gemm_batch)
INPUT_LAYOUT=(nchw nhwc)
OUTPUT_LAYOUT=(nchw nhwc)

for((i=0;i<${#PHASE[@]};i++))
do
for((j=0;j<${#ALG[@]};j++))
do
for((k=0;k<${#INPUT_LAYOUT[@]};k++))
do
for((v=0;v<${#OUTPUT_LAYOUT[@]};v++))
do
if ./samples/cpu/convolution/convolution ${PHASE[$i]} ${ALG[$j]} ${INPUT_LAYOUT[$k]} ${OUTPUT_LAYOUT[$v]} 128 3 224 224 11 11 64 55 55 3 3 4 4 1
then
	echo "Alexnet first layer pass!" ${PHASE[$i]} ${ALG[$j]} ${INPUT_LAYOUT[$k]} ${OUTPUT_LAYOUT[$v]}
else
	echo "Alexnet first layer fail!" ${PHASE[$i]} ${ALG[$j]} ${INPUT_LAYOUT[$k]} ${OUTPUT_LAYOUT[$v]}
	exit 1
fi

if ./samples/cpu/convolution/convolution ${PHASE[$i]} ${ALG[$j]} ${INPUT_LAYOUT[$k]} ${OUTPUT_LAYOUT[$v]} 128 64 27 27 5 5 192 27 27 2 2 1 1 1
then
	echo "Alexnet second layer pass!" ${PHASE[$i]} ${ALG[$j]} ${INPUT_LAYOUT[$k]} ${OUTPUT_LAYOUT[$v]}
else
	echo "Alexnet second layer fail!" ${PHASE[$i]} ${ALG[$j]} ${INPUT_LAYOUT[$k]} ${OUTPUT_LAYOUT[$v]}
	exit 1
fi

if ./samples/cpu/convolution/convolution ${PHASE[$i]} ${ALG[$j]} ${INPUT_LAYOUT[$k]} ${OUTPUT_LAYOUT[$v]} 128 192 13 13 3 3 384 13 13 1 1 1 1 1 
then
	echo "Alexnet third layer pass!" ${PHASE[$i]} ${ALG[$j]} ${INPUT_LAYOUT[$k]} ${OUTPUT_LAYOUT[$v]}
else
	echo "Alexnet third layer fail!" ${PHASE[$i]} ${ALG[$j]} ${INPUT_LAYOUT[$k]} ${OUTPUT_LAYOUT[$v]}
	exit 1
fi

if ./samples/cpu/convolution/convolution ${PHASE[$i]} ${ALG[$j]} ${INPUT_LAYOUT[$k]} ${OUTPUT_LAYOUT[$v]} 128 384 13 13 3 3 256 13 13 1 1 1 1 1
then
	echo "Alexnet fourth layer pass!" ${PHASE[$i]} ${ALG[$j]} ${INPUT_LAYOUT[$k]} ${OUTPUT_LAYOUT[$v]}
else
	echo "Alexnet fourth layer fail!" ${PHASE[$i]} ${ALG[$j]} ${INPUT_LAYOUT[$k]} ${OUTPUT_LAYOUT[$v]}
	exit 1
fi

if ./samples/cpu/convolution/convolution ${PHASE[$i]} ${ALG[$j]} ${INPUT_LAYOUT[$k]} ${OUTPUT_LAYOUT[$v]} 128 256 13 13 3 3 256 13 13 1 1 1 1 1
then
	echo "Alexnet fifth layer pass!" ${PHASE[$i]} ${ALG[$j]} ${INPUT_LAYOUT[$k]} ${OUTPUT_LAYOUT[$v]}
else
	echo "Alexnet fifth layer fail!" ${PHASE[$i]} ${ALG[$j]} ${INPUT_LAYOUT[$k]} ${OUTPUT_LAYOUT[$v]}
	exit 1
fi
done
done
done
done

