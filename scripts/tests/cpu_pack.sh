#!/bin/bash
#phase N C H W R S K P Q pad_h pad_w str_h str_w iterations

PHASE=(unpack pack)
INPUT_LAYOUT=(nchw nhwc)
FILTER_LAYOUT=(kcrs)

for((i=0;i<${#PHASE[@]};i++))
do
for((j=0;j<${#FILTER_LAYOUT[@]};j++))
do
for((k=0;k<${#INPUT_LAYOUT[@]};k++))
do
if ./samples/cpu/convolution/pack ${PHASE[$i]} ${INPUT_LAYOUT[$k]} nchw ${FILTER_LAYOUT[$j]} 3 224 224 11 11 64 55 55 3 3 4 4 1
then
	echo "Alexnet first layer pass!" ${PHASE[$i]} ${INPUT_LAYOUT[$k]} ${FILTER_LAYOUT[$j]}
else
	echo "Alexnet first layer fail!" ${PHASE[$i]} ${INPUT_LAYOUT[$k]} ${FILTER_LAYOUT[$j]}
	exit 1
fi

if ./samples/cpu/convolution/pack ${PHASE[$i]} ${INPUT_LAYOUT[$k]} nchw ${FILTER_LAYOUT[$j]} 64 27 27 5 5 192 27 27 2 2 1 1 1
then
	echo "Alexnet second layer pass!" ${PHASE[$i]} ${INPUT_LAYOUT[$k]} ${FILTER_LAYOUT[$j]}
else
	echo "Alexnet second layer fail!" ${PHASE[$i]} ${INPUT_LAYOUT[$k]} ${FILTER_LAYOUT[$j]}
	exit 1
fi

if ./samples/cpu/convolution/pack ${PHASE[$i]} ${INPUT_LAYOUT[$k]} nchw ${FILTER_LAYOUT[$j]} 192 13 13 3 3 384 13 13 1 1 1 1 1 
then
	echo "Alexnet third layer pass!" ${PHASE[$i]} ${INPUT_LAYOUT[$k]} ${FILTER_LAYOUT[$j]}
else
	echo "Alexnet third layer fail!" ${PHASE[$i]} ${INPUT_LAYOUT[$k]} ${FILTER_LAYOUT[$j]}
	exit 1
fi

if ./samples/cpu/convolution/pack ${PHASE[$i]} ${INPUT_LAYOUT[$k]} nchw ${FILTER_LAYOUT[$j]} 384 13 13 3 3 256 13 13 1 1 1 1 1
then
	echo "Alexnet fourth layer pass!" ${PHASE[$i]} ${INPUT_LAYOUT[$k]} ${FILTER_LAYOUT[$j]}
else
	echo "Alexnet fourth layer fail!" ${PHASE[$i]} ${INPUT_LAYOUT[$k]} ${FILTER_LAYOUT[$j]}
	exit 1
fi

if ./samples/cpu/convolution/pack ${PHASE[$i]} ${INPUT_LAYOUT[$k]} nchw ${FILTER_LAYOUT[$j]} 256 13 13 3 3 256 13 13 1 1 1 1 1
then
	echo "Alexnet fifth layer pass!" ${PHASE[$i]} ${INPUT_LAYOUT[$k]} ${FILTER_LAYOUT[$j]}
else
	echo "Alexnet fifth layer fail!" ${PHASE[$i]} ${INPUT_LAYOUT[$k]} ${FILTER_LAYOUT[$j]}
	exit 1
fi
done
done
done
