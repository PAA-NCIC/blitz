#!/bin/bash
#phase N C H W R S K P Q pad_h pad_w str_h str_w iterations

PHASE=(forward)
BATCH_SIZE=128
ITERS=5

for((i=0;i<${#PHASE[@]};i++))
do
if ./samples/cpu/mkl/convolution ${PHASE[$i]} ${BATCH_SIZE} 3 224 224 11 11 64 55 55 3 3 4 4 ${ITERS}
then
	echo "Alexnet first layer pass!" ${PHASE[$i]}
else
	echo "Alexnet first layer fail!" ${PHASE[$i]}
	exit 1
fi

if ./samples/cpu/mkl/convolution ${PHASE[$i]} ${BATCH_SIZE} 64 27 27 5 5 192 27 27 2 2 1 1 ${ITERS}
then
	echo "Alexnet second layer pass!" ${PHASE[$i]}
else
	echo "Alexnet second layer fail!" ${PHASE[$i]}
	exit 1
fi

if ./samples/cpu/mkl/convolution ${PHASE[$i]} ${BATCH_SIZE} 192 13 13 3 3 384 13 13 1 1 1 1 ${ITERS} 
then
	echo "Alexnet third layer pass!" ${PHASE[$i]}
else
	echo "Alexnet third layer fail!" ${PHASE[$i]}
	exit 1
fi

if ./samples/cpu/mkl/convolution ${PHASE[$i]} ${BATCH_SIZE} 384 13 13 3 3 256 13 13 1 1 1 1 ${ITERS}
then
	echo "Alexnet fourth layer pass!" ${PHASE[$i]} 
else
	echo "Alexnet fourth layer fail!" ${PHASE[$i]}
	exit 1
fi

if ./samples/cpu/mkl/convolution ${PHASE[$i]} ${BATCH_SIZE} 256 13 13 3 3 256 13 13 1 1 1 1 ${ITERS}
then
	echo "Alexnet fifth layer pass!" ${PHASE[$i]} 
else
	echo "Alexnet fifth layer fail!" ${PHASE[$i]}
	exit 1
fi
done

