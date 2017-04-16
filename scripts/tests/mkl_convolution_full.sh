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

if ./samples/cpu/mkl/convolution  ${PHASE[$i]} ${BATCH_SIZE} 256 13 13 3 3 256 13 13 1 1 1 1 ${ITERS}
then
	echo "Alexnet fifth layer pass!" ${PHASE[$i]}
else
	echo "Alexnet fifth layer fail!" ${PHASE[$i]}
	exit 1
fi
done

for((i=0;i<${#PHASE[@]};i++))
do
if ./samples/cpu/mkl/convolution  ${PHASE[$i]} ${BATCH_SIZE} 3 224 224 3 3 64 224 224 1 1 1 1 ${ITERS}
then
	echo "VGG first layer pass!" ${PHASE[$i]}
else
	echo "VGG first layer fail!" ${PHASE[$i]}
	exit 1
fi

if ./samples/cpu/mkl/convolution  ${PHASE[$i]} ${BATCH_SIZE} 64 112 112 3 3 128 112 112 1 1 1 1 ${ITERS}
then
	echo "VGG second layer pass!" ${PHASE[$i]}
else
	echo "VGG second layer fail!" ${PHASE[$i]}
  exit 1
fi

if ./samples/cpu/mkl/convolution  ${PHASE[$i]} ${BATCH_SIZE} 128 56 56 3 3 256 56 56 1 1 1 1 ${ITERS} 
then
	echo "VGG third layer pass!" ${PHASE[$i]}
else
	echo "VGG third layer fail!" ${PHASE[$i]}
  exit 1
fi

if ./samples/cpu/mkl/convolution  ${PHASE[$i]} ${BATCH_SIZE} 256 56 56 3 3 256 56 56 1 1 1 1 ${ITERS}
then
	echo "VGG fourth layer pass!" ${PHASE[$i]}
else
	echo "VGG fourth layer fail!" ${PHASE[$i]}
	exit 1
fi

if ./samples/cpu/mkl/convolution  ${PHASE[$i]} ${BATCH_SIZE} 256 28 28 3 3 512 28 28 1 1 1 1 ${ITERS}
then
	echo "VGG fifth layer pass!" ${PHASE[$i]}
else
	echo "VGG fifth layer fail!" ${PHASE[$i]}
	exit 1
fi

if ./samples/cpu/mkl/convolution  ${PHASE[$i]} ${BATCH_SIZE} 512 28 28 3 3 512 28 28 1 1 1 1 ${ITERS}
then
	echo "VGG sixth layer pass!" ${PHASE[$i]}
else
	echo "VGG sixth layer fail!" ${PHASE[$i]}
	exit 1
fi

if ./samples/cpu/mkl/convolution  ${PHASE[$i]} ${BATCH_SIZE} 512 14 14 3 3 512 14 14 1 1 1 1 ${ITERS}
then
	echo "VGG seventh layer pass!" ${PHASE[$i]}
else
	echo "VGG seventh layer fail!" ${PHASE[$i]}
	exit 1
fi

if ./samples/cpu/mkl/convolution  ${PHASE[$i]} ${BATCH_SIZE} 512 14 14 3 3 512 14 14 1 1 1 1 ${ITERS}
then
	echo "VGG eighth layer pass!" ${PHASE[$i]}
else
	echo "VGG eighth layer fail!" ${PHASE[$i]}
	exit 1
fi
done

for((i=0;i<${#PHASE[@]};i++))
do
if ./samples/cpu/mkl/convolution  ${PHASE[$i]} ${BATCH_SIZE} 3 231 231 11 11 96 56 56 0 0 4 4 ${ITERS}
then
	echo "Overfeat first layer pass!" ${PHASE[$i]} 
else
	echo "Overfeat first layer fail!" ${PHASE[$i]}
	exit 1
fi

if ./samples/cpu/mkl/convolution  ${PHASE[$i]} ${BATCH_SIZE} 96 28 28 5 5 256 24 24 0 0 1 1 ${ITERS}
then
	echo "Overfeat second layer pass!" ${PHASE[$i]}
else
	echo "Overfeat second layer fail!" ${PHASE[$i]}
	exit 1
fi

if ./samples/cpu/mkl/convolution  ${PHASE[$i]} ${BATCH_SIZE} 256 12 12 3 3 512 12 12 1 1 1 1 ${ITERS} 
then
	echo "Overfeat third layer pass!" ${PHASE[$i]}
else
	echo "Overfeat third layer fail!" ${PHASE[$i]}
	exit 1
fi

if ./samples/cpu/mkl/convolution  ${PHASE[$i]} ${BATCH_SIZE} 512 12 12 3 3 1024 12 12 1 1 1 1 ${ITERS}
then
	echo "Overfeat fourth layer pass!" ${PHASE[$i]}
else
	echo "Overfeat fourth layer fail!" ${PHASE[$i]}
	exit 1
fi

if ./samples/cpu/mkl/convolution  ${PHASE[$i]} ${BATCH_SIZE} 1024 12 12 3 3 1024 12 12 1 1 1 1 ${ITERS}
then
	echo "Overfeat fifth layer pass!" ${PHASE[$i]}
else
	echo "Overfeat fifth layer fail!" ${PHASE[$i]}
	exit 1
fi
done
