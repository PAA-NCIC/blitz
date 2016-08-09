#!/bin/bash

#normal mnist
rm mnist_conv.result
bash tackle.sh ../../log/mnist_conv_1.log mnist.py >> mnist_conv.result
bash tackle.sh ../../log/mnist_conv_2.log mnist.py >> mnist_conv.result
bash tackle.sh ../../log/mnist_conv_4.log mnist.py >> mnist_conv.result
bash tackle.sh ../../log/mnist_conv_8.log mnist.py >> mnist_conv.result
bash tackle.sh ../../log/mnist_conv_16.log mnist.py >> mnist_conv.result
sed -e "s/\[//g" mnist_conv.result > tmp
sed -e "s/\]//g" tmp > mnist_conv.result

#batch mnist
rm mnist_conv_batch.result
bash tackle.sh ../../log/mnist_conv_batch_1.log mnist.py >> mnist_conv_batch.result
bash tackle.sh ../../log/mnist_conv_batch_2.log mnist.py >> mnist_conv_batch.result
bash tackle.sh ../../log/mnist_conv_batch_4.log mnist.py >> mnist_conv_batch.result
bash tackle.sh ../../log/mnist_conv_batch_8.log mnist.py >> mnist_conv_batch.result
bash tackle.sh ../../log/mnist_conv_batch_16.log mnist.py >> mnist_conv_batch.result
sed -e "s/\[//g" mnist_conv_batch.result > tmp
sed -e "s/\]//g" tmp > mnist_conv_batch.result

#normal cifar10
rm cifar10_conv.result
bash tackle.sh ../../log/cifar10_conv_1.log cifar10.py >> cifar10_conv.result
bash tackle.sh ../../log/cifar10_conv_2.log cifar10.py >> cifar10_conv.result
bash tackle.sh ../../log/cifar10_conv_4.log cifar10.py >> cifar10_conv.result
bash tackle.sh ../../log/cifar10_conv_8.log cifar10.py >> cifar10_conv.result
bash tackle.sh ../../log/cifar10_conv_16.log cifar10.py >> cifar10_conv.result
sed -e "s/\[//g" cifar10_conv.result > tmp
sed -e "s/\]//g" tmp > cifar10_conv.result

#batch cifar10
rm cifar10_conv_batch.result
bash tackle.sh ../../log/cifar10_conv_batch_1.log cifar10.py >> cifar10_conv_batch.result
bash tackle.sh ../../log/cifar10_conv_batch_2.log cifar10.py >> cifar10_conv_batch.result
bash tackle.sh ../../log/cifar10_conv_batch_4.log cifar10.py >> cifar10_conv_batch.result
bash tackle.sh ../../log/cifar10_conv_batch_8.log cifar10.py >> cifar10_conv_batch.result
bash tackle.sh ../../log/cifar10_conv_batch_16.log cifar10.py >> cifar10_conv_batch.result
sed -e "s/\[//g" cifar10_conv_batch.result > tmp
sed -e "s/\]//g" tmp > cifar10_conv_batch.result

#normal alexnet
rm alexnet_conv.result
bash tackle.sh ../../log/alexnet_conv_1.log alexnet.py >> alexnet_conv.result
bash tackle.sh ../../log/alexnet_conv_2.log alexnet.py >> alexnet_conv.result
bash tackle.sh ../../log/alexnet_conv_4.log alexnet.py >> alexnet_conv.result
bash tackle.sh ../../log/alexnet_conv_8.log alexnet.py >> alexnet_conv.result
bash tackle.sh ../../log/alexnet_conv_16.log alexnet.py >> alexnet_conv.result
sed -e "s/\[//g" alexnet_conv.result > tmp
sed -e "s/\]//g" tmp > alexnet_conv.result

#batch cifar10
rm alexnet_conv_batch.result
bash tackle.sh ../../log/alexnet_conv_batch_1.log alexnet.py >> alexnet_conv_batch.result
bash tackle.sh ../../log/alexnet_conv_batch_2.log alexnet.py >> alexnet_conv_batch.result
bash tackle.sh ../../log/alexnet_conv_batch_4.log alexnet.py >> alexnet_conv_batch.result
bash tackle.sh ../../log/alexnet_conv_batch_8.log alexnet.py >> alexnet_conv_batch.result
bash tackle.sh ../../log/alexnet_conv_batch_16.log alexnet.py >> alexnet_conv_batch.result
sed -e "s/\[//g" alexnet_conv_batch.result > tmp
sed -e "s/\]//g" tmp > alexnet_conv_batch.result

rm tmp
