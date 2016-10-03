#!/bin/bash

THREADS=`grep "BLITZ_NUM_THREADS" Makefile.config | cut -d " " -f3`
PINS=-c S0:0-7@S1:8-15

echo ${THREADS}

likwid-pin ${PINS} ./bin/blitz example/experiments/mnist_conv.yaml &> log/mnist_conv_"${THREADS}".log
echo "mnist_conv finish"
likwid-pin ${PINS} ./bin/blitz example/experiments/mnist_conv_batch.yaml &> log/mnist_conv_batch_"${THREADS}".log
echo "mnist_conv_batch finish"
likwid-pin ${PINS} ./bin/blitz example/experiments/cifar10_conv.yaml &> log/cifar10_conv_"${THREADS}".log
echo "cifar10_conv finish"
likwid-pin ${PINS} ./bin/blitz example/experiments/cifar10_conv_batch.yaml &> log/cifar10_conv_batch_"${THREADS}".log
echo "cifar10_conv_batch finish"
likwid-pin ${PINS} ./bin/blitz example/experiments/alexnet_conv.yaml &> log/alexnet_conv_"${THREADS}".log
echo "alexnet_conv finish"
likwid-pin ${PINS} ./bin/blitz example/experiments/alexnet_conv_batch.yaml &> log/alexnet_conv_batch_"${THREADS}".log
echo "alexnet_conv_batch finish"
