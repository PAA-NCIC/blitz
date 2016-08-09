#!/bin/bash

FILE=$1

#core times
FORWARD_CORE_TIME=(`grep "Forward core time" ${FILE} | rev | cut -d : -f1 | rev`)
BACKWARD_CORE_TIME=(`grep "Backward core time" ${FILE} | rev | cut -d : -f1 | rev`)
echo ${FORWARD_CORE_TIME[@]} > forward
echo ${BACKWARD_CORE_TIME[@]} > backward

#test
#echo ${FORWARD_CORE_TIME[@]}
#echo ${BACKWARD_CORE_TIME[@]}

#bias
FORWARD_BIAS_TIME=(`grep "Forward bias time" ${FILE} | rev | cut -d : -f1 | rev | awk '{printf("%.6f ", $0)}'`)
BACKWARD_BIAS_TIME=(`grep "Backward bias time" ${FILE} | rev | cut -d : -f1 | rev | awk '{printf("%.6f ", $0)}'`)
echo ${FORWARD_BIAS_TIME[@]} > forward_bias
echo ${BACKWARD_BIAS_TIME[@]} > backward_bias

#test
#echo ${FORWARD_BIAS_TIME[@]}
#echo ${BACKWARD_BIAS_TIME[@]}

#batch norm
FORWARD_BATCH_NORM_TIME=(`grep "Forward batch_norm time" ${FILE} | rev | cut -d : -f1 | rev | awk '{printf("%.6f ", $0)}'`)
BACKWARD_BATCH_NORM_TIME=(`grep "Backward batch_norm time" ${FILE} | rev | cut -d : -f1 | rev | awk '{printf("%.6f ", $0)}'`)
echo ${FORWARD_BATCH_NORM_TIME[@]} > forward_batch_norm
echo ${BACKWARD_BATCH_NORM_TIME[@]} > backward_batch_norm

#test
#echo ${FORWARD_BATCH_NORM_TIME[@]}
#echo ${BACKWARD_BATCH_NORM_TIME[@]}

#activation
FORWARD_ACTIVATION_TIME=(`grep "Forward activation time" ${FILE} | rev | cut -d : -f1 | rev | awk '{printf("%.6f ", $0)}'`)
BACKWARD_ACTIVATION_TIME=(`grep "Backward activation time" ${FILE} | rev | cut -d : -f1 | rev | awk '{printf("%.6f ", $0)}'`)
echo ${FORWARD_ACTIVATION_TIME[@]} > forward_activation
echo ${BACKWARD_ACTIVATION_TIME[@]} > backward_activation

#test
#echo ${FORWARD_ACTIVATION_TIME[@]}
#echo ${BACKWARD_ACTIVATION_TIME[@]}

FORWARD_CONVOLUTION_UNPACK_TIME=(`grep "Forward convolution.*unpack" ${FILE} | rev | cut -d : -f1 | rev | awk '{printf("%.6f ", $0)}'`)
BACKWARD_CONVOLUTION_PACK_TIME=(`grep "Backward convolution.*pack" ${FILE} | rev | cut -d : -f1 | rev | awk '{printf("%.6f ", $0)}'`)
BACKWARD_CONVOLUTION_WEIGHT_UNPACK_TIME=(`grep "Backward convolution weight.*unpack" ${FILE} | rev | cut -d : -f1 | rev | awk '{printf("%.6f ", $0)}'`)
echo ${FORWARD_CONVOLUTION_UNPACK_TIME[@]} > forward_convolution_unpack
echo ${BACKWARD_CONVOLUTION_PACK_TIME[@]} > backward_convolution_pack
echo ${BACKWARD_CONVOLUTION_WEIGHT_UNPACK_TIME[@]} > backward_convolution_weight_unpack

FORWARD_CONVOLUTION_GEMM_TIME=(`grep "Forward convolution.*gemm" ${FILE} | rev | cut -d : -f1 | rev | awk '{printf("%.6f ", $0)}'`)
BACKWARD_CONVOLUTION_GEMM_TIME=(`grep "Backward convolution.*gemm" ${FILE} | rev | cut -d : -f1 | rev | awk '{printf("%.6f ", $0)}'`)
BACKWARD_CONVOLUTION_WEIGHT_GEMM_TIME=(`grep "Backward convolution weight.*gemm" ${FILE} | rev | cut -d : -f1 | rev | awk '{printf("%.6f ", $0)}'`)
echo ${FORWARD_CONVOLUTION_GEMM_TIME[@]} > forward_convolution_gemm
echo ${BACKWARD_CONVOLUTION_GEMM_TIME[@]} > backward_convolution_gemm
echo ${BACKWARD_CONVOLUTION_WEIGHT_GEMM_TIME[@]} > backward_convolution_weight_gemm

BACKWARD_AFFINE_GEMM_TIME=(`grep "Backward affine time" ${FILE} | rev | cut -d : -f1 | rev | awk '{printf("%.6f ", $0)}'`)
BACKWARD_AFFINE_WEIGHT_GEMM_TIME=(`grep "Backward affine weight time" ${FILE} | rev | cut -d : -f1 | rev | awk '{printf("%.6f ", $0)}'`)
echo ${BACKWARD_AFFINE_GEMM_TIME[@]} > backward_affine_gemm
echo ${BACKWARD_AFFINE_WEIGHT_GEMM_TIME[@]} > backward_affine_weight_gemm

python $2

rm forward
rm backward
rm forward_bias
rm backward_bias
rm forward_batch_norm
rm backward_batch_norm
rm forward_activation
rm backward_activation
rm forward_convolution_unpack
rm backward_convolution_pack
rm backward_convolution_weight_unpack
rm forward_convolution_gemm
rm backward_convolution_gemm
rm backward_convolution_weight_gemm
rm backward_affine_gemm
rm backward_affine_weight_gemm
