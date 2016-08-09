#!/bin/python

import numpy as np
from compute_average import compute_average

if __name__ == '__main__':
  average = 10
  forward_layers = 13
  backward_layers = 13
  #total
  data = np.loadtxt("forward")
  forward = compute_average(forward_layers, average, data)
  data = np.loadtxt("backward")
  backward = compute_average(backward_layers, average, data)
  backward = backward[::-1] 

  #param layers
  forward_param_layers = 8
  backward_param_layers = 8

  #bias
  #batch norm
  data = np.loadtxt("forward_bias")
  forward_bias = compute_average(forward_param_layers - 1, average, data)
  data = np.loadtxt("backward_bias")
  backward_bias = compute_average(backward_param_layers - 1, average, data)
  backward_bias = backward_bias[::-1] 

  #activation
  data = np.loadtxt("forward_activation")
  forward_activation = compute_average(forward_param_layers, average, data)
  data = np.loadtxt("backward_activation")
  backward_activation = compute_average(backward_param_layers, average, data)
  backward_activation = backward_activation[::-1] 

  #details
  #convolution
  forward_convolution_layers=5
  backward_convolution_layers=5
  data = np.loadtxt("forward_convolution_unpack")
  forward_convolution_unpack = compute_average(forward_convolution_layers, average, data)

  data = np.loadtxt("forward_convolution_gemm")
  forward_convolution_gemm = compute_average(forward_convolution_layers, average, data)

  data = np.loadtxt("backward_convolution_pack")
  backward_convolution_pack = compute_average(backward_convolution_layers - 1, average, data)
  backward_convolution_pack = backward_convolution_pack[::-1] 

  data = np.loadtxt("backward_convolution_gemm")
  backward_convolution_gemm = compute_average(backward_convolution_layers - 1, average, data)
  backward_convolution_gemm = backward_convolution_gemm[::-1]

  data = np.loadtxt("backward_convolution_weight_unpack")
  backward_convolution_weight_unpack = compute_average(backward_convolution_layers, average, data)
  backward_convolution_weight_unpack = backward_convolution_weight_unpack[::-1] 

  data = np.loadtxt("backward_convolution_weight_gemm")
  backward_convolution_weight_gemm = compute_average(backward_convolution_layers, average, data)
  backward_convolution_weight_gemm = backward_convolution_weight_gemm[::-1]

  #affine
  backward_affine_layers=3

  data = np.loadtxt("backward_affine_gemm")
  backward_affine_gemm = compute_average(backward_affine_layers, average, data)
  backward_affine_gemm = backward_affine_gemm[::-1]

  data = np.loadtxt("backward_affine_weight_gemm")
  backward_affine_weight_gemm = compute_average(backward_affine_layers, average, data)
  backward_affine_weight_gemm = backward_affine_weight_gemm[::-1]

  #report
  batch_size=128

  forward_total_nointerval = np.array([forward[0], forward_bias[0], forward_activation[0], forward[1], forward[2], forward_bias[1], forward_activation[1], forward[3], forward[4], forward_bias[2], forward_activation[2], forward[5], forward_bias[3], forward_activation[3], forward[6], forward_bias[4], forward_activation[4], forward[7], forward[8], forward_bias[5], forward_activation[5], forward[9], forward[10], forward_bias[6], forward_activation[6], forward[11], forward[12], forward_activation[7]])

  forward_total = np.array([forward[0], forward_convolution_unpack[0], forward_convolution_gemm[0], forward_bias[0], forward_activation[0], forward[1], forward[2], forward_convolution_unpack[1], forward_convolution_gemm[1], forward_bias[1], forward_activation[1], forward[3], forward[4], forward_convolution_unpack[2], forward_convolution_gemm[2], forward_bias[2], forward_activation[2], forward[5], forward_convolution_unpack[3], forward_convolution_gemm[3], forward_bias[3], forward_activation[3], forward[6], forward_convolution_unpack[4], forward_convolution_gemm[4], forward_bias[4], forward_activation[4], forward[7], forward[8], forward_bias[5], forward_activation[5], forward[9], forward[10], forward_bias[6], forward_activation[6], forward[11], forward[12], forward_activation[7]])

  backward_total_nointerval = np.array([backward[0], backward_bias[0], backward_activation[0], backward[1], backward[2], backward_bias[1], backward_activation[1], backward[3], backward[4], backward_bias[2], backward_activation[2], backward[5], backward_bias[3], backward_activation[3], backward[6], backward_bias[4], backward_activation[4], backward[7], backward[8], backward_bias[5], backward_activation[5], backward[9], backward[10], backward_bias[6], backward_activation[6], backward[11], backward[12], backward_activation[7]])

  backward_total = np.array([backward[0], backward_convolution_weight_unpack[0], backward_convolution_weight_gemm[0], backward_bias[0], backward_activation[0], backward[1], backward[2], backward_convolution_gemm[0], backward_convolution_pack[0], backward_convolution_weight_unpack[1], backward_convolution_weight_gemm[1], backward_bias[1], backward_activation[1], backward[3], backward[4], backward_convolution_gemm[1], backward_convolution_pack[1], backward_convolution_weight_unpack[2], backward_convolution_weight_gemm[2], backward_bias[2], backward_activation[2], backward[5], backward_convolution_gemm[2], backward_convolution_pack[2], backward_convolution_weight_unpack[3], backward_convolution_weight_gemm[3], backward_bias[3], backward_activation[3], backward[6], backward_convolution_gemm[3], backward_convolution_pack[3], backward_convolution_weight_unpack[4], backward_convolution_weight_gemm[4], backward_bias[4], backward_activation[4], backward[7], backward[8], backward_affine_gemm[0], backward_affine_weight_gemm[0], backward_bias[5], backward_activation[5], backward[9], backward[10], backward_affine_gemm[1], backward_affine_weight_gemm[1], backward_bias[6], backward_activation[6], backward[11], backward[12], backward_affine_gemm[2], backward_affine_weight_gemm[2], backward_activation[7]])

  forward_total_computation_nointerval = np.array([batch_size * 140553600, batch_size * 193600, 4 * batch_size * 193600, 9 * batch_size * 11664, batch_size * 447897600, batch_size * 139968, 4 * batch_size * 139968, 9 * batch_size * 139968, batch_size * 197413632, batch_size * 32448, 4 * batch_size * 32448, batch_size * 16613376, batch_size * 43264, 4 * batch_size * 43264, batch_size * 16613376, batch_size * 43264, batch_size * 43264, 9 * batch_size * 9216, batch_size * 75497472, batch_size * 4096, 4 * batch_size * 4096, 3 * batch_size * 4096, batch_size * 33554432, batch_size * 4096, 4 * batch_size * 4096, 3 * batch_size * 4096, batch_size * 8192000, 3 * batch_size * 1000])

  forward_total_computation = np.array([batch_size * 140553600, 0, batch_size * 140553600, batch_size * 193600, 4 * batch_size * 193600, 9 * batch_size * 11664, batch_size * 447897600, 0, batch_size * 447897600, batch_size * 139968, 4 * batch_size * 139968, 9 * batch_size * 139968, batch_size * 197413632, 0, batch_size * 98706816, batch_size * 32448, 4 * batch_size * 32448, batch_size * 16613376, 0, batch_size * 16613376, batch_size * 43264, 4 * batch_size * 43264, batch_size * 16613376, 0, batch_size * 16613376, batch_size * 43264, batch_size * 43264, 9 * batch_size * 9216, batch_size * 75497472, batch_size * 4096, 4 * batch_size * 4096, 3 * batch_size * 4096, batch_size * 33554432, batch_size * 4096, 4 * batch_size * 4096, 3 * batch_size * 4096, batch_size * 8192000, 3 * batch_size * 1000])

  forward_total_computation_convolution = np.array([batch_size * 140553600, batch_size * 447897600, batch_size * 197413632, batch_size * 16613376, batch_size * 16613376])

  backward_total_computation_nointerval = np.array([batch_size * 140553600, batch_size * 193600, 4 * batch_size * 193600, 0, 2 * batch_size * 447897600, batch_size * 139968, 4 * batch_size * 139968, 0, 2 * batch_size * 197413632, batch_size * 32448, 4 * batch_size * 32448, 2 * batch_size * 16613376, batch_size * 43264, 4 * batch_size * 43264, 2 * batch_size * 16613376, batch_size * 43264, 4 * batch_size * 43264, 0, 2 * batch_size * 75497472,batch_size * 4096, 4 * batch_size * 4096, batch_size * 4096, 2 * batch_size * 33554432,batch_size * 4096, 4 * batch_size * 4096, batch_size * 4096, 2 * batch_size * 8192000, 0])

  backward_total_computation = np.array([batch_size * 140553600, 0, batch_size * 140553600, batch_size * 193600, 4 * batch_size * 193600, 0, 2 * batch_size * 447897600, batch_size * 447897600, 0, 0, batch_size * 447897600, batch_size * 139968, 4 * batch_size * 139968, 0, 2 * batch_size * 98706816, batch_size * 98706816, 0, 0, batch_size * 98706816, batch_size * 32448, 4 * batch_size * 32448, 2 * batch_size * 16613376, batch_size * 16613376, 0, 0, batch_size * 16613376, batch_size * 43264, 4 * batch_size * 43264, 2 * batch_size * 16613376, batch_size * 16613376, 0, 0, batch_size * 16613376, batch_size * 43264, 4 * batch_size * 43264, 0, 2 * batch_size * 75497472, batch_size * 75497472, batch_size * 75497472, batch_size * 4096, 4 * batch_size * 4096, batch_size * 4096, 2 * batch_size * 33554432, batch_size * 33554432, batch_size * 33554432, batch_size * 4096, 4 * batch_size * 4096, batch_size * 4096, 2 * batch_size * 8192000, batch_size * 8192000, batch_size * 8192000, 0])

  #print forward_convolution_gemm[len(forward_convolution_gemm) - 1]
  #print backward_convolution_gemm[len(backward_convolution_gemm) - 1]
  #print backward_convolution_weight_gemm[len(backward_convolution_weight_gemm) - 1]
  #print forward_convolution_unpack
  #print forward_convolution_gemm
  #print [0, backward_convolution_pack]
  #print [0, backward_convolution_gemm]
  #print backward_convolution_weight_unpack
  #print backward_convolution_weight_gemm
  #print forward_total.tolist()
  #print backward_total.tolist()
  #print forward_total_nointerval.tolist()
  #print backward_total_nointerval.tolist()
  #print forward_total_computation
  #print backward_total_computation
  #print forward_total_computation_nointerval
  #print backward_total_computation_nointerval
  #print (forward_total_computation / forward_total / (665.6 * 1e9)).tolist()
  #print (backward_total_computation / backward_total / (665.6 * 1e9)).tolist()
  print (forward_total_computation_nointerval / forward_total_nointerval / (665.6 * 1e9)).tolist()
  print (backward_total_computation_nointerval / backward_total_nointerval / (665.6 * 1e9)).tolist()

