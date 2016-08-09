#!/bin/python

import numpy as np
from compute_average import compute_average

if __name__ == '__main__':
  average = 10
  forward_layers = 6
  backward_layers = 6
  #total
  data = np.loadtxt("forward")
  forward = compute_average(forward_layers, average, data)
  data = np.loadtxt("backward")
  backward = compute_average(backward_layers, average, data)
  backward = backward[::-1] 

  #param layers
  forward_param_layers=4
  backward_param_layers=4
  #activation
  data = np.loadtxt("forward_activation")
  forward_activation = compute_average(forward_param_layers, average, data)
  data = np.loadtxt("backward_activation")
  backward_activation = compute_average(backward_param_layers, average, data)
  backward_activation = backward_activation[::-1] 

  #details
  #convolution
  forward_convolution_layers=2
  backward_convolution_layers=2
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
  backward_affine_layers=2
  data = np.loadtxt("backward_affine_gemm")
  backward_affine_gemm = compute_average(backward_affine_layers, average, data)
  backward_affine_gemm = backward_affine_gemm[::-1]

  data = np.loadtxt("backward_affine_weight_gemm")
  backward_affine_weight_gemm = compute_average(backward_affine_layers, average, data)
  backward_affine_weight_gemm = backward_affine_weight_gemm[::-1]

  #report
  batch_size=128

  forward_total_nointerval = np.array([forward[0], forward_activation[0], forward[1], forward[2], forward_activation[1], forward[3], forward[4], forward_activation[2], forward[5], forward_activation[3]])

  forward_total = np.array([forward[0], forward_convolution_unpack[0], forward_convolution_gemm[0], forward_activation[0], forward[1], forward[2], forward_convolution_unpack[1], forward_convolution_gemm[1], forward_activation[1], forward[3], forward[4], forward_activation[2], forward[5], forward_activation[3]])

  backward_total_nointerval = np.array([backward[0], backward_activation[0], backward[1], backward[2], backward_activation[1], backward[3], backward[4], backward_activation[2], backward[5], backward_activation[3]])

  backward_total = np.array([backward[0], backward_convolution_weight_unpack[0], backward_convolution_weight_gemm[0], backward_activation[0], backward[1], backward[2], backward_convolution_gemm[0], backward_convolution_pack[0], backward_convolution_weight_unpack[1], backward_convolution_weight_gemm[1], backward_activation[1], backward[3], backward[4], backward_affine_gemm[0], backward_affine_weight_gemm[0], backward_activation[2], backward[5], backward_affine_gemm[1], backward_affine_weight_gemm[1], backward_activation[3]])

  forward_total_computation_nointerval = np.array([batch_size * 460800, 4 * batch_size * 9216, 4 * batch_size * 2304, batch_size * 1638400, 4 * batch_size * 2048, 4 * batch_size * 512, batch_size * 512000, 4 * batch_size * 2000, batch_size * 10000, 4 * batch_size * 10])

  forward_total_computation = np.array([batch_size * 460800, 0, batch_size * 460800, 4 * batch_size * 9216, 4 * batch_size * 2304, batch_size * 1638400, 0, batch_size * 1638400, 4 * batch_size * 2048, 4 * batch_size * 512, batch_size * 512000, 4 * batch_size * 2000, batch_size * 10000, 4 * batch_size * 10])

  backward_total_computation_nointerval = np.array([batch_size * 86400, 4 * batch_size * 9216, 0, 2 * batch_size * 1638400, 4 * batch_size * 2048, 0, 2 * batch_size * 512000, 4 * batch_size * 2000, 2 * batch_size * 10000, 0])

  backward_total_computation = np.array([batch_size * 86400, 0, batch_size * 86400, 4 * batch_size * 9216, 0, 2 * batch_size * 1638400, batch_size * 1638400, 0, 0, batch_size * 1638400, 4 * batch_size * 2048, 0, 2 * batch_size * 512000, batch_size * 512000, batch_size * 512000, 4 * batch_size * 2000, 2 * batch_size * 10000, batch_size * 10000, batch_size * 10000, 0])

  print forward_convolution_unpack
  print forward_convolution_gemm
  print [0, backward_convolution_pack]
  print [0, backward_convolution_gemm]
  print backward_convolution_weight_unpack
  print backward_convolution_weight_gemm
  #print forward_total
  #print backward_total
  #print forward_total_computation
  #print backward_total_computation
  #print forward_total_nointerval.tolist()
  #print backward_total_nointerval.tolist()
  #print forward_total_computation_nointerval
  #print backward_total_computation_nointerval
  #print (forward_total_computation / forward_total / (665.6 * 1e9)).tolist()
  #print (backward_total_computation / backward_total / (665.6 * 1e9)).tolist()
  #print (forward_total_computation_nointerval / forward_total_nointerval / (665.6 * 1e9)).tolist()
  #print (backward_total_computation_nointerval / backward_total_nointerval / (665.6 * 1e9)).tolist()
