#!/bin/python

def compute_average(layers, average, data):
  ret = []
  for i in range(layers):
    var = 0
    for j in range(average):
      index = j * layers + i
      var += data[index]
    var /= average
    ret.append(var) 

  return ret
