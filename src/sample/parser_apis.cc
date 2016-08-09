#include <yaml-cpp/yaml.h>
#include <iostream>

#include "initializer/parser.h"
#include "util/common.h"
#include "backend/cpu_tensor.h"

int main() {
  blitz::Parser parser(YAML::LoadFile("/home/zkr/codes/blitz/example/mnist_mlp.yaml"));
  parser.SetDefaultArgs();

  const blitz::string& data_path = parser.data_path();
  std::cout << "data_path :" << data_path << std::endl;
  const blitz::string& model_type = parser.model_type();
  std::cout << "model_type :" << model_type << std::endl;

  blitz::shared_ptr<blitz::LayerWrapper<blitz::CPUTensor, float> >
    layer_wrapper = parser.layer_wrapper<blitz::CPUTensor, float>();

  const int batch_size = parser.batch_size();
  std::cout << "batch_size :" << batch_size << std::endl;
  const int epoches = parser.epoches();
  std::cout << "epoches :" << epoches << std::endl;

  return 0;
}


