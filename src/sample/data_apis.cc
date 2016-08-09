#include <yaml-cpp/yaml.h>
#include <iostream>

#include "initializer/parser.h"
#include "util/common.h"
#include "backend/cpu_tensor.h"

int main() {
  blitz::Parser parser(YAML::LoadFile("example/mnist_mlp.yaml"));

  const blitz::string& data_path = parser.data_path();
  std::cout << "data_path :" << data_path << std::endl;
  const blitz::string& model_type = parser.model_type();
  std::cout << "model_type :" << model_type << std::endl;

  blitz::shared_ptr<blitz::DataIterator<blitz::CPUTensor, float> >
    data_iterator = parser.data_label<blitz::CPUTensor, float>();

  data_iterator->Init();

  blitz::shared_ptr<blitz::CPUTensor<float> > tensor = data_iterator->GenerateTensor(0);

  const blitz::Shape& shape = tensor->shape();

  for (size_t i = 0; i < shape.dimension(); ++i) {
    std::cout << "dimension " << i << " : " << shape[i] << std::endl;
  }

  return 0;
}


