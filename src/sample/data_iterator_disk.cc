#include <yaml-cpp/yaml.h>

#include "initializer/parser.h"
#include "util/common.h"
#include "backend/cpu_tensor.h"

int main() {
  blitz::Parser parser(YAML::LoadFile("example/alexnet_conv_batch.yaml"));

  blitz::shared_ptr<blitz::DataIterator<blitz::CPUTensor, float> >
    data_iterator = parser.data_set<blitz::CPUTensor, float>();

  data_iterator->Init();

  // normal
  blitz::shared_ptr<blitz::CPUTensor<float> > tensor = data_iterator->GenerateTensor(0);

  // update index
  tensor = data_iterator->GenerateTensor(45);

  // update index
  tensor = data_iterator->GenerateTensor(90);

  // update index
  tensor = data_iterator->GenerateTensor(900);

  return 0;
}


