// YAML-CPP samples
// for more specific examples, see:
// https://github.com/jbeder/yaml-cpp/wiki/Tutorial
#include <yaml-cpp/yaml.h>

#include <iostream>

int main() {
  // load mnist yaml
  YAML::Node config = YAML::LoadFile("../../example/mnist_mlp.yaml");

  // node types
  switch (config.Type()) {
    case YAML::NodeType::Sequence:
      std::cout << "Sequence" << std::endl;
      break;
    case YAML::NodeType::Scalar:
      std::cout << "Scalar" << std::endl;
      break;
    case YAML::NodeType::Map:
      std::cout << "Map" << std::endl;
      break;
    case YAML::NodeType::Undefined:
      std::cout << "Undefined" << std::endl;
      break;
    default:
      std::cout << "NULL" << std::endl;
      break;
  }

  // read map objects
  if (config["cost"]) {
    std::cout << config["cost"] << std::endl;
  }

  // convert to basic type
  if (config["epochs"]) {
    std::cout << config["epochs"] << std::endl;
  }

  // subnode
  if (config["optimizer"]) {
    YAML::Node optimizer = config["optimizer"];
    std::cout << optimizer["type"] << std::endl;
  }

  // sequence
  if (config["layers"]) {
    YAML::Node layers = config["layers"];
    for (std::size_t i = 0; i < layers.size(); ++i) {
      YAML::Node layer = layers[i];
      std::cout << layer["type"] << i << std::endl;
    }
  }

  // reference
  if (config["layers"]) {
    YAML::Node layers = config["layers"];
    for (std::size_t i = 0; i < layers.size(); ++i) {
      YAML::Node layer = layers[i];
      if (layer["init"] == config["wt_init"]) {
        std::cout << "reference equal" << i << std::endl;
      }
    }
  }

  // composite
  std::vector<YAML::Node> layers = config["layers"].as<std::vector<YAML::Node> >();


  return 0;
}

