#include "initializer/parser.h"

#include <string>
#include <vector>

#include "callbacks/progressbar.h"

namespace blitz {

void Parser::SetDefaultArgs() {
  if (!config_["data_type"]) {
    config_["data_type"] = "float";
  }

  if (!config_["backend_type"]) {
    config_["backend_type"] = "CPU";
  }

  if (!config_["epoches"]) {
    config_["epoches"] = 10;
  }

  if (!config_["batch_size"]) {
    config_["batch_size"] = 128;
  }
}

shared_ptr<Callback> Parser::SetCallback(const YAML::Node& node) const {
  shared_ptr<Callback> callback;
  string type = node["type"].as<string>();

  if (type == "Progressbar") {
    if (node["step"]) {
      int step = node["step"].as<int>();
      callback = static_pointer_cast<Callback>(make_shared<Progressbar>(step));
    } else {
      LOG(FATAL) << "'step' parameter missing";
    }
  } else {
    LOG(FATAL) << "Unkown callback type: " << type;
  }

  return callback;
}

}  // namespace blitz
