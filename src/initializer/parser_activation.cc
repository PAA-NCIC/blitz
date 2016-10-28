#include "initializer/parser.h"

#include "backends/backends.h"
#include "transforms/logistic.h"
#include "transforms/rectlin.h"
#include "transforms/softmax.h"

namespace blitz {

template<template <typename> class TensorType, typename DType>
shared_ptr<Activation<TensorType, DType> > Parser::SetActivation(const YAML::Node& node) const {
	shared_ptr<Activation<TensorType, DType> > activation;
	string type = node["type"].as<string>();

	if (type == "Rectlin") {
		activation = static_pointer_cast<Activation<TensorType, DType> >(
			make_shared<Rectlin<TensorType, DType> >());
	} else if (type == "Logistic") {
		bool short_cut;
		if (node["short_cut"]) {
			short_cut = node["short_cut"].as<bool>();
		} else {
			short_cut = false;
			LOG(WARNING) << "'short_cut' parameter missing";
		}
		activation = static_pointer_cast<Activation<TensorType, DType> >(
			make_shared<Logistic<TensorType, DType> >(short_cut));
	} else if (type == "Softmax") {
		bool short_cut;
		if (node["short_cut"]) {
			short_cut = node["short_cut"].as<bool>();
		} else {
			short_cut = false;
			LOG(WARNING) << "'short_cut' parameter missing";
		}
		activation = static_pointer_cast<Activation<TensorType, DType> >(
			make_shared<Softmax<TensorType, DType> >(short_cut));
	} else {
		LOG(FATAL) << "Unkown activation type: " << type;
	}

	return activation;
}

INSTANTIATE_SETTER_CPU(Activation);
#ifdef BLITZ_USE_GPU
  INSTANTIATE_SETTER_GPU(Activation);
#endif
#ifdef BLITZ_USE_MIC
  INSTANTIATE_SETTER_MIC(Activation);
#endif

}  // namespace blitz
