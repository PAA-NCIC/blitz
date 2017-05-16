#ifndef INCLUDE_INITIALIZER_INITIALIZER_H_
#define INCLUDE_INITIALIZER_INITIALIZER_H_

#include <map>
#include <string>
#include <utility>

#include <boost/thread/once.hpp>
#include <boost/noncopyable.hpp>

#include "utils/common.h"
#include "backends/gpu_tensor.h"
#include "backends/cpu_tensor.h"

namespace blitz {

class Parser;

// singleton
class Initializer {
 public:
  typedef void (*Runner)(const Parser& parser);
  typedef map<std::pair<string, string>, Runner> Factory;

  static Factory& GetInstance() {
    // thread safe
    boost::call_once(&Initializer::Create, flag_);
    return *(Initializer::instance_);
  }

  static void Create() {
    Initializer::instance_.reset(new Factory());
  }

  template<template <typename> class TensorType, typename DType>
  static void Initialize(const Parser& parser);

  template<template <typename> class TensorType, typename DType>
  static void Add(const string& data_type, const string& backend_type) {
    Factory& factory = Initializer::GetInstance();
    // TODO(keren) check exist
    factory[std::make_pair(data_type, backend_type)] =
      Initializer::Initialize<TensorType, DType>;
  }

  static void Run(const string& data_type, const string& backend_type,
      const Parser& parser) {
    Factory& factory = Initializer::GetInstance();
    // TODO(keren) check not exist
    factory[std::make_pair(data_type, backend_type)](parser);
  }

  virtual ~Initializer();  // without implementation

 private:
  Initializer();

  static scoped_ptr<Factory> instance_;
  static boost::once_flag flag_;
  
  DISABLE_COPY_AND_ASSIGN(Initializer);
};

template<template <typename> class TensorType, typename DType>
class InitializerRegister {
 public:
  InitializerRegister(const string& data_type, const string& backend_type) {
    Initializer::Add<TensorType, DType>(data_type, backend_type);
  }

  DISABLE_COPY_AND_ASSIGN(InitializerRegister);
};

#define INSTANTIATE_INITIALIZE_CPU \
  char BlitzInstantiatiionInitializerCPUGuard; \
  template void Initializer::Initialize<CPUTensor, float>(const Parser& parser)

#define INSTANTIATE_INITIALIZE_GPU \
  char BlitzInstantiatiionInitializerGPUGuard; \
  template void Initializer::Initialize<GPUTensor, float>(const Parser& parser)
  
#define INSTANTIATE_ADD_CPU \
  char BlitzInstantiatiionAddCPUGuard; \
  template void Initializer::Add<CPUTensor, float>(const string& data_type, \
    const string& backend) 

#define INSTANTIATE_ADD_GPU \
  char BlitzInstantiatiionAddGPUGuard; \
  template void Initializer::Add<GPUTensor, float>(const string& data_type, \
    const string& backend)

#define REGISTER_INITIALIZER_CPU \
  INSTANTIATE_INITIALIZE_CPU; \
  INSTANTIATE_ADD_CPU; \
  static InitializerRegister<CPUTensor, float> initializer_float_CPU("float", "CPU")

#define REGISTER_INITIALIZER_GPU \
  INSTANTIATE_INITIALIZE_GPU; \
  INSTANTIATE_ADD_GPU; \
  static InitializerRegister<CPUTensor, float> initializer_float_GPU("float", "GPU")

}  // namespace blitz

#endif  // INCLUDE_INITIALIZER_INITIALIZER_H_
