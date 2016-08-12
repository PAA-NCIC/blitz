#ifndef SRC_INITIALIZER_INITIALIZER_H_
#define SRC_INITIALIZER_INITIALIZER_H_

#include <map>
#include <string>
#include <utility>

#include <boost/thread/once.hpp>
#include <boost/noncopyable.hpp>

#include "util/common.h"
#include "backend/gpu_tensor.h"
#include "backend/cpu_tensor.h"

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
  Initializer(const Initializer& initializer);
  Initializer& operator=(const Initializer& rhs);

  static scoped_ptr<Factory> instance_;
  static boost::once_flag flag_;
};

template<template <typename> class TensorType, typename DType>
class InitializerRegister {
 public:
  InitializerRegister(const string& data_type, const string& backend_type) {
    Initializer::Add<TensorType, DType>(data_type, backend_type);
  }
};

#ifdef BLITZ_CPU_ONLY
  #define INSTANTIATE_INITIALIZE \
      char BlitzInstantiatiionInitializerGuard; \
    template void Initializer::Initialize<CPUTensor, float>(const Parser& parser); \
    template void Initializer::Initialize<CPUTensor, double>(const Parser& parser)
  
  #define INSTANTIATE_ADD \
      char BlitzInstantiatiionAddGuard; \
    template void Initializer::Add<CPUTensor, float>(const string& data_type, \
        const string& backend); \
    template void Initializer::Add<CPUTensor, double>(const string& data_type, \
        const string& backend)
  
  #define REGISTER_INITIALIZER \
    INSTANTIATE_INITIALIZE; \
    INSTANTIATE_ADD; \
    static InitializerRegister<CPUTensor, float> initializer_float_CPU("float", "CPU"); \
    static InitializerRegister<CPUTensor, double> initializer_double_CPU("double", "CPU")
#else
  #define INSTANTIATE_INITIALIZE \
      char BlitzInstantiatiionInitializerGuard; \
    template void Initializer::Initialize<CPUTensor, float>(const Parser& parser); \
    template void Initializer::Initialize<CPUTensor, double>(const Parser& parser)
  
  #define INSTANTIATE_ADD \
      char BlitzInstantiatiionAddGuard; \
    template void Initializer::Add<CPUTensor, float>(const string& data_type, \
        const string& backend); \
    template void Initializer::Add<CPUTensor, double>(const string& data_type, \
        const string& backend); \
    template void Initializer::Add<GPUTensor, float>(const string& data_type, \
        const string& backend); \
    template void Initializer::Add<GPUTensor, double>(const string& data_type, \
        const string& backend)
  
  #define REGISTER_INITIALIZER \
    INSTANTIATE_INITIALIZE; \
    INSTANTIATE_ADD; \
    static InitializerRegister<CPUTensor, float> initializer_float_CPU("float", "CPU"); \
    static InitializerRegister<CPUTensor, double> initializer_double_CPU("double", "CPU"); \
    static InitializerRegister<GPUTensor, float> initializer_float_GPU("float", "GPU"); \
    static InitializerRegister<GPUTensor, double> initializer_double_GPU("double", "GPU")
#endif

}  // namespace blitz

#endif  // SRC_INITIALIZER_INITIALIZER_H_
