#ifndef INCLUDE_KERNELS_SASS_FUNCTION_H_
#define INCLUDE_KERNELS_SASS_FUNCTION_H_

#include <cuda.h>
#include <boost/thread/once.hpp>
#include <boost/noncopyable.hpp>

#include <map>
#include <string>
#include <vector>
#include <utility>

#include "utils/common.h"

namespace blitz {

class CubinLoadModule {
 public:
  CubinLoadModule() {
    // init kernels list
    const size_t kernel_size = 11;
    const string kernel_name[kernel_size] = {
      "sgemm_nn_128x128",
      "sgemm_nt_128x128",
      "sgemm_tn_128x128",
      "sgemm_nn_128x128_vec",
      "sgemm_nt_128x128_vec",
      "sgemm_tn_128x128_vec",
      "sconv_fprop_K64_N64",
      "sconv_bprop_C1_N64",
      "sconv_bprop_C64_N64",
      "sconv_update_C128_K128",
      "sconv_update_C128_K64"
    };

    for (size_t i = 0; i < kernel_size; ++i) {
      const string& name = kernel_name[i];
      const string path = "./cubin/" + name + ".cubin";

      CUmodule module;
      CUfunction function;
      CUresult res;

      // load module
      res = cuModuleLoad(&module, path.c_str());
      if (res != CUDA_SUCCESS) {
        LOG(FATAL) << "Failed to load module: " << name << std::endl;
      }

      // load function
      res = cuModuleGetFunction(&function, module, name.c_str());
      if (res != CUDA_SUCCESS) {
        LOG(FATAL) << "Failed to load function: " << name << std::endl;
      }

      functions_[name] = function;
      modules_.push_back(module);
    }
  }

  ~CubinLoadModule() {
    typedef vector<CUmodule>::iterator ModuleIterator;
    for (ModuleIterator it = modules_.begin(); it != modules_.end(); ++it) {
      cuModuleUnload(*it);
    }
  }

  CUfunction GetFunction(const string& name) {
    if (functions_.find(name) == functions_.end()) {
      LOG(FATAL) << "Cannot find kernel: " << name;
    }

    return functions_[name];
  }

 private:
  map<string, CUfunction> functions_;
  vector<CUmodule> modules_;

  DISABLE_COPY_AND_ASSIGN(CubinLoadModule);
};

class CubinModule {
 public:
  static CubinLoadModule& GetInstance() {
    // thread safe
    boost::call_once(&CubinModule::Create, flag_);
    return *(CubinModule::instance_);
  }

  static void Create() {
    CubinModule::instance_.reset(new CubinLoadModule());
  }

  static CUfunction GetFunction(const string& name) {
    CubinLoadModule& cubin_load_module = CubinModule::GetInstance();
    return cubin_load_module.GetFunction(name);
  }

  virtual ~CubinModule();

 private:
  CubinModule();

  static scoped_ptr<CubinLoadModule> instance_;
  static boost::once_flag flag_;

  DISABLE_COPY_AND_ASSIGN(CubinModule);
};

template<typename DType>
void BlitzSassGemm(
  const DType* A,
  const DType* B,
  DType* C,
  bool transa, bool transb,
  DType alpha,
  DType beta,
  size_t M, size_t N, size_t K);

template<typename DType>
void BlitzSassConvolution2D(
  DType* input,
  DType* output,
  DType* filter,
  size_t batch_size,
  size_t input_channel,
  size_t input_height, size_t input_width,
  size_t filter_height, size_t filter_width,
  size_t output_channel,
  size_t output_height, size_t output_width,
  size_t stride_height, size_t stride_width,
  size_t padding_height, size_t padding_width,
  const string& phase);

template<typename DType>
void BlitzFilter2DShuffle(
  const DType* input,
  DType* output,
  size_t K, size_t C, size_t R, size_t S);

}  // namespace blitz

#endif  // INCLUDE_KERNELS_SASS_FUNCTION_H_
