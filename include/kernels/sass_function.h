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

namespace kernels {

class CubinLoadModule {
 public:
  CubinLoadModule() {
    // init kernels list
    const size_t kernel_size = 13;
    const string kernel_name[kernel_size] = {
      "sgemm_nn_128x128",
      "sgemm_nt_128x128",
      "sgemm_tn_128x128",
      "sgemm_nn_128x128_vec",
      "sgemm_nt_128x128_vec",
      "sgemm_tn_128x128_vec",
      "sconv_fprop_K64_N64",
      "sconv_fprop_K128_N128",
      "sconv_bprop_C1_N64",
      "sconv_bprop_C64_N64",
      "sconv_bprop_C128_N128",
      "sconv_update_C128_K128",
      "sconv_update_C128_K64"
    };

    string version;
    {
      int major;
      CUdevice device;
      cuDeviceGet(&device, 0);
      cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device);
      if (major == 3) {
        version = "Kepler";
      } else if (major > 3) {
        version = "Pascal";
      }
    }

    for (size_t i = 0; i < kernel_size; ++i) {
      const string& name = kernel_name[i];
      const string path = "./cubin/" + version + "/" + name + ".cubin";

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
void SassGemm(
  const DType* A,
  const DType* B,
  DType* C,
  bool transa, bool transb,
  DType alpha,
  DType beta,
  size_t M, size_t N, size_t K);

template<typename DType>
void SassConvolution2DForward(
  DType *I, DType *O, DType *F,
  size_t N, size_t C, size_t H, size_t W,
  size_t R, size_t S,
  size_t K, size_t P, size_t Q,
  size_t pad_h, size_t pad_w,
  size_t str_h, size_t str_w);

template<typename DType>
void SassConvolution2DBackward(
  DType *I, DType *O, DType *F,
  size_t N, size_t C, size_t H, size_t W,
  size_t R, size_t S,
  size_t K, size_t P, size_t Q,
  size_t pad_h, size_t pad_w,
  size_t str_h, size_t str_w);

template<typename DType>
void SassConvolution2DUpdate(
  DType *I, DType *O, DType *F,
  size_t N, size_t C, size_t H, size_t W,
  size_t R, size_t S,
  size_t K, size_t P, size_t Q,
  size_t pad_h, size_t pad_w,
  size_t str_h, size_t str_w);

template<typename DType>
void Filter2DShuffle(
  const DType* input,
  DType* output,
  size_t K, size_t C, size_t R, size_t S);

}  // namespace kernels

}  // namespace blitz

#endif  // INCLUDE_KERNELS_SASS_FUNCTION_H_
