#include <yaml-cpp/yaml.h>

#include <omp.h>
#include <mkl.h>

#include "initializer/initializer.h"
#include "initializer/parser.h"
#include "util/common.h"

void InitThreads() {
#ifdef BLITZ_NUM_THREADS
  omp_set_num_threads(BLITZ_NUM_THREADS);
  LOG(INFO) << "Number of threads: " << BLITZ_NUM_THREADS;
#else
  omp_set_num_threads(1);
  LOG(INFO) << "Number of threads: 1";
#endif
  // mkl_domain_set_num_threads(1, MKL_DOMAIN_BLAS);
}

void InitGlog(char** argv) {
  FLAGS_logtostderr = true;
  google::InitGoogleLogging(argv[0]);
}

int main(int argc, char** argv) {
  // glog init
  InitGlog(argv);
  // thread init
  InitThreads();

  if (argc != 2) {
    LOG(INFO) << "blitz <model_path>";
    LOG(FATAL) << "Check arguments, you must pass a model to blitz";
  }

  // yaml init
  blitz::string yaml_path = blitz::string(argv[1]);
  LOG(INFO) << "Load config from: " << yaml_path;
  const YAML::Node config = YAML::LoadFile(yaml_path);
  LOG(INFO) << "Load parser";
  blitz::Parser parser(config);
  parser.SetDefaultArgs();

  // model init
  const blitz::string& data_type = parser.data_type();
  const blitz::string& backend_type = parser.backend_type();
  LOG(INFO) << "Init model";
  blitz::Initializer::Run(data_type, backend_type, parser);

  return 0;
}
