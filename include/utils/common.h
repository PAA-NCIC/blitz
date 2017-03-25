#ifndef INCLUDE_UTIL_COMMON_H_
#define INCLUDE_UTIL_COMMON_H_

// logging
#include <glog/logging.h>
// standard libraries
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <list>
#include <map>
#include <string>
#include <sstream>
#include <vector>
// boost instead of c++11
// ptr
#include <boost/shared_ptr.hpp>
#include <boost/scoped_ptr.hpp>
#include <boost/make_shared.hpp>
#include <boost/pointer_cast.hpp>
// timer
#include <boost/date_time/posix_time/posix_time.hpp>
#include <boost/date_time/local_time_adjustor.hpp>
#include <boost/date_time/c_local_time_adjustor.hpp>
// algorithm
#include <boost/algorithm/string/trim.hpp>
// random
#include <boost/random.hpp>
#include <boost/generator_iterator.hpp>

// disable copy & assign
#define DISABLE_COPY_AND_ASSIGN(classname) \
  private:\
  classname(const classname&);\
  classname& operator=(const classname&)

// instaniate backends
#define INSTANTIATE_BACKEND(tensor) \
  char BlitzInstantiationBackendGuard##tensor; \
  template class Backend<tensor, float>

// instaniate class
#define INSTANTIATE_CLASS_CPU(classname) \
  char BlitzInstantiationClassCPUGuard##classname; \
  template class classname<CPUTensor, float>

#define INSTANTIATE_CLASS_GPU(classname) \
  char BlitzInstantiationClassGPUGuard##classname; \
  template class classname<GPUTensor, float>

// instaniate functions
#define INSTANTIATE_SETTER_CPU(object) \
  char BlitzInstantiatiionSetterCPUGuard##object; \
  template shared_ptr<object<CPUTensor, float> > \
    Parser::Set##object<CPUTensor, float>(const YAML::Node& node) const

#define INSTANTIATE_SETTER_GPU(object) \
  char BlitzInstantiatiionSetterGPUGuard##object; \
  template shared_ptr<object<GPUTensor, float> > \
    Parser::Set##object<GPUTensor, float>(const YAML::Node& node) const

namespace blitz {

using boost::shared_ptr;
using boost::scoped_ptr;
using boost::make_shared;
using boost::static_pointer_cast;
using boost::posix_time::ptime;
using boost::posix_time::second_clock;
using boost::posix_time::to_simple_string;
using boost::gregorian::day_clock;
using std::string;
using std::list;
using std::vector;
using std::map;
using std::stringstream;
using std::ofstream;

// TODO(keren) random seeding

}  // namespace blitz

#endif  // INCLUDE_UTIL_COMMON_H_
