#ifndef SRC_UTIL_COMMON_H_
#define SRC_UTIL_COMMON_H_

// logging
#include <glog/logging.h>

#include <iostream>

#include <limits>
#include <list>
#include <map>
#include <vector>
#include <string>
#include <sstream>
#include <fstream>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <cmath>

// boost instead of c++11
// ptr
#include <boost/shared_ptr.hpp>
#include <boost/scoped_ptr.hpp>
#include <boost/make_shared.hpp>
#include <boost/pointer_cast.hpp>

// timer
#include <boost/chrono/chrono.hpp>
#include <boost/chrono/time_point.hpp>
#include <boost/chrono/system_clocks.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>
#include <boost/date_time/local_time_adjustor.hpp>
#include <boost/date_time/c_local_time_adjustor.hpp>

// algorithm
#include <boost/algorithm/string/trim.hpp>

// disable copy & assign
#define DISABLE_COPY_AND_ASSIGN(classname) \
  private:\
  classname(const classname&);\
  classname& operator=(const classname&)

// instaniate backends
#define INSTANTIATE_BACKEND(tensor) \
    char BlitzInstantiationBackendGuard##tensor; \
  template class Backend<tensor, float>; \
  template class Backend<tensor, double> \

// instaniate class
#ifdef BLITZ_CPU_ONLY
  #define INSTANTIATE_CLASS(classname) \
      char BlitzInstantiationClassGuard##classname; \
    template class classname<CPUTensor, float>; \
    template class classname<CPUTensor, double>
#else
  #define INSTANTIATE_CLASS(classname) \
      char BlitzInstantiationClassGuard##classname; \
    template class classname<CPUTensor, float>; \
    template class classname<CPUTensor, double>; \
    template class classname<GPUTensor, float>; \
    template class classname<GPUTensor, double>
#endif

// instaniate functions
#ifdef BLITZ_CPU_ONLY
#define INSTANTIATE_SETTER(object) \
    char BlitzInstantiatiionSetterGuard##object; \
  template shared_ptr<object<CPUTensor, float> > \
    Parser::Set##object<CPUTensor, float>(const YAML::Node& node) const; \
  template shared_ptr<object<CPUTensor, double> > \
    Parser::Set##object<CPUTensor, double>(const YAML::Node& node) const
#else
#define INSTANTIATE_SETTER(object) \
    char BlitzInstantiatiionSetterGuard##object; \
  template shared_ptr<object<CPUTensor, float> > \
    Parser::Set##object<CPUTensor, float>(const YAML::Node& node) const; \
  template shared_ptr<object<CPUTensor, double> > \
    Parser::Set##object<CPUTensor, double>(const YAML::Node& node) const; \
  template shared_ptr<object<GPUTensor, float> > \
    Parser::Set##object<GPUTensor, float>(const YAML::Node& node) const; \
  template shared_ptr<object<GPUTensor, double> > \
    Parser::Set##object<GPUTensor, double>(const YAML::Node& node) const
#endif

namespace blitz {

using boost::shared_ptr;
using boost::scoped_ptr;
using boost::make_shared;
using boost::static_pointer_cast;
using boost::chrono::time_point;
using boost::chrono::duration;
using boost::chrono::system_clock;
using boost::posix_time::ptime;
using boost::posix_time::second_clock;
using boost::posix_time::to_simple_string;
using boost::gregorian::day_clock;
using std::string;
using std::list;
using std::vector;
using std::map;
using std::size_t;
using std::stringstream;
using std::ofstream;

// TODO(keren) random seeding

}  //  namespace blitz

#endif  // SRC_UTIL_COMMON_H_
