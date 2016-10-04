#ifndef INCLUDE_CALLBACK_CALLBACK_H_
#define INCLUDE_CALLBACK_CALLBACK_H_

#include "util/common.h"

namespace blitz {

class Callback {
 public:
  Callback() {}
  virtual ~Callback() {}

  virtual void OnEpochBegin(const int index) = 0;
  virtual void OnEpochEnd(const int index) = 0;
  virtual void OnBatchBegin(const int index) = 0;
  virtual void OnBatchEnd(const int index, const float loss) = 0;

  DISABLE_COPY_AND_ASSIGN(Callback);
};

}  // namespace blitz

#endif  // INCLUDE_CALLBACK_CALLBACK_H_

