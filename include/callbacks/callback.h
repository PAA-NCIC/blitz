#ifndef INCLUDE_CALLBACKS_CALLBACK_H_
#define INCLUDE_CALLBACKS_CALLBACK_H_

#include "utils/common.h"

namespace blitz {

class Callback {
 public:
  Callback() {}
  virtual ~Callback() {}

  virtual void OnEpochBegin(const size_t index) = 0;
  virtual void OnEpochEnd(const size_t index) = 0;
  virtual void OnBatchBegin(const size_t index) = 0;
  virtual void OnBatchEnd(const size_t index, const float loss) = 0;

  DISABLE_COPY_AND_ASSIGN(Callback);
};

}  // namespace blitz

#endif  // INCLUDE_CALLBACKS_CALLBACK_H_
