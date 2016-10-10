#ifndef INCLUDE_CALLBACKS_PROGRESSBAR_H_
#define INCLUDE_CALLBACKS_PROGRESSBAR_H_

#include "callbacks/callback.h"
#include "utils/common.h"

namespace blitz {

class Progressbar : public Callback {
 public:
  explicit Progressbar(const int step) : step_(step) {}
  ~Progressbar() {}

  virtual void OnEpochBegin(const int index);
  virtual void OnEpochEnd(const int index);
  virtual void OnBatchBegin(const int index);
  virtual void OnBatchEnd(const int index, const float loss);

 private:
  const int step_;

  DISABLE_COPY_AND_ASSIGN(Progressbar);
};

}  // namespace blitz

#endif  // INCLUDE_CALLBACKS_PROGRESSBAR_H_
