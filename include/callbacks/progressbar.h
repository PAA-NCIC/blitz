#ifndef INCLUDE_CALLBACKS_PROGRESSBAR_H_
#define INCLUDE_CALLBACKS_PROGRESSBAR_H_

#include "callbacks/callback.h"
#include "utils/common.h"

namespace blitz {

class Progressbar : public Callback {
 public:
  explicit Progressbar(const size_t step) : step_(step) {}
  ~Progressbar() {}

  virtual void OnEpochBegin(const size_t index);
  virtual void OnEpochEnd(const size_t index);
  virtual void OnBatchBegin(const size_t index);
  virtual void OnBatchEnd(const size_t index, const float loss);

 private:
  const size_t step_;

  DISABLE_COPY_AND_ASSIGN(Progressbar);
};

}  // namespace blitz

#endif  // INCLUDE_CALLBACKS_PROGRESSBAR_H_
