#ifndef INCLUDE_CALLBACK_PROGRESSBAR_H_
#define INCLUDE_CALLBACK_PROGRESSBAR_H_

#include "callback/callback.h"
#include "util/common.h"

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

#endif  // INCLUDE_CALLBACK_PROGRESSBAR_H_
