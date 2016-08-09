#ifndef SRC_CALLBACK_PROGRESSBAR_H_
#define SRC_CALLBACK_PROGRESSBAR_H_

#include "util/common.h"
#include "callback/callback.h"

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
};

}  // namespace blitz

#endif  // SRC_CALLBACK_PROGRESSBAR_H_
