#ifndef INCLUDE_CALLBACKS_CALLBACK_WRAPPER_H_
#define INCLUDE_CALLBACKS_CALLBACK_WRAPPER_H_

#include <list>

#include "utils/common.h"
#include "callbacks/callback.h"

namespace blitz {

class CallbackWrapper {
 public:
  explicit CallbackWrapper(const list<shared_ptr<Callback> >& callbacks) :
    callbacks_(callbacks) {}

  void OnEpochBegin(const size_t index);
  void OnEpochEnd(const size_t index);
  void OnBatchBegin(const size_t index);
  void OnBatchEnd(const size_t index, const float loss);

  // STL like function
  void push_back(shared_ptr<Callback> callback) {
    callbacks_.push_back(callback);
  }

 private:
  list<shared_ptr<Callback> > callbacks_;

  DISABLE_COPY_AND_ASSIGN(CallbackWrapper);
};

}  // namespace blitz

#endif  // INCLUDE_CALLBACKS_CALLBACK_WRAPPER_H_
