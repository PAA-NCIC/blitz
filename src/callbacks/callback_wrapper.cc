#include <list>

#include "callbacks/callback_wrapper.h"

namespace blitz {

void CallbackWrapper::OnEpochBegin(const size_t index) {
  for (list<shared_ptr<Callback> >::iterator it = callbacks_.begin();
      it != callbacks_.end(); ++it) {
    (*it)->OnEpochBegin(index);
  }
}

void CallbackWrapper::OnEpochEnd(const size_t index) {
  for (list<shared_ptr<Callback> >::iterator it = callbacks_.begin();
      it != callbacks_.end(); ++it) {
    (*it)->OnEpochEnd(index);
  }
}

void CallbackWrapper::OnBatchBegin(const size_t index) {
  for (list<shared_ptr<Callback> >::iterator it = callbacks_.begin();
      it != callbacks_.end(); ++it) {
    (*it)->OnBatchBegin(index);
  }
}

void CallbackWrapper::OnBatchEnd(const size_t index, const float loss) {
  for (list<shared_ptr<Callback> >::iterator it = callbacks_.begin();
      it != callbacks_.end(); ++it) {
    (*it)->OnBatchEnd(index, loss);
  }
}

}  // namespace blitz
