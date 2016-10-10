#include <list>

#include "callbacks/callback_wrapper.h"

namespace blitz {

void CallbackWrapper::OnEpochBegin(const int index) {
  for (list<shared_ptr<Callback> >::iterator it = callbacks_.begin();
      it != callbacks_.end(); ++it) {
    (*it)->OnEpochBegin(index);
  }
}

void CallbackWrapper::OnEpochEnd(const int index) {
  for (list<shared_ptr<Callback> >::iterator it = callbacks_.begin();
      it != callbacks_.end(); ++it) {
    (*it)->OnEpochEnd(index);
  }
}

void CallbackWrapper::OnBatchBegin(const int index) {
  for (list<shared_ptr<Callback> >::iterator it = callbacks_.begin();
      it != callbacks_.end(); ++it) {
    (*it)->OnBatchBegin(index);
  }
}

void CallbackWrapper::OnBatchEnd(const int index, const float loss) {
  for (list<shared_ptr<Callback> >::iterator it = callbacks_.begin();
      it != callbacks_.end(); ++it) {
    (*it)->OnBatchEnd(index, loss);
  }
}

}  // namespace blitz
