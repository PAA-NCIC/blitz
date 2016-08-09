#include "callback/progressbar.h"

namespace blitz {

void Progressbar::OnEpochBegin(const int index) {
  // TODO(keren) progressbar
  LOG(INFO) << "Epoch: " << index << " begin";
}

void Progressbar::OnEpochEnd(const int index) {
  LOG(INFO) << "Epoch: " << index << " end";
}

void Progressbar::OnBatchBegin(const int index) {
  if (index % step_ == 0) {
    LOG(INFO) << "Batch: " << index << " begin";
  }
}

void Progressbar::OnBatchEnd(const int index, const float loss) {
  if (index % step_ == 0) {
    LOG(INFO) << "Loss: " << loss;
    LOG(INFO) << "Batch: " << index << " end";
  }
}

}  // namespace blitz

