#include "callbacks/progressbar.h"

namespace blitz {

void Progressbar::OnEpochBegin(const size_t index) {
  // TODO(keren) progressbar
  LOG(INFO) << "Epoch: " << index << " begin";
}

void Progressbar::OnEpochEnd(const size_t index) {
  LOG(INFO) << "Epoch: " << index << " end";
}

void Progressbar::OnBatchBegin(const size_t index) {
  if (index % step_ == 0) {
    LOG(INFO) << "Batch: " << index << " begin";
  }
}

void Progressbar::OnBatchEnd(const size_t index, const float loss) {
  if (index % step_ == 0) {
    LOG(INFO) << "Loss: " << loss;
    LOG(INFO) << "Batch: " << index << " end";
  }
}

}  // namespace blitz

