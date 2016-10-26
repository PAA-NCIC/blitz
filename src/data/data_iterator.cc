#include "data/data_iterator.h"

#include <hdf5.h>
#include <fstream>

#include "backends/backends.h"
#include "utils/blitz_cpu_function.h"

namespace blitz {

template<template <typename> class TensorType, typename DType>
void DataIterator<TensorType, DType>::Init() {
  std::ifstream hdf5_files(data_path_.c_str());

  if (hdf5_files.is_open()) {
    string file;
    while (hdf5_files >> file) {
      files_.push_back(file);
    }
  } else {
    LOG(FATAL) << "Hdf5 file open error: " << data_path_;
  }
  hdf5_files.close();

  int num_sample;
  herr_t status;
  for (size_t i = 0; i < files_.size(); ++i) {
    int file_id = H5Fopen(files_[i].c_str(), H5F_ACC_RDWR, H5P_DEFAULT);
    int sample_id = H5Dopen2(file_id, "sample_num", H5P_DEFAULT);

    status = H5Dread(sample_id, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT,
        &num_sample);
    CHECK_GE(status, 0);

    status = H5Dclose(sample_id);
    CHECK_GE(status, 0);

    status = H5Fclose(file_id);
    CHECK_GE(status, 0);

    file_row_mapping_.push_back(total_);
    total_ += num_sample;
  }
  file_row_mapping_.push_back(total_);

  tensor_pool_.resize(pool_size_);
  CopyFileBuffer(0);
}

template<template <typename> class TensorType, typename DType>
void DataIterator<TensorType, DType>::CopyFileBuffer(int begin_offset) {
  vector<string> current_files;
  vector<int> current_files_row_mapping;
  int end_offset = begin_offset + pool_size_ * batch_size_;
  int begin_file_offset = 0;
  int end_file_offset = 0;
  int cur_file_offset = 0;
  int accumulate = 0;

  bool find_begin = false, find_end = false;
  int exit_index = 0;
  // find all files in the interval [begin_offset, end_offset]
  // end_offset maybe greater than total_
  // otherwise record the last file
  // TODO(keren): binary search optimization
  for (size_t i = 0; i < file_row_mapping_.size() - 1; ++i) {
    if (begin_offset >= file_row_mapping_[i] &&
      begin_offset < file_row_mapping_[i + 1] && !find_begin) {
      find_begin = true;
      begin_file_offset = begin_offset - file_row_mapping_[i];
      cur_file_offset = begin_file_offset;
    } else {
      cur_file_offset = 0;
    }

    if (find_begin) {
      current_files_row_mapping.push_back(file_row_mapping_[i + 1] -
        file_row_mapping_[i]);
      accumulate += *(current_files_row_mapping.rbegin()) -
        cur_file_offset;
      current_files.push_back(files_[i]);
    }

    if (end_offset >= file_row_mapping_[i] &&
      end_offset < file_row_mapping_[i + 1] && !find_end) {
      find_end = true;
      end_file_offset = end_offset - file_row_mapping_[i];
    }

    if (find_begin && find_end) {
      exit_index = i;
      break;
    }
  }
  // there are at least two indice in file_row_mapping:
  // the first file and its index
  if (find_end) {
    accumulate -= file_row_mapping_[exit_index + 1] - end_offset;
  } else {
    end_file_offset = total_ - file_row_mapping_[file_row_mapping_.size() - 2];
  }

  // create a temporary buffer to copy into memory
  size_t input_size = input_shape_.size() / input_shape_[0];
  DType* file_data = new DType[accumulate * input_size];
  int file_data_offset_size = 0;
  for (size_t i = 0; i < current_files.size(); ++i) {
    int file_id = H5Fopen(current_files[i].c_str(), H5F_ACC_RDWR, H5P_DEFAULT);
    int data_id = H5Dopen2(file_id, "data", H5P_DEFAULT);
    int file_size = current_files_row_mapping[i] * input_size;
    DType* current_files_data = new DType[file_size];

    herr_t status;
    status = H5Dread(data_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT,
      current_files_data);
    CHECK_GE(status, 0);

    status = H5Dclose(data_id);
    CHECK_GE(status, 0);

    status = H5Fclose(file_id);
    CHECK_GE(status, 0);

    int copy_size = 0;
    int current_files_data_offset = 0;
    // calculate copy_size
    if (i == 0) {
      current_files_data_offset = begin_file_offset;
      if (current_files.size() == 1) {
        copy_size = (end_file_offset - begin_file_offset) * input_size;
      } else {
        copy_size = (current_files_row_mapping[i] -
          begin_file_offset) * input_size;
      }
    } else if (current_files.size() - 1 == i) {
      copy_size = end_file_offset * input_size;
    }

    BlitzCPUCopy(current_files_data + current_files_data_offset,
      file_data + file_data_offset_size, copy_size);
    file_data_offset_size += copy_size;

    delete [] current_files_data;
  }

  // construct tensor
  int tensor_offset_size = 0;
  // only in the last pool_size_, remaining samples are ignored
  for (int j = 0; j < accumulate / batch_size_; ++j) {
    shared_ptr<TensorType<DType> > tensor =
      make_shared<TensorType<DType> >(input_shape_);
    Backend<TensorType, DType>::HostCopyToFunc(file_data + tensor_offset_size,
      tensor->data(), input_shape_.size());
    tensor_pool_[j] = tensor;
    tensor_offset_size += input_shape_.size();
  }

  delete [] file_data;
}

template<template <typename> class TensorType, typename DType>
shared_ptr<TensorType<DType> > DataIterator<TensorType, DType>
  ::GenerateTensor(int index) {
  if (index >= total_) {
    LOG(FATAL) << "Index out of range: " <<
      "index " << index << " total " << total_;
  } else if (index >= current_begin_index_ &&
      index < current_begin_index_ + pool_size_) {
  } else if (index >= 0) {
    // udpate current_begin_index_;
    CopyFileBuffer(index * batch_size_);
    current_begin_index_ = index;
    LOG(INFO) << "Update tensor index to: " << index;
  } else {
    LOG(FATAL) << "Index negative: " << "index " << index;
  }

  return tensor_pool_[index - current_begin_index_];
}

INSTANTIATE_CLASS(DataIterator);

}  // namespace blitz
