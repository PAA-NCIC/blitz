#include "data/data_iterator.h"
#include "backend/backend.h"
#include "backend/cpu_tensor.h"
#include "backend/gpu_tensor.h"
#include <fstream>
#include <hdf5.h>

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
    //TODO: LOG FATAL
  }
  hdf5_files.close();

  int num_sample;
  for (size_t i = 0; i < files_.size(); ++i) {
    int file_id = H5Fopen(files_[i].c_str(), H5F_ACC_RDWR, H5P_DEFAULT); 
    int sample_id = H5Dopen2(file_id, "sample_num", H5P_DEFAULT);
    //TODO: status check
    herr_t status;
    status = H5Dread(sample_id, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, &num_sample); 
    status = H5Dclose(sample_id);
    status = H5Fclose(file_id);

    file_row_mapping_.push_back(num_sample);
    total_ += num_sample;
  }
}

template<template <typename> class TensorType, typename DType>
shared_ptr<TensorType<DType> > DataIterator<TensorType, DType>::GenerateTensor(int index) {
  vector<string> current_file;
  vector<int> current_file_size;
  int accumulate_rows = 0;
  int begin_offset = 0;
  int end_offset = 0;
  //find files list to read for current batch index
  //TODO file pool, save the last file in the memory

  for (size_t i = 0; i < file_row_mapping_.size(); ++i) {
    if (index * batch_size_ >= accumulate_rows) {
      current_file.push_back(files_[i]);
      current_file_size.push_back(file_row_mapping_[i]);

      begin_offset = index * batch_size_ - accumulate_rows;
      if ((index + 1)* batch_size_ < file_row_mapping_[i] + accumulate_rows) 
        end_offset = (index + 1) * batch_size_ - accumulate_rows;
        break;    
    }
    accumulate_rows += file_row_mapping_[i];
  } 

  Shape shape(2);
  shape.at(0) = batch_size_;
  shape.at(1) = data_size_; 
  shared_ptr<TensorType<DType> > tensor = make_shared<TensorType<DType> >(shape);
  int tensor_offset = 0;
  for (size_t i = 0; i < current_file.size(); ++i) {
    int file_id = H5Fopen(current_file[i].c_str(), H5F_ACC_RDWR, H5P_DEFAULT); 
    int data_id = H5Dopen2(file_id, "data", H5P_DEFAULT);
    shared_ptr<DType> file_data = shared_ptr<DType>(new DType[current_file_size[i] * data_size_], 
        default_delete<DType[]>());

    //TODO: status check
    herr_t status;
    status = H5Dread(data_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, file_data.get());
    status = H5Dclose(data_id);
    status = H5Fclose(file_id);

    int copy_size = 0;
    int current_begin_offset = 0;
    if (i == 0) {
      if (current_file.size() - 1 == 0) {
        copy_size = end_offset - begin_offset;
      } else {
        copy_size = current_file_size[i] - begin_offset;
      }
      current_begin_offset = begin_offset;
    } else if (current_file.size() - 1 == i) {
      copy_size = end_offset;
    }

    Backend<TensorType, DType>::CopyFunc(tensor->data(), tensor_offset, file_data.get(), 
        current_begin_offset, copy_size * data_size_);
    tensor_offset += copy_size * data_size_;
  }

  return tensor;
}

INSTANTIATE_CLASS(DataIterator);

}// namespace blitz
