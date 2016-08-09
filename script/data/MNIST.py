import gzip
import numpy as np
import os
import struct
import sys
import h5py

PY3 = (sys.version_info[0] >= 3)

# keep range calls consistent between python 2 and 3
# note: if you need a list and not an iterator you can do list(range(x))
range = range
if not PY3:
  range = xrange

class DataRead(object):
  raw_train_input_gz = 'train-images-idx3-ubyte.gz'
  raw_train_target_gz = 'train-labels-idx1-ubyte.gz'
  raw_test_input_gz = 't10k-images-idx3-ubyte.gz'
  raw_test_target_gz = 't10k-labels-idx1-ubyte.gz'

  inputs = {'train': None, 'test': None, 'validation': None}
  targets = {'train': None, 'test': None, 'validation': None}

  repo_path = './data/'

  num_test_sample = 10000
  num_train = 60000 #this is very important, at present you must set it mannually
  num_test = 10000
  heights = 0
  widths = 0

  def save_hdf5(self):
    self.repo_path = os.path.expandvars(os.path.expanduser(
      self.repo_path))
    save_dir = os.path.join(self.repo_path,
        'HDF5')
    if not os.path.exists(save_dir):
      os.makedirs(save_dir)

      fname = os.path.join(save_dir, 'train_data.h5')
      file_train_data = h5py.File(fname, 'w')

      fname = os.path.join(save_dir, 'train_label.h5')
      file_train_label = h5py.File(fname, 'w')

      fname = os.path.join(save_dir, 'test_data.h5')
      file_test_data = h5py.File(fname, 'w')

      fname = os.path.join(save_dir, 'test_label.h5')
      file_test_label = h5py.File(fname, 'w')

      file_train_data.create_dataset('data', data = self.inputs['train'])
      file_train_data.create_dataset('sample_num', data = self.num_train)
      #note it is very important, the targets is 60000*10, not 60000*1
      file_train_label.create_dataset('data', data = self.targets['train'])
      file_train_label.create_dataset('sample_num', data = self.num_train)
      file_test_data.create_dataset('data', data = self.inputs['test'])
      file_test_data.create_dataset('sample_num', data = self.num_test)
      file_test_label.create_dataset('data', data = self.targets['test'])
      file_test_label.create_dataset('sample_num', data = self.num_test)

      file_train_data.close()
      file_train_label.close()
      file_test_data.close()
      file_test_label.close()

  def read_image_file(self, fname, dtype=None):
    """
      Carries out the actual reading of MNIST image files.
      """
      with open(fname, 'rb') as f:
        magic, num_images, rows, cols = struct.unpack('>iiii', f.read(16))
          self.widths = rows
          self.heights = cols
          if magic != 2051:
            raise ValueError('invalid MNIST image file: ' + fname)
          full_image = np.fromfile(f, dtype='uint8').reshape((num_images,
            rows * cols))

          if dtype is not None:
            dtype = np.dtype(dtype)
          full_image = full_image.astype(dtype)
          full_image /= 255.

      return full_image

  def read_label_file(self, fname):
    """
      Carries out the actual reading of MNIST label files.
      """
      with open(fname, 'rb') as f:
        magic, num_labels = struct.unpack('>ii', f.read(8))
          self.num_test = num_labels
          if magic != 2049:
            raise ValueError('invalid MNIST label file:' + fname)
          array = np.fromfile(f, dtype='uint8')
      return array

  def load(self):
    if True:
      self.repo_path = os.path.expandvars(os.path.expanduser(
        self.repo_path))
      save_dir = os.path.join(self.repo_path,
          self.__class__.__name__)
      if not os.path.exists(save_dir):
        os.makedirs(save_dir)

          for url in (self.raw_train_input_gz, self.raw_train_target_gz,
              self.raw_test_input_gz, self.raw_test_target_gz):
            name = os.path.basename(url).rstrip('.gz')
              repo_gz_file = os.path.join(self.repo_path, name + '.gz')
              repo_gz_file_second = os.path.join(save_dir, name + '.gz')
              repo_file = repo_gz_file_second.rstrip('.gz')
              with gzip.open(repo_gz_file, 'rb') as infile:
                with open(repo_file, 'w') as outfile:
                  for line in infile:
                    outfile.write(line)
              if 'images' in repo_file and 'train' in repo_file:
                indat = self.read_image_file(repo_file, 'float32')
                  # flatten to 1D images
                  self.inputs['train'] = indat
              elif 'images' in repo_file and 't10k' in repo_file:
                indat = self.read_image_file(repo_file, 'float32')
                  self.inputs['test'] = indat[0:self.num_test_sample]
              elif 'labels' in repo_file and 'train' in repo_file:
                indat = self.read_label_file(repo_file)
                  # Prep a 1-hot label encoding
                  tmp = np.zeros((indat.shape[0], 10))
                  for col in range(10):
                    tmp[:, col] = indat == col
                  self.targets['train'] = tmp
              elif 'labels' in repo_file and 't10k' in repo_file:
                indat = self.read_label_file(
                    repo_file)[0:self.num_test_sample]
                tmp = np.zeros((self.num_test_sample, 10))
                  for col in range(10):
                    tmp[:, col] = indat == col
                  self.targets['test'] = tmp
              else:
                print ('problems loading: %s', name)

if __name__ == '__main__':
  dr = DataRead()
  dr.load()
  dr.save_hdf5()
