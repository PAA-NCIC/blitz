import os
import sys
import cPickle
import h5py
import numpy as np

def _valid_path_append(path, *args):
    """
    Helper to validate passed path directory and append any subsequent
    filename arguments.

    Arguments:
        path (str): Initial filesystem path.  Should expand to a valid
                    directory.
        *args (list, optional): Any filename or path suffices to append to path
                                for returning.

    Returns:
        (list, str): path prepended list of files from args, or path alone if
                     no args specified.

    Raises:
        ValueError: if path is not a valid directory on this filesystem.
    """
    full_path = os.path.expanduser(path)
    res = []
    if not os.path.exists(full_path):
        os.makedirs(full_path)
    if not os.path.isdir(full_path):
        raise ValueError("path: {0} is not a valid directory".format(path))
    for suffix_path in args:
        res.append(os.path.join(full_path, suffix_path))
    if len(res) == 0:
        return path
    elif len(res) == 1:
        return res[0]
    else:
        return res




def load_cifar10(path="./data", normalize=True, contrast_normalize=False, whiten=False):
    """
    Fetch the CIFAR-10 dataset and load it into memory.

    Args:
        path (str, optional): Local directory in which to cache the raw
                              dataset.  Defaults to current directory.
        normalize (bool, optional): Whether to scale values between 0 and 1.
                                    Defaults to True.

    Returns:
        tuple: Both training and test sets are returned.
    """
    cifar = dataset_meta['cifar-10']
    workdir, filepath = _valid_path_append(path, '', cifar['file'])
    batchdir = os.path.join(workdir, '')

    train_batches = [os.path.join(batchdir, 'data_batch_' + str(i)) for i in range(1, 6)]
    Xlist, ylist = [], []
    for batch in train_batches:
        with open(batch, 'rb') as f:
            d = cPickle.load(f)
            Xlist.append(d['data'])
            ylist.append(d['labels'])

    X_train = np.vstack(Xlist)
    y_train = np.vstack(ylist)

    with open(os.path.join(batchdir, 'test_batch'), 'rb') as f:
        d = cPickle.load(f)
        X_test, y_test = d['data'], d['labels']

    y_train = y_train.reshape(-1, 1)
    y_test = np.array(y_test).reshape(-1, 1)
    num_train = y_train.shape[0]
    num_test = y_test.shape[0]


    y_train_new = np.zeros((num_train, 10))
    y_test_new = np.zeros((num_test, 10))
    for col in range(10):
        y_train_new[:, col] = y_train[:,0] = col
        y_test_new[:, col] = y_test[:,0] = col

    if contrast_normalize:
        norm_scale = 55.0  # Goodfellow
        X_train = global_contrast_normalize(X_train, scale=norm_scale)
        X_test = global_contrast_normalize(X_test, scale=norm_scale)

    if normalize:
        X_train = X_train / 255.
        X_test = X_test / 255.

    if whiten:
        zca_cache = os.path.join(workdir, 'cifar-10-zca-cache.pkl')
        X_train, X_test = zca_whiten(X_train, X_test, cache=zca_cache)

    #save the hdf5 files
    repo_path = os.path.expandvars(os.path.expanduser(workdir))
    save_dir = os.path.join(repo_path, 'HDF5')
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

    file_train_data.create_dataset('data', data = X_train)
    file_train_data.create_dataset('sample_num', data = num_train)
    file_train_label.create_dataset('data', data = y_train_new)
    file_train_label.create_dataset('sample_num', data = num_train)
    file_test_data.create_dataset('data', data = X_test)
    file_test_data.create_dataset('sample_num', data = num_test)
    file_test_label.create_dataset('data', data = y_test_new)
    file_test_label.create_dataset('sample_num', data = num_test)

    file_train_data.close()
    file_train_label.close()
    file_test_data.close()
    file_test_label.close()

    return (X_train, y_train_new), (X_test, y_test_new), 10


dataset_meta = {
    'cifar-10': {
        'size': 170498071,
        'file': 'cifar-10-python.tar.gz',
        'url': 'http://www.cs.toronto.edu/~kriz',
        'func': load_cifar10
    }
}



if __name__ == '__main__':
    load_cifar10()
