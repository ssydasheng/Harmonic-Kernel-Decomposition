#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import os.path as osp
import gzip

import numpy as np
import six
from six.moves import urllib
from six.moves import cPickle as pickle
import tarfile
from urllib.request import urlretrieve
import pandas as pd
import xlrd

BASE_SEED = 1234

def download_dataset(url, path):
    print('Downloading data from %s' % url)
    urllib.request.urlretrieve(url, path)




def to_one_hot(x, depth):
    """
    Get one-hot representation of a 1-D numpy array of integers.
    :param x: 1-D Numpy array of type int.
    :param depth: A int.
    :return: 2-D Numpy array of type int.
    """
    ret = np.zeros((x.shape[0], depth))
    ret[np.arange(x.shape[0]), x] = 1
    return ret


def load_mnist_realval(path, one_hot=True, dequantify=False):
    """
    Loads the real valued MNIST dataset.
    :param path: Path to the dataset file.
    :param one_hot: Whether to use one-hot representation for the labels.
    :param dequantify:  Whether to add uniform noise to dequantify the data
        following (Uria, 2013).
    :return: The MNIST dataset.
    """
    if not os.path.isfile(path):
        data_dir = os.path.dirname(path)
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(data_dir)
        download_dataset('http://www.iro.umontreal.ca/~lisa/deep/data/mnist'
                         '/mnist.pkl.gz', path)

    f = gzip.open(path, 'rb')
    if six.PY2:
        train_set, valid_set, test_set = pickle.load(f)
    else:
        train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
    f.close()
    x_train, t_train = train_set[0], train_set[1]
    x_valid, t_valid = valid_set[0], valid_set[1]
    x_test, t_test = test_set[0], test_set[1]
    if dequantify:
        x_train += np.random.uniform(0, 1. / 256,
                                     size=x_train.shape).astype('float32')
        x_valid += np.random.uniform(0, 1. / 256,
                                     size=x_valid.shape).astype('float32')
        x_test += np.random.uniform(0, 1. / 256,
                                    size=x_test.shape).astype('float32')
    n_y = t_train.max() + 1
    t_transform = (lambda x: to_one_hot(x, n_y)) if one_hot else (lambda x: x)
    return x_train, t_transform(t_train), x_valid, t_transform(t_valid), \
        x_test, t_transform(t_test)

def load_mnist(dtype='float32'):
    data_path = os.path.join('/h/ssy/codes/data', 'mnist.pkl.gz')
    x_train, y_train, x_valid, y_valid, x_test, y_test = load_mnist_realval(data_path)
    x_train, y_train = np.concatenate([x_train, x_valid]), np.concatenate([y_train, y_valid])
    x_train = x_train.reshape([-1, 28, 28, 1]).astype(dtype)
    x_test =  x_test.reshape([-1, 28, 28, 1]).astype(dtype)
    y_train, y_test = y_train.argmax(1).reshape([-1, 1]), y_test.argmax(1).reshape([-1, 1])
    return x_train, y_train, x_test, y_test

def load_cifar10(dtype='float32'): #TODO: data augmentation
    x_train, y_train, x_test, y_test = CIFAR10('/h/ssy/codes/data', onehot=False, dtype=dtype)
    return x_train, y_train.reshape([-1, 1]), x_test, y_test.reshape([-1, 1])

def CIFAR10(path=None, onehot=False, dtype='float32'):
    r"""Return (train_images, train_labels, test_images, test_labels).

    Args:
        path (str): Directory containing CIFAR-10. Default is
            /home/USER/data/cifar10 or C:\Users\USER\data\cifar10.
            Create if nonexistant. Download CIFAR-10 if missing.

    Returns:
        Tuple of (train_images, train_labels, test_images, test_labels), each
            a matrix. Rows are examples. Columns of images are pixel values,
            with the order (red -> blue -> green). Columns of labels are a
            onehot encoding of the correct class.
    """
    url = 'https://www.cs.toronto.edu/~kriz/'
    tar = 'cifar-10-binary.tar.gz'
    files = ['cifar-10-batches-bin/data_batch_1.bin',
             'cifar-10-batches-bin/data_batch_2.bin',
             'cifar-10-batches-bin/data_batch_3.bin',
             'cifar-10-batches-bin/data_batch_4.bin',
             'cifar-10-batches-bin/data_batch_5.bin',
             'cifar-10-batches-bin/test_batch.bin']

    if path is None:
        # Set path to /home/USER/data/mnist or C:\Users\USER\data\mnist
        path = os.path.join(os.path.expanduser('~'), 'data', 'cifar10')

    # Create path if it doesn't exist
    os.makedirs(path, exist_ok=True)

    # Download tarfile if missing
    if tar not in os.listdir(path):
        urlretrieve(''.join((url, tar)), os.path.join(path, tar))
        print("Downloaded %s to %s" % (tar, path))

    # Load data from tarfile
    with tarfile.open(os.path.join(path, tar)) as tar_object:
        # Each file contains 10,000 color images and 10,000 labels
        fsize = 10000 * (32 * 32 * 3) + 10000

        # There are 6 files (5 train and 1 test)
        buffr = np.zeros(fsize * 6, dtype='uint8')

        # Get members of tar corresponding to data files
        # -- The tar contains README's and other extraneous stuff
        members = [file for file in tar_object if file.name in files]

        # Sort those members by name
        # -- Ensures we load train data in the proper order
        # -- Ensures that test data is the last file in the list
        members.sort(key=lambda member: member.name)

        # Extract data from members
        for i, member in enumerate(members):
            # Get member as a file object
            f = tar_object.extractfile(member)
            # Read bytes from that file object into buffr
            buffr[i * fsize:(i + 1) * fsize] = np.frombuffer(f.read(), 'B')

    # Parse data from buffer
    # -- Examples are in chunks of 3,073 bytes
    # -- First byte of each chunk is the label
    # -- Next 32 * 32 * 3 = 3,072 bytes are its corresponding image

    # Labels are the first byte of every chunk
    labels = buffr[::3073]

    # Pixels are everything remaining after we delete the labels
    pixels = np.delete(buffr, np.arange(0, buffr.size, 3073))
    images = pixels.reshape(-1, 3072).astype(dtype) / 255
    images = images.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)

    # Split into train and test
    train_images, test_images = images[:50000], images[50000:]
    train_labels, test_labels = labels[:50000], labels[50000:]

    def _onehot(integer_labels):
        """Return matrix whose rows are onehot encodings of integers."""
        n_rows = len(integer_labels)
        n_cols = integer_labels.max() + 1
        onehot = np.zeros((n_rows, n_cols), dtype='uint8')
        onehot[np.arange(n_rows), integer_labels] = 1
        return onehot

    if onehot:
        return train_images, _onehot(train_labels), test_images, _onehot(test_labels)
    else:
        return train_images, train_labels, test_images, test_labels


def get_automl_data():
    """
    returns the train/test splits of the dataset as N x D matrices and the
    train/test dataset features used for warm-starting bo as D x F matrices.
    N is the number of pipelines, D is the number of datasets (in train/test),
    and F is the number of dataset features.
    """
    fn_data = 'all_normalized_accuracy_with_pipelineID.csv'
    fn_train_ix = 'ids_train.csv'
    fn_test_ix = 'ids_test.csv'
    fn_data_feats = 'data_feats_featurized.csv'

    BASE_PATH = '/h/ssy/codes/data'
    fn_data = os.path.join(BASE_PATH, fn_data)
    fn_train_ix = os.path.join(BASE_PATH, fn_train_ix)
    fn_test_ix = os.path.join(BASE_PATH, fn_test_ix)
    fn_data_feats = os.path.join(BASE_PATH, fn_data_feats)

    df = pd.read_csv(fn_data)
    pipeline_ids = df['Unnamed: 0'].tolist()
    dataset_ids = df.columns.tolist()[1:]
    dataset_ids = [int(dataset_ids[i]) for i in range(len(dataset_ids))]
    Y = df.values[:,1:].astype(np.float64)

    ids_train = np.loadtxt(fn_train_ix).astype(int).tolist()
    ids_test = np.loadtxt(fn_test_ix).astype(int).tolist()

    ix_train = [dataset_ids.index(i) for i in ids_train]
    ix_test = [dataset_ids.index(i) for i in ids_test]

    Ytrain = Y[:, ix_train]
    Ytest = Y[:, ix_test]

    df = pd.read_csv(fn_data_feats)
    dataset_ids = df[df.columns[0]].tolist()

    ix_train = [dataset_ids.index(i) for i in ids_train]
    ix_test = [dataset_ids.index(i) for i in ids_test]

    Ftrain = df.values[ix_train, 1:]
    Ftest = df.values[ix_test, 1:]

    return Ytrain, Ytest, Ftrain, Ftest


def get_jester_data(seed=0, model_item=False):
    """
    returns the train/test splits of the dataset as N x D matrices and the
    train/test dataset features used for warm-starting bo as D x F matrices.
    N is the number of pipelines, D is the number of datasets (in train/test),
    and F is the number of dataset features.
    """
    train_prop = 0.9
    BASE_PATH = '/h/ssy/codes/data/JesterJoke'

    data1 = os.path.join(BASE_PATH, 'jester-data-1.xls')
    data2 = os.path.join(BASE_PATH, 'jester-data-1.xls')
    data3 = os.path.join(BASE_PATH, 'jester-data-1.xls')

    xl_file = pd.ExcelFile(data1)
    dfs1 = {sheet_name: xl_file.parse(sheet_name)
           for sheet_name in xl_file.sheet_names}

    xl_file = pd.ExcelFile(data2)
    dfs2 = {sheet_name: xl_file.parse(sheet_name)
           for sheet_name in xl_file.sheet_names}

    xl_file = pd.ExcelFile(data3)
    dfs3 = {sheet_name: xl_file.parse(sheet_name)
           for sheet_name in xl_file.sheet_names}

    dfs = pd.concat([dfs1['jester-data-1-new'], dfs2['jester-data-1-new'], dfs3['jester-data-1-new']])
    data = dfs.to_numpy()[:, 1:] # (74946, 100)

    np.random.seed(BASE_SEED + seed)
    train_flag = np.random.choice(2, size=data.shape, p=[1-train_prop, train_prop])

    valid_ratio_flag = np.random.choice(2, size=data.shape, p=[0.05, 0.95])
    valid_flag = train_flag * (1 - valid_ratio_flag)
    train_flag = train_flag * valid_ratio_flag

    test_flag = 1-train_flag 

    train_data = data * train_flag + 99 * (1-train_flag) # [n_train, 100]
    valid_data = data * valid_flag + 99 * (1-valid_flag) # [n_train, 100]
    test_data = data * test_flag + 99 * (1-test_flag) # [n_test, 100]


    test_flag = test_data < 90
    test_data = test_data / 20. + 0.5
    test_data = test_data * test_flag + 99. * (1-test_flag)

    flag = train_data < 90
    train_data = train_data / 20. + 0.5
    train_data = train_data * flag + 99. * (1-flag)

    flag = valid_data < 90
    valid_data = valid_data / 20. + 0.5
    valid_data = valid_data * flag + 99. * (1-flag)

    # flag = train_data < 90
    # row_mean = np.sum(train_data * flag, 0) / np.sum(flag, 0)
    # col_mean = np.sum(train_data * flag, 1) / np.sum(flag, 1)
    # cross_mean = row_mean[None] + col_mean[:, None]
    # all_mean = np.sum(train_data * flag) / np.sum(flag)
    #
    # train_data = train_data + all_mean - cross_mean
    # test_data = test_data + all_mean - cross_mean
    #
    # train_data = train_data * flag + 99. * (1-flag)
    # test_data = test_data * test_flag + 99. * (1-test_flag)

    if model_item:
        train_data, valid_data, test_data = train_data.T, valid_data.T, test_data.T
    return train_data, valid_data, test_data


def get_movie_100k(seed=1, model_item=False):
    # Define file directories
    MOVIELENS_DIR = '/h/ssy/codes/data/MovieLens/ml-100k'

    train_data = pd.read_csv(os.path.join(MOVIELENS_DIR, 'u%d.base' % seed), sep="\t", header=None)
    test_data = pd.read_csv(os.path.join(MOVIELENS_DIR, 'u%d.test' % seed), sep="\t", header=None)

    train_data = train_data.pivot(index=0, columns=1, values=2).fillna(99.).to_numpy()
    test_data = test_data.pivot(index=0, columns=1, values=2).fillna(99.).to_numpy()

    np.random.seed(BASE_SEED + seed)
    valid_ratio_flag = np.random.choice(2, size=train_data.shape, p=[0.02, 0.98])

    valid_data = train_data * (1-valid_ratio_flag) + 99 * valid_ratio_flag # [n_test, 100]
    train_data = train_data * valid_ratio_flag + 99 * (1-valid_ratio_flag) # [n_train, 100]

    flag = train_data < 90
    train_data = (train_data - 1) / (5 - 1)
    train_data = train_data * flag + 99 * (1-flag)

    flag = valid_data < 90
    valid_data = (valid_data - 1) / (5 - 1)
    valid_data = valid_data * flag + 99 * (1-flag)

    flag = test_data < 90
    test_data = (test_data - 1) / (5 - 1)
    test_data = test_data * flag + 99 * (1-flag)

    if model_item:
        train_data, valid_data, test_data = train_data.T, valid_data.T, test_data.T
    return train_data, valid_data, test_data


def get_movie_1M(seed=1, normalize=True, model_item=False):
    # Define file directories
    MOVIELENS_DIR = '/h/ssy/codes/data/MovieLens/ml-1m'
    RATING_DATA_FILE = 'ratings.dat'

    # Read the Ratings File
    ratings = pd.read_csv(os.path.join(MOVIELENS_DIR, RATING_DATA_FILE),
                          sep='::',
                          engine='python',
                          encoding='latin-1',
                          names=['user_id', 'movie_id', 'rating', 'timestamp'])

    ratings_array = ratings.pivot(index='user_id',
                                  columns='movie_id',
                                  values='rating').fillna(99.)
    data = ratings_array.to_numpy()
    np.random.seed(BASE_SEED + seed)
    train_flag = np.random.choice(2, size=data.shape, p=[0.1, 0.9])
    test_flag = 1 - train_flag
    valid_ratio_flag = np.random.choice(2, size=data.shape, p=[0.005, 0.995])
    valid_flag = train_flag * (1-valid_ratio_flag)
    train_flag = train_flag * valid_ratio_flag

    train_data = data * train_flag + 99 * (1-train_flag) # [n_train, 100]
    valid_data = data * valid_flag + 99 * (1-valid_flag) # [n_test, 100]
    test_data = data * test_flag + 99 * (1-test_flag) # [n_test, 100]

    flag = train_data < 90
    if normalize: train_data = (train_data - 1) / (5 - 1)
    else: train_data = (train_data - 1)
    train_data = train_data * flag + 99 * (1-flag)

    flag = valid_data < 90
    if normalize: valid_data = (valid_data - 1) / (5 - 1)
    else: valid_data = (valid_data - 1)
    valid_data = valid_data * flag + 99 * (1-flag)

    flag = test_data < 90
    if normalize: test_data = (test_data - 1) / (5 - 1)
    else: test_data = (test_data - 1)
    test_data = test_data * flag + 99 * (1-flag)

    if model_item:
        train_data, valid_data, test_data = train_data.T, valid_data.T, test_data.T
    return train_data, valid_data, test_data


def select_label(values, label, n):
    probs = np.equal(values, label) / np.sum(np.equal(values, label))
    if n >= values.shape[0]:
        return np.arange(values.shape[0])[np.equal(values, label)]
    return np.random.choice(np.arange(values.shape[0]), n, replace=False, p=probs)

def get_mnist_missing(seed, n_per, missing_rate=0.9):
    np.random.seed(seed)

    data_path = os.path.join('/h/ssy/codes/data', 'mnist.pkl.gz')
    y_train, t_train, y_valid, t_valid, y_test, t_test = \
        load_mnist_realval(data_path)
    y_train = np.vstack([y_train, y_valid])
    t_train = np.vstack([t_train, t_valid]).argmax(1)
    indices = np.concatenate([select_label(t_train, idx, n_per) for idx in range(10)], 0)
    np.random.shuffle(indices)
    y_train = y_train[indices]
    t_train = t_train[indices]

    data = y_train

    train_mask = np.random.choice(2, size=y_train.shape, p=[missing_rate, 1-missing_rate])
    test_mask = 1 - train_mask
    valid_ratio_mask = np.random.choice(2, size=y_train.shape, p=[0., 1.])
    valid_mask = train_mask * (1 - valid_ratio_mask)
    train_mask = train_mask * valid_ratio_mask

    train = data * train_mask + 99. * (1-train_mask)
    valid = data * valid_mask + 99. * (1-valid_mask)
    test = data * test_mask + 99. * (1-test_mask)
    return train, valid, test, t_train


if __name__ == '__main__':
    get_jester_data()
