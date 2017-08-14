from __future__ import absolute_import
from __future__ import print_function

import os
import sys
import numpy as np
import tarfile
import shutil
from keras.utils.generic_utils import Progbar

from six.moves.urllib.error import HTTPError
from six.moves.urllib.error import URLError
from six.moves.urllib.request import urlretrieve
from six.moves import cPickle


MNIST_ORIGIN = 'https://s3.amazonaws.com/img-datasets/mnist.npz'
CIFAR10_ORIGIN = 'http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'


def load_mnist(path="mnist/"):

    if not os.path.exists(path):
        os.makedirs(path)

    fname = "mnist.npz"
    fpath = os.path.join(path, fname)

    if os.path.exists(fpath):
        download = False
    else:
        download = True

    if download:
        print('Downloading data from', MNIST_ORIGIN)

        class ProgressTracker(object):
            progbar = None

        def dl_progress(count, block_size, total_size):
            if ProgressTracker.progbar is None:
                if total_size is -1:
                    total_size = None
                ProgressTracker.progbar = Progbar(total_size)
            else:
                ProgressTracker.progbar.update(count * block_size)

        error_msg = 'URL fetch failure on {}: {} -- {}'
        try:
            try:
                urlretrieve(MNIST_ORIGIN, fpath, dl_progress)
            except URLError as e:
                raise Exception(error_msg.format(MNIST_ORIGIN, e.errno, e.reason))
            except HTTPError as e:
                raise Exception(error_msg.format(MNIST_ORIGIN, e.code, e.msg))
        except (Exception, KeyboardInterrupt) as e:
            if os.path.exists(fpath):
                os.remove(fpath)
            raise
        ProgressTracker.progbar = None

    f = np.load(fpath)
    x_train, y_train = f['x_train'], f['y_train']
    x_test, y_test = f['x_test'], f['y_test']
    f.close()
    return (x_train, y_train), (x_test, y_test)

def load_cifar10(path="cifar10/"):

    if not os.path.exists(path):
        os.makedirs(path)

    fname = "cifar-10-batches-py"
    untar_fpath = os.path.join(path, fname)

    if os.path.exists(untar_fpath):
        download = False
    else:
        download = True

    if download:
        # download data
        tar_fpath = untar_fpath + '.tar.gz'

        print('Downloading data from', CIFAR10_ORIGIN)

        class ProgressTracker(object):
            progbar = None

        def dl_progress(count, block_size, total_size):
            if ProgressTracker.progbar is None:
                if total_size is -1:
                    total_size = None
                ProgressTracker.progbar = Progbar(total_size)
            else:
                ProgressTracker.progbar.update(count * block_size)

        error_msg = 'URL fetch failure on {}: {} -- {}'
        try:
            try:
                urlretrieve(CIFAR10_ORIGIN, tar_fpath, dl_progress)
            except URLError as e:
                raise Exception(error_msg.format(CIFAR10_ORIGIN, e.errno, e.reason))
            except HTTPError as e:
                raise Exception(error_msg.format(CIFAR10_ORIGIN, e.code, e.msg))
        except (Exception, KeyboardInterrupt) as e:
            if os.path.exists(tar_fpath):
                os.remove(tar_fpath)
            raise
        ProgressTracker.progbar = None

        if tarfile.is_tarfile(tar_fpath):
            with tarfile.open(tar_fpath) as archive:
                try:
                    archive.extractall(path)
                except (tarfile.TarError, RuntimeError,
                        KeyboardInterrupt):
                    if os.path.exists(untar_fpath):
                        if os.path.isfile(untar_fpath):
                            os.remove(untar_fpath)
                        else:
                            shutil.rmtree(untar_fpath)

    fpath = os.path.join(path, "cifar-10-batches-py")
    num_train_samples = 50000
    x_train = np.zeros((num_train_samples, 3, 32, 32), dtype='uint8')
    y_train = np.zeros((num_train_samples,), dtype='uint8')

    for i in range(1, 6):
        current_fpath = os.path.join(fpath, 'data_batch_' + str(i))
        data, labels = load_batch(current_fpath)
        x_train[(i - 1) * 10000: i * 10000, :, :, :] = data
        y_train[(i - 1) * 10000: i * 10000] = labels

    fpath = os.path.join(fpath, 'test_batch')
    x_test, y_test = load_batch(fpath)

    y_train = np.reshape(y_train, (len(y_train), 1))
    y_test = np.reshape(y_test, (len(y_test), 1))

    # channels_last:
    x_train = x_train.transpose(0, 2, 3, 1)
    x_test = x_test.transpose(0, 2, 3, 1)

    return (x_train, y_train), (x_test, y_test)


def load_batch(fpath, label_key='labels'):
    """Internal utility for parsing CIFAR data.
    # Arguments
        fpath: path the file to parse.
        label_key: key for label data in the retrieve
            dictionary.
    # Returns
        A tuple `(data, labels)`.
    """
    f = open(fpath, 'rb')
    if sys.version_info < (3,):
        d = cPickle.load(f)
    else:
        d = cPickle.load(f, encoding='bytes')
        # decode utf8
        d_decoded = {}
        for k, v in d.items():
            d_decoded[k.decode('utf8')] = v
        d = d_decoded
    f.close()
    data = d['data']
    labels = d[label_key]

    data = data.reshape(data.shape[0], 3, 32, 32)
    return data, labels