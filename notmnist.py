# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 10:14:34 2017

@author: cheers
"""

import tensorflow as tf
import numpy as np
import os
import sys
import tarfile
from scipy import ndimage
# from six.moves.urllib.request import urlretrieve
from six.moves import cPickle as pickle


url = 'http://cn-static.udacity.com/mlnd/'
last_percent_reported = None


# def download_progress_hook(count, blockSize, totalSize):
#     """A hook to report the progress of a download. This is mostly intended for users with
#     slow internet connections. Reports every 5% change in download progress.
#     """
#     global last_percent_reported
#     percent = int(count * blockSize * 100 / totalSize)
#     if last_percent_reported != percent:
#         if percent % 5 == 0:
#             sys.stdout.write("%s%%" % percent)
#             sys.stdout.flush()
#         else:
#             sys.stdout.write(".")
#             sys.stdout.flush()
#
#         last_percent_reported = percent


# def maybe_download(filename, expected_bytes, force=False):
#     """Download a file if not present, and make sure it's the right size."""
#     if force or not os.path.exists(filename):
#         print('Attempting to download:', filename)
#         filename, _ = urlretrieve(url + filename, filename, reporthook=download_progress_hook)
#         print('\nDownload Complete!')
#     statinfo = os.stat(filename)
#     if statinfo.st_size == expected_bytes:
#         print('Found and verified', filename)
#     else:
#         raise Exception(
#             'Failed to verify ' + filename + '. Can you get to it with a browser?')
#     return filename


num_classes = 10
np.random.seed(133)


def maybe_extract(filename, force=False):
    root = os.path.splitext(os.path.splitext(filename)[0])[0]  # remove .tar.gz
    if os.path.isdir(root) and not force:
        # You may override by setting force=True.
        print('%s already present - Skipping extraction of %s.' % (root, filename))
    else:
        print('Extracting data for %s. This may take a while. Please wait.' % root)
        tar = tarfile.open(filename)
        sys.stdout.flush()
        tar.extractall()
        tar.close()
    data_folders = [
        os.path.join(root, d) for d in sorted(os.listdir(root))
        if os.path.isdir(os.path.join(root, d))]
    if len(data_folders) != num_classes:
        raise Exception(
            'Expected %d folders, one per class. Found %d instead.' % (
                num_classes, len(data_folders)))
    print(data_folders)
    return data_folders


def display_oriImage():
    """
    display three image of original
    """
    from IPython.display import Image, display
    print("examples of original images")
    listOfImageNames = ['/home/yuxin/data/notMNIST/notMNIST_small/A/MDEtMDEtMDAudHRm.png',
                        '/home/yuxin/data/notMNIST/notMNIST_small/G/MTIgV2FsYmF1bSBJdGFsaWMgMTMyNjMudHRm.png',
                        '/home/yuxin/data/notMNIST/notMNIST_small/J/Q0cgT21lZ2EudHRm.png', ]
    for imageName in listOfImageNames:
        display(Image(filename=imageName))


image_size = 28  # Pixel width and height.
pixel_depth = 255.0  # Number of levels per pixel.


def load_letter(folder, min_num_images):
    """Load the data for a single letter label."""
    image_files = os.listdir(folder)  # 图像个数
    dataset = np.ndarray(shape=(len(image_files), image_size, image_size),  # 新建一个numpy三维数组
                         dtype=np.float32)
    print(folder)
    num_images = 0
    for image in image_files:
        image_file = os.path.join(folder, image)
        try:
            image_data = (ndimage.imread(image_file).astype(float) -
                          pixel_depth / 2) / pixel_depth  # （图像像素-128）/255
            if image_data.shape != (image_size, image_size):
                raise Exception('Unexpected image shape: %s' % str(image_data.shape))
            dataset[num_images, :, :] = image_data
            num_images = num_images + 1
        except IOError as e:
            print('Could not read:', image_file, ':', e, '- it\'s ok, skipping.')

    dataset = dataset[0:num_images, :, :]
    if num_images < min_num_images:
        raise Exception('Many fewer images than expected: %d < %d' %
                        (num_images, min_num_images))

    print('Full dataset tensor:', dataset.shape)
    print('Mean:', np.mean(dataset))
    print('Standard deviation:', np.std(dataset))
    return dataset


def maybe_pickle(data_folders, min_num_images_per_class, force=False):
    dataset_names = []
    for folder in data_folders:
        set_filename = folder + '.pickle'
        dataset_names.append(set_filename)
        if os.path.exists(set_filename) and not force:  # 如果已经存在并且没有强制执行，就不执行
            # You may override by setting force=True.
            print('%s already present - Skipping pickling.' % set_filename)
        else:
            print('Pickling %s.' % set_filename)
            dataset = load_letter(folder, min_num_images_per_class)
            try:
                with open(set_filename, 'wb') as f:
                    pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)
            except Exception as e:
                print('Unable to save data to', set_filename, ':', e)

    return dataset_names


def check_balance():
    # count numbers in different classes
    print("start check the balance of different calsses")
    file_path = '/home/yuxin/data/notMNIST/notMNIST_large/{0}.pickle'
    for ele in 'ABCDEFJHIJ':
        with open(file_path.format(ele), 'rb') as pk_f:
            dat = pickle.load(pk_f)
        print('number of pictures in {}.pickle = '.format(ele), dat.shape[0])
    print("balance checked ok")


def make_arrays(nb_rows, img_size):
    if nb_rows:
        dataset = np.ndarray((nb_rows, img_size, img_size), dtype=np.float32)
        labels = np.ndarray(nb_rows, dtype=np.int32)
    else:
        dataset, labels = None, None
    return dataset, labels


def merge_datasets(pickle_files, train_size, valid_size=0):
    num_classes = len(pickle_files)
    valid_dataset, valid_labels = make_arrays(valid_size, image_size)  # 10000
    train_dataset, train_labels = make_arrays(train_size, image_size)  # 200000
    vsize_per_class = valid_size // num_classes
    tsize_per_class = train_size // num_classes

    start_v, start_t = 0, 0
    end_v, end_t = vsize_per_class, tsize_per_class
    end_l = vsize_per_class + tsize_per_class
    for label, pickle_file in enumerate(pickle_files):
        try:
            with open(pickle_file, 'rb') as f:
                letter_set = pickle.load(f)
                # let's shuffle the letters to have random validation and training set
                np.random.shuffle(letter_set)
                if valid_dataset is not None:
                    valid_letter = letter_set[:vsize_per_class, :, :]
                    valid_dataset[start_v:end_v, :, :] = valid_letter
                    valid_labels[start_v:end_v] = label
                    start_v += vsize_per_class
                    end_v += vsize_per_class

                train_letter = letter_set[vsize_per_class:end_l, :, :]
                train_dataset[start_t:end_t, :, :] = train_letter
                train_labels[start_t:end_t] = label
                start_t += tsize_per_class
                end_t += tsize_per_class
        except Exception as e:
            print('Unable to process data from', pickle_file, ':', e)
            raise

    return valid_dataset, valid_labels, train_dataset, train_labels


def randomize(dataset, labels):
    permutation = np.random.permutation(labels.shape[0])
    shuffled_dataset = dataset[permutation, :, :]
    shuffled_labels = labels[permutation]
    return shuffled_dataset, shuffled_labels


def pickle_datas(notMNIST):
    print("start pick data")
    pickle_file = '/home/yuxin/data/notMNIST/notMNIST.pickle'

    try:
        f = open(pickle_file, 'wb')
        save = {
            'train_dataset': notMNIST.train_dataset,
            'train_labels': notMNIST.train_labels,
            'valid_dataset': notMNIST.valid_dataset,
            'valid_labels': notMNIST.valid_labels,
            'test_dataset': notMNIST.test_dataset,
            'test_labels': notMNIST.test_labels,
        }
        pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
        f.close()
    except Exception as e:
        print('Unable to save data to', pickle_file, ':', e)
        raise
    statinfo = os.stat(pickle_file)
    print('Compressed pickle size:', statinfo.st_size)


def prepare_data(data_dir="/home/yuxin/data/notMNIST/"):
    class notMNIST(object):
        pass

    train_size = 200000
    valid_size = 10000
    test_size = 10000

    train_filename = data_dir + 'notMNIST_large.tar.gz'
    test_filename = data_dir + 'notMNIST_small.tar.gz'

    train_folders = maybe_extract(train_filename)
    test_folders = maybe_extract(test_filename)
    display_oriImage()
    train_datasets = maybe_pickle(train_folders, 45000)
    test_datasets = maybe_pickle(test_folders, 1800)
    check_balance()
    valid_dataset, valid_labels, train_dataset, train_labels = merge_datasets(
        train_datasets, train_size, valid_size)
    _, _, test_dataset, test_labels = merge_datasets(test_datasets, test_size)

    print('Training:', train_dataset.shape, train_labels.shape)
    print('Validation:', valid_dataset.shape, valid_labels.shape)
    print('Testing:', test_dataset.shape, test_labels.shape)

    notMNIST.train_dataset, notMNIST.train_labels = randomize(train_dataset, train_labels)
    notMNIST.test_dataset, notMNIST.test_labels = randomize(test_dataset, test_labels)
    notMNIST.valid_dataset, notMNIST.valid_labels = randomize(valid_dataset, valid_labels)

    pickle_datas(notMNIST)

    print('notMNIST data prepared ok')


image_size = 28
num_labels = 10


def reformat(dataset, labels):
    """
    reformat the imagedata with shape [-1,28,28,1]
    reformat the label with one-hot shape
    """
    new_dataset = dataset.reshape((-1, image_size, image_size, 1)).astype(np.float32)

    # Map 0 to [1.0, 0.0, 0.0 ...], 1 to [0.0, 1.0, 0.0 ...]
    # np.arange(num_labels)默认是生成0 1，2，3，4，5，6，7，8，9 取出labels，例如2，然后比较是否相等
    # 生成 FALSE， FALSE， TURE，FALSE， FALSE。。。再转换成32浮点 0，0，1，0，0...这样便成one_hot 数据
    new_labels = (np.arange(num_labels) == labels[:, None]).astype(np.float32)
    return new_dataset, new_labels


class DataSet(object):
    def __init__(self, images, labels, fake_data=False):
        if fake_data:
            self._num_examples = 10000
        else:
            assert images.shape[0] == labels.shape[0], (
                "images.shape: %s labels.shape: %s" % (images.shape,
                                                       labels.shape))
            self._num_examples = images.shape[0]
            # Convert shape from [num examples, rows, columns, depth]
            # to [num examples, rows*columns] (assuming depth == 1)
            assert images.shape[3] == 1
            images = images.reshape(images.shape[0],
                                    images.shape[1] * images.shape[2])
            # Convert from [0, 255] -> [0.0, 1.0].
            images = images.astype(np.float32)
            # images = np.multiply(images, 1.0 / 255.0)
        self._images = images
        self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0

    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size, fake_data=False):
        """Return the next `batch_size` examples from this data set."""
        if fake_data:
            fake_image = [1.0 for _ in range(784)]
            fake_label = 0
            return [fake_image for _ in range(batch_size)], [
                fake_label for _ in range(batch_size)]
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Shuffle the data
            perm = np.arange(self._num_examples)
            np.random.shuffle(perm)
            self._images = self._images[perm]
            self._labels = self._labels[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch
        return self._images[start:end], self._labels[start:end]


def load_data(pickle_file, one_hot=True, fake_data=False):
    class DataSets(object):
        pass

    data_sets = DataSets()
    if fake_data:
        data_sets.train = DataSet([], [], fake_data=True)
        data_sets.validation = DataSet([], [], fake_data=True)
        data_sets.test = DataSet([], [], fake_data=True)
        return data_sets

    with open(pickle_file, 'rb') as f:
        save = pickle.load(f)
        train_dataset = save['train_dataset']
        train_labels = save['train_labels']
        valid_dataset = save['valid_dataset']
        valid_labels = save['valid_labels']
        test_dataset = save['test_dataset']
        test_labels = save['test_labels']
        del save  # hint to help gc free up memory

        if one_hot:
            train_dataset, train_labels = reformat(train_dataset, train_labels)
            valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
            test_dataset, test_labels = reformat(test_dataset, test_labels)

        print('Training set', train_dataset.shape, train_labels.shape, type(train_dataset))
        print('Validation set', valid_dataset.shape, valid_labels.shape)
        print('Test set', test_dataset.shape, test_labels.shape)
        # print(test_labels)

        data_sets.train = DataSet(train_dataset, train_labels)
        data_sets.validation = DataSet(valid_dataset, valid_labels)
        data_sets.test = DataSet(test_dataset, test_labels)

        # return data_sets.train.images, data_sets.train.labels, data_sets.test.images, data_sets.test.labels,\
        #        data_sets.validation.images, data_sets.validation.labels
        return data_sets


# def generate_batch(batch_size=28):
#     train_images, train_labels, test_image, test_labels, val_images, val_labels = load_data(one_hot=True)
#     train_images = train_images.reshape([-1, 28, 28, 1])
#     test_images = test_image.reshape([-1, 28, 28, 1])
#     val_images = val_images.reshape([-1, 28, 28, 1])
#
#     train_batch, train_label_batch = tf.train.batch([train_images, train_labels],
#                                                     batch_size=batch_size,
#                                                     num_threads=64,
#                                                     capacity=20000)
#     test_batch, test_label_batch = tf.train.batch([test_images, test_labels],
#                                                   batch_size=batch_size,
#                                                   num_threads=64,
#                                                   capacity=20000)
#     val_batch, val_label_batch = tf.train.batch([val_images, val_labels],
#                                                 batch_size=batch_size,
#                                                 num_threads=64,
#                                                 capacity=20000)
#
#     return train_batch, train_label_batch, test_batch, test_label_batch, val_batch, val_label_batch


if __name__ == '__main__':
    # prepare_data()
    a = load_data()
