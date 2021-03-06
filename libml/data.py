"""
Input data for image models.
"""

import glob
import itertools
import os

import numpy as np
import tensorflow as tf
from absl import flags
from tqdm import tqdm

from libml import utils

_DATA_CACHE = None
os.environ['ML_DATA'] = './input'
DATA_DIR = os.environ['ML_DATA']
flags.DEFINE_string('dataset', 'cifar10.1@4000-5000', 'Data to train on.')
flags.DEFINE_integer('para_parse', 4, 'Parallel parsing.')
flags.DEFINE_integer('para_augment', 4, 'Parallel augmentation.')
flags.DEFINE_integer('shuffle', 8192, 'Size of dataset shuffling.')
flags.DEFINE_bool('whiten', False, 'Whether to normalize images.')
FLAGS = flags.FLAGS


def record_parse(serialized_example):
    features = tf.parse_single_example(
        serialized_example,
        features={'image': tf.FixedLenFeature([], tf.string),
                  'label': tf.FixedLenFeature([], tf.int64)})
    image = tf.image.decode_image(features['image'])
    image = tf.cast(image, tf.float32) * (2.0 / 255) - 1.0 #convert [0,255] into [-1,1]
    label = features['label']
    return dict(image=image, label=label)

def default_parse(dataset: tf.data.Dataset, parse_fn=record_parse) -> tf.data.Dataset:
    para = 4 * max(1, len(utils.get_available_gpus())) * FLAGS.para_parse
    return dataset.map(parse_fn, num_parallel_calls=para)


def dataset(filenames: list) -> tf.data.Dataset:
    filenames = sorted(sum([glob.glob(x) for x in filenames], []))
    if not filenames:
        raise ValueError('Empty dataset, did you mount gcsfuse bucket?')
    return tf.data.TFRecordDataset(filenames)


def memoize(dataset: tf.data.Dataset) -> tf.data.Dataset:
    data = []
    with tf.Session(config=utils.get_config()) as session:
        dataset = dataset.prefetch(16)
        it = dataset.make_one_shot_iterator().get_next()
        try:
            while 1:
                data.append(session.run(it))
        except tf.errors.OutOfRangeError:
            pass
    images = np.stack([x['image'] for x in data])
    labels = np.stack([x['label'] for x in data])

    def tf_get(index):
        def get(index):
            return images[index], labels[index]

        image, label = tf.py_func(get, [index], [tf.float32, tf.int64])
        return dict(image=image, label=label)

    dataset = tf.data.Dataset.range(len(data)).repeat()
    dataset = dataset.shuffle(len(data) if len(data) < FLAGS.shuffle else FLAGS.shuffle)
    return dataset.map(tf_get)


def augment_mirror(x):
    return tf.image.random_flip_left_right(x)


def augment_shift(x, w):
    y = tf.pad(x, [[w] * 2, [w] * 2, [0] * 2], mode='REFLECT')
    return tf.random_crop(y, tf.shape(x))


def augment_noise(x, std):
    return x + std * tf.random_normal(tf.shape(x), dtype=x.dtype)


def compute_mean_std(data: tf.data.Dataset):
    data = data.map(lambda x: x['image']).batch(1024).prefetch(1)
    data = data.make_one_shot_iterator().get_next()
    count = 0
    stats = []
    with tf.Session(config=utils.get_config()) as sess:
        def iterator():
            while True:
                try:
                    yield sess.run(data)
                except tf.errors.OutOfRangeError:
                    break

        for batch in tqdm(iterator(), unit='kimg', desc='Computing dataset mean and std'):
            ratio = batch.shape[0] / 1024.
            count += ratio
            stats.append((batch.mean((0, 1, 2)) * ratio, (batch ** 2).mean((0, 1, 2)) * ratio))
    mean = sum(x[0] for x in stats) / count
    sigma = sum(x[1] for x in stats) / count - mean ** 2
    std = np.sqrt(sigma)
    print('Mean %s  Std: %s' % (mean, std))
    return mean, std


class DataSet:
    def __init__(self, name, train_labeled, test, valid, eval_labeled,
                 height=32, width=32, colors=3, nclass=10, mean=0, std=1):
        self.name = name
        self.train_labeled = train_labeled
        self.eval_labeled = eval_labeled
        self.test = test
        self.valid = valid
        self.height = height
        self.width = width
        self.colors = colors
        self.nclass = nclass
        self.mean = mean
        self.std = std

    @classmethod
    def creator(cls, name, seed, label, valid, augment, parse_fn=default_parse, do_memoize=True, colors=3,
                nclass=10, height=32, width=32, name_suffix=''):
        fullname = '.%d@%d' % (seed, label)
        train_files = [os.path.join(DATA_DIR, name, name + fullname) + '.tfrecord']
        test_files = [os.path.join(DATA_DIR, '%s-test.tfrecord' % name)]
        fn = memoize if do_memoize else lambda x: x.repeat().shuffle(FLAGS.shuffle)

        def create():
            para = max(1, len(utils.get_available_gpus())) * FLAGS.para_augment
            # valid: sample size of a validation set
            train_labeled = parse_fn(dataset(train_files).skip(valid))
            if FLAGS.whiten:
                mean, std = compute_mean_std(train_labeled)
            else:
                mean, std = 0, 1

            return cls(name + name_suffix + fullname + '-' + str(valid),
                       train_labeled=fn(train_labeled).map(augment, para),
                       eval_labeled=train_labeled.take(5000), # No need to eval on everything.
                       valid=parse_fn(dataset(train_files).take(valid)),
                       test=parse_fn(dataset(test_files)),
                       nclass=nclass, colors=colors, height=height, width=width, mean=mean, std=std)

        return name + name_suffix + fullname + '-' + str(valid), create

augment_stl10 = lambda x: dict(image=augment_shift(augment_mirror(x['image']), 12), label=x['label'])
augment_cifar10 = lambda x: dict(image=augment_shift(augment_mirror(x['image']), 4), label=x['label'])
augment_svhn = lambda x: dict(image=augment_shift(x['image'], 4), label=x['label'])


DATASETS = {}
DATASETS.update([DataSet.creator('cifar10', seed, label, valid, augment_cifar10)
                 for seed, label, valid in
                 itertools.product(range(6), [100, 500, 1000, 2500, 5000, 25000, 50000], [1, 5000])])
DATASETS.update([DataSet.creator('cifar100', seed, label, valid, augment_cifar10, nclass=100)
                 for seed, label, valid in
                 itertools.product(range(6), [1000, 2500, 5000, 25000, 50000], [1, 5000])])
