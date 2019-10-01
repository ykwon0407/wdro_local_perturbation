# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Mixup fully supervised training.
"""

import functools
import os

from absl import app
from absl import flags
from easydict import EasyDict
from skimage.util import random_noise

from libml import utils
from libml.models import MultiModel
from libml.data import DATASETS

import tensorflow as tf

FLAGS = flags.FLAGS


class MixupGrad(MultiModel):

    def augment(self, x, l, beta, **kwargs):
        del kwargs
        mix = tf.distributions.Beta(beta, beta).sample([tf.shape(x)[0], 1, 1, 1])
        mix = tf.maximum(mix, 1 - mix)
        xmix = x * mix + x[::-1] * (1 - mix)
        lmix = l * mix[:, :, 0, 0] + l[::-1] * (1 - mix[:, :, 0, 0])
        return xmix, lmix

    def model(self, lr, wd, ema, regularizer, gamma, LH, **kwargs):
        hwc = [self.dataset.height, self.dataset.width, self.dataset.colors]
        x_in = tf.placeholder(tf.float32, [None] + hwc, 'x')
        l_in = tf.placeholder(tf.int32, [None], 'labels')
        wd *= lr
        classifier = functools.partial(self.classifier, **kwargs)
        def get_logits(x):
            logits = classifier(x, training=True)
            return logits

        x, labels_x = self.augment(x_in, tf.one_hot(l_in, self.nclass), **kwargs)
        logits_x = get_logits(x)
        loss_xe = tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels_x, logits=logits_x) #shape = (batchsize,)
        gradient = tf.gradients(loss_xe, x)[0] #output is list (batchsize, height, width, colors)
        loss_main = tf.reduce_mean(loss_xe)

        if regularizer == 'maxsup':
            loss_grad = tf.maximum(tf.reduce_max(tf.abs(gradient)) - tf.constant(LH), tf.constant(0.0))
        elif regularizer == 'maxl2':
            loss_grad = tf.maximum(tf.reduce_sum(tf.square(gradient))/tf.constant(FLAGS.batch, dtype=tf.float32) - tf.square(LH), tf.constant(0.0))
        elif regularizer == 'l2':
            loss_grad = gamma*tf.reduce_sum(tf.square(gradient))/tf.constant(FLAGS.batch, dtype=tf.float32)
        elif regularizer == 'None':
            # Same with mixup
            loss_grad = tf.constant(0.0)
        else:
            assert False, 'unavailable regularizer, (maxsup, maxl2, l2, None)'
            
        tf.summary.scalar('losses/main', loss_main)
        tf.summary.scalar('losses/gradient', loss_grad)
        tf.summary.scalar('gradient/max_gradient', tf.reduce_max(tf.abs(gradient)))
        loss_xe = loss_main + loss_grad

        #For lipschitz
        sup_gradient = tf.reduce_max(tf.abs(gradient), axis=[1,2,3]) #(batchsize, )

        #EMA part
        if ema > 0 :
            ema = tf.train.ExponentialMovingAverage(decay=ema)
            ema_op = ema.apply(utils.model_vars())
            ema_getter = functools.partial(utils.getter_ema, ema)
            post_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS) + [ema_op]
            post_ops.extend([tf.assign(v, v * (1 - wd)) for v in utils.model_vars('classify') if 'kernel' in v.name])

            train_op = tf.train.AdamOptimizer(lr).minimize(loss_xe, colocate_gradients_with_ops=True)
            with tf.control_dependencies([train_op]):
                train_op = tf.group(*post_ops)

            # Tuning op: only retrain batch norm.
            skip_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            classifier(x_in, training=True)
            train_bn = tf.group(*[v for v in tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                                  if v not in skip_ops])

            return EasyDict(
                x=x_in, label=l_in, train_op=train_op, tune_op=train_bn,
                classify_raw=tf.nn.softmax(classifier(x_in, training=False)),  # No EMA, for debugging.
                classify_op=tf.nn.softmax(classifier(x_in, getter=ema_getter, training=False)),
                sup_gradient = sup_gradient)
        else:
            # No EMA
            post_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            train_op = tf.train.AdamOptimizer(lr).minimize(loss_xe, colocate_gradients_with_ops=True)
            with tf.control_dependencies([train_op]):
                train_op = tf.group(*post_ops)

            # Tuning op: only retrain batch norm.
            skip_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            classifier(x_in, training=True)
            train_bn = tf.group(*[v for v in tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                                  if v not in skip_ops])

            return EasyDict(
                x=x_in, label=l_in, train_op=train_op, tune_op=train_bn,
                classify_raw=tf.nn.softmax(classifier(x_in, training=False)),  # No EMA, for debugging.
                classify_op=tf.nn.softmax(classifier(x_in, training=False)),  # No EMA by rule.
                sup_gradient = sup_gradient)


def main(argv):
    del argv  # Unused.
    dataset = DATASETS[FLAGS.dataset]() #create()
    log_width = utils.ilog2(dataset.width)

    #generating model directory
    if FLAGS.regularizer == 'l2':
        model_dir = 'l2_' + str(FLAGS.gamma)
    elif FLAGS.regularizer == 'maxsup':
        model_dir = 'maxsup_' + str(FLAGS.LH)
    elif FLAGS.regularizer == 'None':
        model_dir = 'mixup'
    else:
        assert False, 'Type of regularizer must be either: None, maxsup, maxl2, l2'

    model = MixupGrad(
        os.path.join(FLAGS.train_dir, model_dir, dataset.name),
        dataset,
        lr=FLAGS.lr,
        wd=FLAGS.wd,
        arch=FLAGS.arch,
        batch=FLAGS.batch,
        nclass=dataset.nclass,
        ema=FLAGS.ema,
        beta=FLAGS.beta,
        gamma=FLAGS.gamma,
        regularizer=FLAGS.regularizer,
        LH=FLAGS.LH,
        scales=FLAGS.scales or (log_width - 2),
        filters=FLAGS.filters,
        repeat=FLAGS.repeat)
    model.train(FLAGS.nepoch*FLAGS.epochsize, FLAGS.epochsize) #(total # of data, epoch size)
    # Gradient ??
    #model.train(FLAGS.train_kimg << 10, FLAGS.report_kimg << 10)


if __name__ == '__main__':
    utils.setup_tf()
    flags.DEFINE_float('wd', 0.002, 'Weight decay.')
    flags.DEFINE_float('ema', 0.999, 'Exponential moving average of params.')
    flags.DEFINE_float('beta', 0.5, 'Mixup beta distribution.')
    flags.DEFINE_integer('scales', 0, 'Number of 2x2 downscalings in the classifier.')
    flags.DEFINE_integer('filters', 32, 'Filter size of convolutions.')
    flags.DEFINE_integer('repeat', 4, 'Number of residual layers per stage.')
    flags.DEFINE_integer('nepoch', 1 << 7, 'Number of training epochs')
    flags.DEFINE_integer('epochsize', 1 << 16, 'Size of 1 epoch')
    flags.DEFINE_float('LH', 1, 'Lipschitz upper bound')
    flags.DEFINE_float('gamma', 1, 'Regularization parameter')
    flags.DEFINE_float('amount', 0, 'Probability of being salt or pepper')
    flags.DEFINE_string('regularizer', 'None', 'Type of regularizer: None, maxsup, maxl2, l2')
    flags.DEFINE_string('mode', '', 'Type of regularizer: None, maxsup, maxl2, l2')
    FLAGS.set_default('dataset', 'cifar10-1')
    FLAGS.set_default('batch', 64)
    FLAGS.set_default('lr', 0.002)
    #FLAGS.set_default('train_kimg', 1 << 16)
    app.run(main)
