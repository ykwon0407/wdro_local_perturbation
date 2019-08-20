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
"""Classifier architectures."""

import functools
import itertools
from absl import flags
from libml import layers
from libml.train import Model_clf
import tensorflow as tf
from libml import spectral_norm as sn_layers

class ConvNet(Model_clf):
    def classifier(self, x, scales, filters, getter=None, **kwargs):
        del kwargs
        conv_args = dict(kernel_size=3, activation=tf.nn.leaky_relu, padding='same')

        with tf.variable_scope('classify', reuse=tf.AUTO_REUSE, custom_getter=getter):
            y = tf.layers.conv2d(x, filters, **conv_args)
            for scale in range(scales):
                y = tf.layers.conv2d(y, filters << scale, **conv_args)
                y = tf.layers.conv2d(y, filters << (scale + 1), **conv_args)
                y = tf.layers.average_pooling2d(y, 2, 2)
            y = tf.layers.conv2d(y, self.nclass, 3, padding='same')
            logits = tf.reduce_mean(y, [1, 2])
        return logits


class ResNet(Model_clf):
    def classifier(self, x, scales, filters, repeat, training, getter=None, **kwargs):
        del kwargs
        leaky_relu = functools.partial(tf.nn.leaky_relu, alpha=0.1)
        bn_args = dict(training=training, momentum=0.999)

        def conv_args(k, f):
            return dict(padding='same',
                        kernel_initializer=tf.random_normal_initializer(stddev=tf.rsqrt(0.5 * k * k * f)))

        def residual(x0, filters, stride=1, activate_before_residual=False):
            x = leaky_relu(tf.layers.batch_normalization(x0, **bn_args))
            if activate_before_residual:
                x0 = x

            x = tf.layers.conv2d(x, filters, 3, strides=stride, **conv_args(3, filters))
            x = leaky_relu(tf.layers.batch_normalization(x, **bn_args))
            x = tf.layers.conv2d(x, filters, 3, **conv_args(3, filters))

            if x0.get_shape()[3] != filters:
                x0 = tf.layers.conv2d(x0, filters, 1, strides=stride, **conv_args(1, filters))

            return x0 + x

        with tf.variable_scope('classify', reuse=tf.AUTO_REUSE, custom_getter=getter):
            y = tf.layers.conv2d((x - self.dataset.mean) / self.dataset.std, 16, 3, **conv_args(3, 16))
            
            for scale in range(scales):
                with tf.variable_scope('{}_layer_0'.format(scale)):
                    y = residual(y, filters << scale, stride=2 if scale else 1, activate_before_residual=scale == 0)
                
                for i in range(repeat - 1):
                    with tf.variable_scope('{}_layer_{}'.format(scale, str(i+1))):
                        y = residual(y, filters << scale)
            
            y = leaky_relu(tf.layers.batch_normalization(y, **bn_args))
            y = tf.reduce_mean(y, [1, 2])
            logits = tf.layers.dense(y, self.nclass, kernel_initializer=tf.glorot_normal_initializer())
        return logits

class ShakeNet(Model_clf):
    def classifier(self, x, scales, filters, repeat, training, getter=None, **kwargs):
        del kwargs
        bn_args = dict(training=training, momentum=0.999)

        def conv_args(k, f):
            return dict(padding='same', use_bias=False,
                        kernel_initializer=tf.random_normal_initializer(stddev=tf.rsqrt(0.5 * k * k * f)))

        def residual(x0, filters, stride=1):
            def branch():
                x = tf.nn.relu(x0)
                x = tf.layers.conv2d(x, filters, 3, strides=stride, **conv_args(3, filters))
                x = tf.nn.relu(tf.layers.batch_normalization(x, **bn_args))
                x = tf.layers.conv2d(x, filters, 3, **conv_args(3, filters))
                x = tf.layers.batch_normalization(x, **bn_args)
                return x

            x = layers.shakeshake(branch(), branch(), training)

            if stride == 2:
                x1 = tf.layers.conv2d(tf.nn.relu(x0[:, ::2, ::2]), filters >> 1, 1, **conv_args(1, filters >> 1))
                x2 = tf.layers.conv2d(tf.nn.relu(x0[:, 1::2, 1::2]), filters >> 1, 1, **conv_args(1, filters >> 1))
                x0 = tf.concat([x1, x2], axis=3)
                x0 = tf.layers.batch_normalization(x0, **bn_args)
            elif x0.get_shape()[3] != filters:
                x0 = tf.layers.conv2d(x0, filters, 1, **conv_args(1, filters))
                x0 = tf.layers.batch_normalization(x0, **bn_args)

            return x0 + x

        with tf.variable_scope('classify', reuse=tf.AUTO_REUSE, custom_getter=getter):
            y = tf.layers.conv2d((x - self.dataset.mean) / self.dataset.std, 16, 3, **conv_args(3, 16))
            for scale, i in itertools.product(range(scales), range(repeat)):
                with tf.variable_scope('layer%d.%d' % (scale + 1, i)):
                    if i == 0:
                        y = residual(y, filters << scale, stride=2 if scale else 1)
                    else:
                        y = residual(y, filters << scale)

            y = tf.reduce_mean(y, [1, 2])
            logits = tf.layers.dense(y, self.nclass, kernel_initializer=tf.glorot_normal_initializer())
        return logits

class SPNet(Model_clf):
    '''
    The same network architecture with ResNet but with Spectral normalization layers.
    '''

    def classifier(self, x, scales, filters, repeat, training, BN=False, getter=None, **kwargs):
        del kwargs
        leaky_relu = functools.partial(tf.nn.leaky_relu, alpha=0.1)
        bn_args = dict(training=training, momentum=0.999)

        def conv_args(k, f):
            return dict(padding='same',
                        kernel_initializer=tf.random_normal_initializer(stddev=tf.rsqrt(0.5 * k * k * f)))

        def residual(x0, filters, stride=1, activate_before_residual=False):
            if BN is True:
                x = leaky_relu(tf.layers.batch_normalization(x0, **bn_args))
            else:
                x = leaky_relu(x0)
            if activate_before_residual:
                x0 = x

            x = sn_layers.conv2d(x, filters, 3, strides=stride, name="conv_0")
            if BN is True:
                x = leaky_relu(tf.layers.batch_normalization(x, **bn_args))
            else:
                x = leaky_relu(x)
            x = sn_layers.conv2d(x, filters, 3, name="conv_1")

            if x0.get_shape()[3] != filters:
                x0 = sn_layers.conv2d(x0, filters, 1, strides=stride, name="conv_2")

            return x0 + x
            
        with tf.variable_scope('classify', reuse=tf.AUTO_REUSE, custom_getter=getter):
            y = tf.layers.conv2d((x - self.dataset.mean) / self.dataset.std, 16, 3, **conv_args(3, 16))

            for scale in range(scales):
                with tf.variable_scope('{}_layer_0'.format(scale)):
                    y = residual(y, filters << scale, stride=2 if scale else 1, activate_before_residual=scale == 0)
                
                for i in range(repeat - 1):
                    with tf.variable_scope('{}_layer_{}'.format(scale, str(i+1))):
                        y = residual(y, filters << scale)
                        
            if BN is True:
                y = leaky_relu(tf.layers.batch_normalization(y, **bn_args))
            else:
                y = leaky_relu(y)
            y = tf.reduce_mean(y, [1, 2])
            logits = tf.layers.dense(y, self.nclass, kernel_initializer=tf.glorot_normal_initializer())
        return logits

class MultiModel(ConvNet, ResNet, ShakeNet, SPNet):
    MODEL_CONVNET, MODEL_RESNET, MODEL_SHAKE, MODEL_SP = 'convnet resnet shake spnet'.split()
    MODELS = MODEL_CONVNET, MODEL_RESNET, MODEL_SHAKE, MODEL_SP

    def augment(self, x, l, smoothing, **kwargs):
        del kwargs
        return x, l - smoothing * (l - 1. / self.nclass)

    def classifier(self, x, arch, **kwargs):
        if arch == self.MODEL_CONVNET:
            return ConvNet.classifier(self, x, **kwargs)
        elif arch == self.MODEL_RESNET:
            return ResNet.classifier(self, x, **kwargs)
        elif arch == self.MODEL_SHAKE:
            return ShakeNet.classifier(self, x, **kwargs)
        elif arch == self.MODEL_SP:
            return SPNet.classifier(self, x, **kwargs)
        raise ValueError('Model %s does not exists, available ones are %s' % (arch, self.MODELS))

# default architecture is RESNET.
flags.DEFINE_enum('arch', MultiModel.MODEL_RESNET, MultiModel.MODELS, 'Architecture.') 



"""
class ResNet_ORIGINAL(Model_clf):
    def classifier(self, x, scales, filters, repeat, training, getter=None, **kwargs):
        del kwargs
        leaky_relu = functools.partial(tf.nn.leaky_relu, alpha=0.1)
        bn_args = dict(training=training, momentum=0.999)

        def conv_args(k, f):
            return dict(padding='same',
                        kernel_initializer=tf.random_normal_initializer(stddev=tf.rsqrt(0.5 * k * k * f)))

        def residual(x0, filters, stride=1, activate_before_residual=False):
            x = leaky_relu(tf.layers.batch_normalization(x0, **bn_args))
            if activate_before_residual:
                x0 = x

            x = tf.layers.conv2d(x, filters, 3, strides=stride, **conv_args(3, filters))
            x = leaky_relu(tf.layers.batch_normalization(x, **bn_args))
            x = tf.layers.conv2d(x, filters, 3, **conv_args(3, filters))

            if x0.get_shape()[3] != filters:
                x0 = tf.layers.conv2d(x0, filters, 1, strides=stride, **conv_args(1, filters))

            return x0 + x

        with tf.variable_scope('classify', reuse=tf.AUTO_REUSE, custom_getter=getter):
            y = tf.layers.conv2d((x - self.dataset.mean) / self.dataset.std, 16, 3, **conv_args(3, 16))
            for scale in range(scales):
                y = residual(y, filters << scale, stride=2 if scale else 1, activate_before_residual=scale == 0)
                for i in range(repeat - 1):
                    y = residual(y, filters << scale)

            y = leaky_relu(tf.layers.batch_normalization(y, **bn_args))
            y = tf.reduce_mean(y, [1, 2])
            logits = tf.layers.dense(y, self.nclass, kernel_initializer=tf.glorot_normal_initializer())
        return logits
"""        