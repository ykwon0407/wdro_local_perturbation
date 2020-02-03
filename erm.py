import functools
import os

from absl import app
from absl import flags
from easydict import EasyDict

from libml import utils
from libml.models import MultiModel
from libml.data import DATASETS

import tensorflow as tf

FLAGS = flags.FLAGS

class FSBaseline(MultiModel):

    def model(self, lr, wd, ema, gamma, **kwargs):
        # Define Input data variables
        hwc = [self.dataset.height, self.dataset.width, self.dataset.colors]
        x_in = tf.placeholder(tf.float32, [None] + hwc, 'x')
        l_in = tf.placeholder(tf.int32, [None], 'labels')
        wd *= lr
        l = tf.one_hot(l_in, self.nclass)

        # Define classifier in Multimodel
        x, l = self.augment(x_in, l, **kwargs)
        classifier = functools.partial(self.classifier, **kwargs)
        logits = classifier(x, training=True)

        # Define loss and gradients
        loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=l, logits=logits)
        gradient = tf.gradients(loss, x)[0]
        tf.summary.scalar('gradient/max_gradient', tf.reduce_max(tf.abs(gradient)))
        loss_main = tf.reduce_mean(loss)
        tf.summary.scalar('losses/main', loss_main)

        # Define minimizing losses
        if gamma == None:
            loss_xe = loss_main
        elif gamma > 0:
            loss_grad = gamma * tf.reduce_sum(tf.square(gradient))/tf.constant(FLAGS.batch, dtype=tf.float32)
            tf.summary.scalar('losses/gradient', loss_grad)
            loss_xe = loss_main + loss_grad
        else:
            assert False, 'Check the regularization parameter gamma'

        #sup_norm of gradients
        sup_gradients = tf.reduce_max(tf.abs(gradient), axis=[1,2,3]) #(batchsize, )

        # Define hyperparameters
        ema = tf.train.ExponentialMovingAverage(decay=ema)
        ema_op = ema.apply(utils.model_vars())
        ema_getter = functools.partial(utils.getter_ema, ema)
        post_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS) + [ema_op]
        post_ops.extend([tf.assign(v, v * (1 - wd)) for v in utils.model_vars('classify') if 'kernel' in v.name])

        # Define optimizer
        train_op = tf.train.AdamOptimizer(lr).minimize(loss_xe, colocate_gradients_with_ops=True)
        with tf.control_dependencies([train_op]):
            train_op = tf.group(*post_ops)

        return EasyDict(
            x=x_in, label=l_in, train_op=train_op,
            classify_op=tf.nn.softmax(classifier(x_in, getter=ema_getter, training=False)),
            sup_gradients=sup_gradients)



def main(argv):
    del argv  # Unused.
    dataset = DATASETS[FLAGS.dataset]()
    log_width = utils.ilog2(dataset.width)

    # Generating model directory
    if FLAGS.gamma == None:
        model_dir = 'ERM'
    elif FLAGS.gamma > 0:
        model_dir = 'WDRO_' + str(FLAGS.gamma)
    else:
        assert False, 'Check the regularization parameter gamma'

    model = FSBaseline(
        os.path.join(FLAGS.train_dir, model_dir, dataset.name),
        dataset,
        lr=FLAGS.lr,
        wd=FLAGS.wd,
        arch=FLAGS.arch,
        batch=FLAGS.batch,
        nclass=dataset.nclass,
        ema=FLAGS.ema,
        smoothing=FLAGS.smoothing,
        scales=FLAGS.scales or (log_width - 2),
        filters=FLAGS.filters,
        repeat=FLAGS.repeat,
        gamma=FLAGS.gamma)
    model.train(FLAGS.nepoch*FLAGS.epochsize, FLAGS.epochsize) #(total # of data, epoch size)

if __name__ == '__main__':
    utils.setup_tf()
    flags.DEFINE_float('wd', 0.02, 'Weight decay.')
    flags.DEFINE_float('ema', 0.999, 'Exponential moving average of params.')
    flags.DEFINE_float('smoothing', 0.001, 'Label smoothing.')
    flags.DEFINE_integer('scales', 0, 'Number of 2x2 downscalings in the classifier.')
    flags.DEFINE_integer('filters', 32, 'Filter size of convolutions.')
    flags.DEFINE_integer('repeat', 4, 'Number of residual layers per stage.')
    flags.DEFINE_integer('nepoch', 1 << 7, 'Number of training epochs')
    flags.DEFINE_integer('epochsize', 1 << 16, 'Size of 1 epoch')
    flags.DEFINE_float('gamma', None, 'Regularization parameter')
    FLAGS.set_default('dataset', 'cifar10')
    FLAGS.set_default('batch', 64)
    FLAGS.set_default('lr', 0.002)
    app.run(main)
