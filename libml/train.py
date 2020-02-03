"""
Training loop, checkpoint saving and loading, evaluation code.
"""

import json
import os
import os.path
import shutil

import numpy as np
import tensorflow as tf
from absl import flags
from easydict import EasyDict
from tqdm import trange
from PIL import Image

from libml import data, utils
from libml.noise import random_noise

FLAGS = flags.FLAGS
flags.DEFINE_string('train_dir', './experiments',
                    'Folder where to save training data.')
flags.DEFINE_float('lr', 0.002, 'Learning rate.')
flags.DEFINE_integer('batch', 64, 'Batch size.')
flags.DEFINE_integer('train_kimg', 1 << 14, 'Training duration in kibi-samples.')
flags.DEFINE_integer('report_kimg', 64, 'Report summary period in kibi-samples.')
flags.DEFINE_integer('save_kimg', 64, 'Save checkpoint period in kibi-samples.')
flags.DEFINE_integer('keep_ckpt', 50, 'Number of checkpoints to keep.')
flags.DEFINE_string('eval_ckpt', '', 'Checkpoint to evaluate. If provided, do not do training, just do eval.')
flags.DEFINE_integer('saveperepoch', 10, 'epochs per Save checkpoint')

#noise arguments
flags.DEFINE_float('noise_p', None, 'Proportion of image pixels to replace with noise on range [0, 1]. Used in salt, pepper, and salt & pepper.')
flags.DEFINE_integer('noise_seed', None, 'If provided, this will set the random seed before generating noise, for valid pseudo-random comparisons')

class Model:
    def __init__(self, train_dir: str, dataset: data.DataSet, **kwargs):
        self.train_dir = train_dir
        self.params = EasyDict(kwargs)
        self.dataset = dataset
        self.session = None
        self.tmp = EasyDict(print_queue=[], cache=EasyDict())
        self.step = tf.train.get_or_create_global_step()
        self.ops = self.model(**kwargs)
        self.ops.update_step = tf.assign_add(self.step, FLAGS.batch)
        self.add_summaries(**kwargs)

        #Initialize accuracies.txt
        if os.path.exists(self.train_dir + "/accuracies.txt"):
            with open(self.train_dir + "/accuracies.txt", 'r') as infile:
                self.accuracies = json.loads(infile.read())
        else:
            self.accuracies = {}

        #Initialize noise.txt
        if os.path.exists(FLAGS.eval_ckpt[:-23] + "/noise.txt"):
            with open(FLAGS.eval_ckpt[:-23] + "/noise.txt", 'r') as infile:
                self.noise = json.loads(infile.read())
        else:
            self.noise = {}

        #Print model Config.
        print()
        print(' Config '.center(80, '-'))
        print('train_dir', self.train_dir)
        print('%-32s %s' % ('Model', self.__class__.__name__))
        print('%-32s %s' % ('Dataset', dataset.name))
        for k, v in sorted(kwargs.items()):
            print('%-32s %s' % (k, v))
        print(' Model '.center(80, '-'))
        to_print = [tuple(['%s' % x for x in (v.name, np.prod(v.shape), v.shape)]) for v in utils.model_vars(None)]
        to_print.append(('Total', str(sum(int(x[1]) for x in to_print)), ''))
        sizes = [max([len(x[i]) for x in to_print]) for i in range(3)]
        fmt = '%%-%ds  %%%ds  %%%ds' % tuple(sizes)
        for x in to_print[:-1]:
            print(fmt % x)
        print()
        print(fmt % to_print[-1])
        print('-' * 80)
        if not FLAGS.eval_ckpt:
            self._create_initial_files()
        else:
            print('-'*50)
            print('Evaluation mode')
            print('-'*50)
            pass

    #Creating /args /tf in train.dir
    @property
    def arg_dir(self):
        return os.path.join(self.train_dir, 'args')

    @property
    def checkpoint_dir(self):
        return os.path.join(self.train_dir, 'tf')

    def train_print(self, text):
        self.tmp.print_queue.append(text)

    def _create_initial_files(self):
        for dir in (self.checkpoint_dir, self.arg_dir):
            if not os.path.exists(dir):
                os.makedirs(dir)
        self.save_args()

    def _reset_files(self):
        shutil.rmtree(self.train_dir)
        self._create_initial_files()

    def save_args(self, **extra_params):
        with open(os.path.join(self.arg_dir, 'args.json'), 'w') as f:
            json.dump({**self.params, **extra_params}, f, sort_keys=True, indent=4)

    @classmethod
    def load(cls, train_dir):
        with open(os.path.join(train_dir, 'args/args.json'), 'r') as f:
            params = json.load(f)
        instance = cls(train_dir=train_dir, **params)
        instance.train_dir = train_dir
        return instance

    def experiment_name(self, **kwargs):
        args = [x + str(y) for x, y in sorted(kwargs.items())]
        return '_'.join([self.__class__.__name__] + args)

    def eval_mode(self, ckpt=None):
        self.session = tf.Session(config=utils.get_config())
        saver = tf.train.Saver()

        print('The cpkt path is : ', ckpt)
        saver.restore(self.session, ckpt)
        self.tmp.step = self.session.run(self.step)
        print('Eval model %s at global_step %d imgs' % (self.__class__.__name__, self.tmp.step))
        return self

    def model(self, **kwargs):
        raise NotImplementedError()

    def add_summaries(self, **kwargs):
        raise NotImplementedError()


class Model_clf(Model):
    """Base model for classification."""

    def __init__(self, train_dir: str, dataset: data.DataSet, nclass: int, **kwargs):
        self.nclass = nclass
        Model.__init__(self, train_dir, dataset, nclass=nclass, **kwargs)

    def train_step(self, train_session, data_labeled):
        # Load data
        x = self.session.run(data_labeled)
        # Update parameters
        self.tmp.step = train_session.run([self.ops.train_op, self.ops.update_step],
                                          feed_dict={self.ops.x: x['image'],
                                                     self.ops.label: x['label']})[1]

    def train(self, train_nimg, report_nimg, **kwargs):
        if FLAGS.eval_ckpt:
            if FLAGS.noise_p:
                print("Evaluating the accuracy with noisy train, valid, test images...")
                self.eval_noise(FLAGS.eval_ckpt, **kwargs)
            else:
                print("Evaluating the gradients with test images...")
                self.eval_checkpoint(FLAGS.eval_ckpt)
            return
        batch = FLAGS.batch

        train_labeled = self.dataset.train_labeled.batch(batch).prefetch(16)
        train_labeled = train_labeled.make_one_shot_iterator().get_next()
        scaffold = tf.train.Scaffold(saver=tf.train.Saver(max_to_keep=FLAGS.keep_ckpt,
                                                          pad_step_number=10))

        with tf.Session(config=utils.get_config()) as sess:
            self.session = sess
            self.cache_eval()

        with tf.train.MonitoredTrainingSession(
                scaffold=scaffold,
                checkpoint_dir=self.checkpoint_dir,
                config=utils.get_config(),
                save_checkpoint_steps=FLAGS.saveperepoch*report_nimg,
                save_summaries_steps=report_nimg - batch) as train_session:
            self.session = train_session._tf_sess()
            self.tmp.step = self.session.run(self.step)

            while self.tmp.step < train_nimg:
                loop = trange(self.tmp.step % report_nimg, report_nimg, batch,
                              leave=False, unit='img', unit_scale=batch,
                              desc='Epoch %d/%d' % (1 + (self.tmp.step // report_nimg), train_nimg // report_nimg))
                for _ in loop:
                    self.train_step(train_session, train_labeled)
                    while self.tmp.print_queue:
                        loop.write(self.tmp.print_queue.pop(0))
            while self.tmp.print_queue:
                print(self.tmp.print_queue.pop(0))

    def tune(self, train_nimg):
        batch = FLAGS.batch
        train_labeled = self.dataset.train_labeled.batch(batch).prefetch(16)
        train_labeled = train_labeled.make_one_shot_iterator().get_next()
        for _ in trange(0, train_nimg, batch, leave=False, unit='img', unit_scale=batch, desc='Tuning'):
            x = self.session.run([train_labeled])
            self.session.run([self.ops.tune_op], feed_dict={self.ops.x: x['image'],
                                                            self.ops.label: x['label']})

    def eval_checkpoint(self, ckpt=None):
        self.eval_mode(ckpt)
        self.cache_eval()
        ema, clean_correctness = self.eval_stats(classify_op=self.ops.classify_op)
        gradients = self.eval_sup_gradients(sup_gradients=self.ops.sup_gradients, ckpt=ckpt)
        print('%16s %8s %8s %8s' % ('', 'labeled', 'valid', 'test'))
        print('%16s %8s %8s %8s' % (('ema',) + tuple('%.2f' % x for x in ema)))
        if os.path.exists(os.path.join(ckpt[:-23], 'gradients_{}.txt'.format(ckpt[-8:]))) is False:
            with open(os.path.join(ckpt[:-23], 'gradients_{}.txt'.format(ckpt[-8:])), 'w') as outfile:
                json.dump({'gradients': gradients.tolist()}, outfile)
            print('gradients_{}.txt is saved at {}'.format(ckpt[-8:], ckpt[:-23]))


    def eval_noise(self, ckpt=None):
        self.eval_mode(ckpt)
        self.cache_eval()
        ema, clean_correctness = self.eval_stats(classify_op=self.ops.classify_op)
        gradients = self.eval_sup_gradients(sup_gradients=self.ops.sup_gradients, ckpt=ckpt)
        ema_noise, noise_settings, noisy_correctness = self.eval_stats_noise(classify_op=self.ops.classify_op)
        print('%16s %8s %8s %8s' % ('', 'labeled', 'valid', 'test'))
        print('%16s %8s %8s %8s' % (('ema',) + tuple('%.2f' % x for x in ema)))
        print('%16s %8s %8s %8s' % (('ema_noise',) + tuple('%.2f' % x for x in ema_noise)))
        #Saving noise.txt
        noise_settings.update({'steps':ckpt[-8:]})
        self.noise.update({str(noise_settings):{'noise_acc':ema_noise.tolist(), 'clean_acc':ema.tolist(), 'clean_correctness':clean_correctness.tolist(), 'noisy_correctness':noisy_correctness.tolist()}})
        with open(os.path.join(ckpt[:-23], 'noise.txt'), 'w') as outfile:
            json.dump(self.noise, outfile)

    def cache_eval(self):
        """Cache datasets for computing eval stats."""

        def collect_samples(dataset):
            """Return numpy arrays of all the samples from a dataset."""
            it = dataset.batch(1).prefetch(16).make_one_shot_iterator().get_next()
            images, labels = [], []
            while 1:
                try:
                    v = self.session.run(it)
                except tf.errors.OutOfRangeError:
                    break
                images.append(v['image'])
                labels.append(v['label'])

            images = np.concatenate(images, axis=0)
            labels = np.concatenate(labels, axis=0)
            return images, labels

        if 'test' not in self.tmp.cache:
            self.tmp.cache.test = collect_samples(self.dataset.test)
            self.tmp.cache.valid = collect_samples(self.dataset.valid)
            self.tmp.cache.train_labeled = collect_samples(self.dataset.eval_labeled)

    def eval_stats(self, batch=None, feed_extra=None, classify_op=None):
        """Evaluate model on train, valid and test."""
        batch = batch or FLAGS.batch
        classify_op = self.ops.classify_op if classify_op is None else classify_op
        accuracies = []
        for subset in ('train_labeled', 'valid', 'test'):
            images, labels = self.tmp.cache[subset]
            predicted = []
            for x in range(0, images.shape[0], batch):
                p = self.session.run(
                    classify_op,
                    feed_dict={
                        self.ops.x: images[x:x + batch],
                        **(feed_extra or {})
                    })
                predicted.append(p)
            predicted = np.concatenate(predicted, axis=0)
            if subset == 'test':
                clean_correctness = predicted.argmax(1) == labels
            accuracies.append((predicted.argmax(1) == labels).mean() * 100)  #[train, valid, test]
        #print accuracies
        self.train_print('%-5d k imgs  accuracy train/valid/test  %.2f  %.2f  %.2f' %
                         tuple([self.tmp.step >> 10] + accuracies))
        self.accuracies['epoch' + str(self.tmp.step // FLAGS.epochsize)] = accuracies
        #Saving accurcies.txt
        if FLAGS.eval_ckpt:
            return np.array(accuracies, 'f'), clean_correctness
        with open(os.path.join(self.train_dir, 'accuracies.txt'), 'w') as outfile:
            json.dump(self.accuracies, outfile)
        return np.array(accuracies, 'f')

    def eval_stats_noise(self, batch=None, feed_extra=None, classify_op=None, **kwargs):
        """Evaluate model on noisy train, valid and test."""
        batch = batch or FLAGS.batch
        classify_op = self.ops.classify_op if classify_op is None else classify_op
        accuracies =  []
        for subset in ('train_labeled', 'valid', 'test'):
            images, labels = self.tmp.cache[subset]

            # Saving an example clean image
            print('saving an example of original {} image...'.format(subset))
            image_directory = os.path.join(self.train_dir, 'example_images')
            if os.path.exists(image_directory) is not True:
                try:
                    os.mkdir(image_directory)
                except OSError:
                    print("Creation of the directory %s failed" % image_directory)
            example_image = (images[0]+1)*255/2
            n = images.shape[0] #number of images
            marginal_raw = np.ones((n,))/n
            marginal_noise = np.ones((n,))/n
            img = Image.fromarray(example_image.astype('uint8'))
            img.save(self.train_dir + '/example_images/{}.png'.format(subset))

            # Adding noise
            if FLAGS.noise_p > 0:
                print("{}- p:{}, seed:{}".format(subset, FLAGS.noise_p, FLAGS.noise_seed))
                raw_image_vectors = images.reshape((images.shape[0],-1)) # (number of images, width*height*depth)
                images = random_noise(images, mode='s&p', amount=FLAGS.noise_p, seed=FLAGS.noise_seed)
                noisy_image_vectors = images.reshape((images.shape[0],-1))

                #save noise settings
                noise_settings = dict(p=FLAGS.noise_p, seed=FLAGS.noise_seed)
                print('saving an example of noisy {} image...'.format(subset))
                example_image = (images[0]+1)*255/2
                img = Image.fromarray(example_image.astype('uint8'))
                img.save(self.train_dir + '/example_images/{}_{}.png'.format(FLAGS.noise_p, subset))
            else:
                assert False, "Please check noise_p."

            predicted = []
            for x in range(0, images.shape[0], batch):
                p = self.session.run(
                    classify_op,
                    feed_dict={
                        self.ops.x: images[x:x + batch],
                        **(feed_extra or {})
                    })
                predicted.append(p)
            predicted = np.concatenate(predicted, axis=0)
            if subset == 'test':
                noisy_correctness = predicted.argmax(1) == labels
            accuracies.append((predicted.argmax(1) == labels).mean() * 100)
        return np.array(accuracies, 'f'), noise_settings, noisy_correctness


    def eval_sup_gradients(self, ckpt, batch=None, feed_extra=None, sup_gradients=None):
        """Evaluate sup norm of gradients of h(x,y)=loss(f(x),y) on test images."""
        batch = batch or FLAGS.batch
        sup_gradient = self.ops.sup_gradients if sup_gradients is None else sup_gradients
        predicted = []
        x_test, y_test = self.tmp.cache['test']
        for i in range(0, x_test.shape[0], batch):
            l = self.session.run(
                sup_gradient,
                feed_dict={
                    self.ops.x: x_test[i:i+batch],
                    self.ops.label: y_test[i:i+batch],
                    **(feed_extra or {})
                })
            predicted.append(l)
        predicted = np.concatenate(predicted, axis=0)
        return predicted


    def add_summaries(self, feed_extra=None, **kwargs):
        del kwargs

        def gen_stats():
            return self.eval_stats(feed_extra=feed_extra)

        accuracies = tf.py_func(gen_stats, [], tf.float32)
        tf.summary.scalar('accuracy/train_labeled', accuracies[0])
        tf.summary.scalar('accuracy/valid', accuracies[1])
        tf.summary.scalar('accuracy', accuracies[2])
