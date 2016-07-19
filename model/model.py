import tensorflow as tf
from model.base_model import BaseTower
import numpy as np

from my.tensorflow.nn import linear


class Tower(BaseTower):
    def _initialize_forward(self):
        params = self.params
        ph = self.placeholders
        tensors = self.tensors
        N = params.batch_size

        is_train = tf.placeholder('bool', shape=[], name='is_train')
        # TODO : define placeholders and put them in ph
        num_classes = params.num_classes
        x = tf.placeholder("float", shape=[N, 1], name='x')
        ph['x'] = x
        ph['is_train'] = is_train

        # TODO : put your codes here
        with tf.variable_scope("main"):
            logits = linear([x], num_classes, True, scope='logits')
            yp = tf.cast(tf.argmax(logits, 1), 'int32')
            tensors['logits'] = logits
            tensors['yp'] = yp

    def _initialize_supervision(self):
        params = self.params
        tensors = self.tensors
        ph = self.placeholders
        N = params.batch_size
        y = tf.placeholder("int32", shape=[N], name='y')
        y_mask = tf.placeholder("bool", shape=[N], name='y_mask')
        ph['y'] = y
        ph['y_mask'] = y_mask

        logits = tensors['logits']
        yp = tensors['yp']

        with tf.name_scope("eval"):
            correct = tf.logical_and(tf.equal(yp, y), y_mask)
            wrong = tf.logical_and(tf.not_equal(yp, y), y_mask)
            # TODO : this must be properly defined
            tensors['correct'] = correct
            tensors['wrong'] = wrong

        with tf.name_scope("loss"):
            ce = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, y, name='ce')
            avg_ce = tf.reduce_mean(ce, name='avg_ce')
            tf.add_to_collection('losses', avg_ce)

            losses = tf.get_collection('losses')
            loss = tf.add_n(losses, name='loss')
            # TODO : this must be properly defined
            tensors['loss'] = loss

    def _get_feed_dict(self, batch, mode, **kwargs):
        params = self.params
        ph = self.placeholders
        N = params.batch_size
        # TODO : put more parameters

        # TODO : define your inputs to _initialize here
        x = np.zeros([N, 1], dtype='float')
        feed_dict = {ph['x']: x,
                     ph['is_train']: mode == 'train'}

        if params.supervise:
            y = np.zeros([N], dtype='int32')
            y_mask = np.zeros([N], dtype='bool')
            feed_dict[ph['y']] = y
            feed_dict[ph['y_mask']] = y_mask

        # Batch can be empty in multi GPU parallelization
        if batch is None:
            return feed_dict

        # TODO : retrieve data and store it in the numpy arrays; example shown below
        X, Y = batch['X'], batch['Y']

        for i, xx in enumerate(X):
            x[i, 0] = xx

        if params.supervise:
            for i, yy in enumerate(Y):
                y[i] = yy
                y_mask[i] = True

        return feed_dict
