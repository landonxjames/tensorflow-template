import tensorflow as tf


class Model(object):
    def __init__(self, config):
        self.config = config
        self.global_step = tf.get_variable('global_step', shape=[], dtype='int32',
                                           initializer=tf.constant_initializer(0), trainable=False)
        self.loss = None
        self.var_list = None
        self._build_forward()
        if config.supervised:
            self._build_loss()

    def _build_forward(self):
        pass

    def _build_loss(self):
        loss = tf.no_op()
        self.loss = loss

    def get_loss(self):
        return self.loss

    def get_global_step(self):
        return self.global_step

    def get_var_list(self):
        return self.var_list

    def _get_feed_dict(self, batch):
        return {}

    def step(self, sess, train_op, batch):
        assert isinstance(sess, tf.Session)
        feed_dict = self._get_feed_dict(batch)
        loss = tf.get_collection('loss')
        return sess.run([loss, train_op], feed_dict=feed_dict)

    def log(self, sess):
        pass

    def save(self, sess):
        pass

    def load(self, sess):
        pass

    def eval(self, sess, batch, Evaluator):
        pass