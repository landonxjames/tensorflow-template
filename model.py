import os

import tensorflow as tf


class Model(object):
    def __init__(self, config):
        self.config = config
        self.writer = None
        self.saver = tf.train.Saver()
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

    def get_feed_dict(self, batch):
        return {}

    def initialize(self, sess):
        if self.config.load:
            self._load(sess)
        else:
            sess.run(tf.initialize_all_variables())

        if self.config.train:
            self.writer = tf.train.SummaryWriter(self.config.log_dir)

    def save(self, sess):
        self.saver.save(sess, self.config.save_dir, self.global_step)

    def _load(self, sess):
        save_dir = self.config.save_dir
        checkpoint = tf.train.get_checkpoint_state(save_dir)
        assert checkpoint is not None, "Cannot load checkpoint at {}".format(save_dir)
        self.saver.restore(sess, checkpoint.model_checkpoint_path)

