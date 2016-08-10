import itertools
import tensorflow as tf

from model import Model


class Trainer(object):
    def __init__(self, config, model):
        assert isinstance(model, Model)
        self.config = config
        self.model = model
        self.opt = tf.train.AdagradOptimizer(config.init_lr)
        self.loss = model.get_loss()
        self.var_list = model.get_var_list()
        self.global_step = model.get_global_step()
        self.grads = self.opt.compute_gradients(self.loss, var_list=self.var_list)
        self.train_op = self.opt.apply_gradients(self.grads, global_step=self.global_step)

    def get_train_op(self):
        return self.train_op

    def step(self, sess, batch, write=False):
        assert isinstance(sess, tf.Session)
        feed_dict = self.model.get_feed_dict(batch)
        if write:
            global_step, loss, summary, train_op = \
                sess.run([self.global_step, self.loss, self.model.summary, self.train_op], feed_dict=feed_dict)
            self.model.add_summary(summary, global_step)
        else:
            loss, train_op = sess.run([self.loss, self.train_op], feed_dict=feed_dict)
        return loss, train_op
