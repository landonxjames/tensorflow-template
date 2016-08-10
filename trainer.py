import itertools
import tensorflow as tf

from model import Model


class Trainer(object):
    def __init__(self, config, model):
        assert isinstance(model, Model)
        self.config = config
        self.model = model
        opt = tf.train.AdagradOptimizer(config.init_lr)
        loss = model.get_loss()
        var_list = model.get_var_list()
        global_step = model.get_global_step()
        grads = opt.compute_gradients(loss, var_list=var_list)
        train_op = opt.apply_gradients(grads, global_step=global_step)
        self.train_op = train_op
        self.loss = model.get_loss()

    def get_train_op(self):
        return self.train_op

    def step(self, sess, batch, write=False):
        assert isinstance(sess, tf.Session)
        feed_dict = self.model.get_feed_dict(batch)
        return sess.run([self.loss, self.train_op], feed_dict=feed_dict)
