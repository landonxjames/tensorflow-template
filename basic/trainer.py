import tensorflow as tf

from basic.model import Model


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
        opt_op = self.opt.apply_gradients(self.grads, global_step=self.global_step)
        self.train_op = opt_op

        # enforce other dependencies to the train_op
        # EMA dependency
        with tf.control_dependencies([opt_op]):
            self.train_op = tf.group(model.ema_op)

    def get_train_op(self):
        return self.train_op

    def step(self, sess, batch, get_summary=False):
        assert isinstance(sess, tf.Session)
        feed_dict = self.model.get_feed_dict(batch)
        if get_summary:
            summary_op = tf.merge_all_summaries()
            loss, summary, train_op = \
                sess.run([self.loss, summary_op, self.train_op], feed_dict=feed_dict)
        else:
            loss, train_op = sess.run([self.loss, self.train_op], feed_dict=feed_dict)
            summary = None
        return loss, summary, train_op
