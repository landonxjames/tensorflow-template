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


class MultiGPUTrainer(object):
    def __init__(self, config, models):
        self.config = config
        self.models = models
        ref_model = models[0]
        global_step = ref_model.get_global_step()
        var_list = ref_model.get_var_list()
        opt = tf.train.AdagradOptimizer(config.init_lr)
        grads_list = []
        for model in models:
            assert isinstance(model, Model)
            loss = model.get_loss()
            grads = opt.compute_gradients(loss, var_list=var_list)
            grads_list.append(grads)

        train_op = opt.apply_gradients(grads, global_step=global_step)
        self.train_op = train_op
        self.loss = model.get_loss()  # FIXME : average of losses

    def get_train_op(self):
        return self.train_op

    def step(self, sess, batches):
        assert isinstance(sess, tf.Session)
        feed_dicts = (model.get_feed_dict(batch) for model, batch in zip(self.models, batches))
        feed_dict = dict(itertools.chain(feed_dict.items() for feed_dict in feed_dicts))
        return sess.run([self.loss, self.train_op], feed_dict=feed_dict)
