import tensorflow as tf

from model import Model


class Trainer(object):
    def __init__(self, config, model):
        assert isinstance(model, Model)
        opt = tf.train.AdagradOptimizer(config.init_lr)
        loss = model.get_loss()
        var_list = model.get_var_list()
        global_step = model.get_global_step()
        grads = opt.compute_gradients(loss, var_list=var_list)
        train_op = opt.apply_gradients(grads, global_step=global_step)
        self.train_op = train_op

    def get_train_op(self):
        return self.train_op
