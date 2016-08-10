import numpy as np

from read_data import DataSet


class AccuracyEvaluation(object):
    def __init__(self, data_type, global_step, correct, loss):
        self.data_type = data_type
        self.global_step = global_step
        self.loss = loss
        self.correct = correct
        self.num_examples = len(correct)
        self.acc = sum(correct) / len(correct)

    def __repr__(self):
        return "step {}: accuracy={}, loss={}".format(self.global_step, self.acc, self.loss)

    def __add__(self, other):
        if other == 0:
            return self
        assert self.data_type == other.data_type
        assert self.global_step == other.global_step
        new_correct = np.append(self.correct, other.correct)
        new_loss = (self.loss * self.num_examples + other.loss * other.num_examples) / len(new_correct)
        return AccuracyEvaluation(self.data_type, self.global_step, new_correct, new_loss)

    def __radd__(self, other):
        return self.__add__(other)


class Evaluator(object):
    def __init__(self, config, model):
        self.config = config
        self.model = model

    def get_evaluation(self, sess, data_set, write=False):
        assert isinstance(data_set, DataSet)
        feed_dict = self.model.get_feed_dict(data_set)
        global_step, logits, loss = sess.run([self.model.global_step, self.model.logits, self.model.loss], feed_dict=feed_dict)
        logits = logits[:data_set.num_examples]
        correct = np.argmax(logits, 1) == np.array(data_set.data['Y'])
        e = AccuracyEvaluation(data_set.data_type, global_step, correct, loss)
        return e

    def get_evaluation_from_batches(self, sess, batches, write=False):
        return sum(self.get_evaluation(sess, batch) for batch in batches)