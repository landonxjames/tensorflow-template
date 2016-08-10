class Evaluation(object):
    def __init__(self, data_name, step_idx, acc):
        self.data_name = data_name
        self.step_idx = step_idx
        self.acc = acc

    def __repr__(self):
        return "accuracy on {} at step {}: {}".format(self.data_name, self.step_idx, self.acc)

    def __add__(self, other):
        pass


class Evaluator(object):
    def __init__(self, config, model):
        self.config = config
        self.model = model
        self.input_names = ('x', 'y')

    def get_evaluation(self, sess, batches, write=False):
        return Evaluation('train', 0, 0.0)

