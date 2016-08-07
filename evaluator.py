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
    input_names = ('x', 'y')

    @staticmethod
    def evaluate(dict_):
        return Evaluation('train', 0, 0.0)