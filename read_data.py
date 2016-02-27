import numpy as np


class DataSet(object):
    def __init__(self, name, batch_size, data, idxs, include_leftover=False):
        self.name = name
        self.num_epochs_completed = 0
        self.idx_in_epoch = 0
        self.batch_size = batch_size
        self.data = data
        self.include_leftover = include_leftover
        self.idxs = idxs
        self.num_examples = len(idxs)
        self.num_batches = self.num_examples / self.batch_size + int(self.include_leftover)
        self.reset()

    def get_next_labeled_batch(self):
        assert self.has_next_batch(), "End of data, reset required."
        from_, to = self.idx_in_epoch, self.idx_in_epoch + self.batch_size
        if self.include_leftover and to > self.num_examples:
            to = self.num_examples
        cur_idxs = self.idxs[from_:to]
        batch = [[each[i] for i in cur_idxs] for each in self.data]
        self.idx_in_epoch += self.batch_size
        return batch

    def has_next_batch(self):
        if self.include_leftover:
            return self.idx_in_epoch + 1 < self.num_examples
        return self.idx_in_epoch + self.batch_size <= self.num_examples

    def complete_epoch(self):
        self.reset()
        self.num_epochs_completed += 1

    def reset(self):
        self.idx_in_epoch = 0
        np.random.shuffle(self.idxs)


def read_data(params, data_dir, name):
    batch_size = params.batch_size
    data = []  # TODO : override!
    idxs = []  # TODO : override!
    include_leftover = not params.train
    data_set = DataSet(name, batch_size, data, idxs, include_leftover=include_leftover)
    return data_set
