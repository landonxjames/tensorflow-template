import numpy as np


def pad(array, inc):
    assert len(array.shape) > 0, "Array must be at least 1D!"
    if len(array.shape) == 1:
        return np.concatenate([array, np.zeros([inc])], 0)
    else:
        return np.concatenate([array, np.zeros([inc, array.shape[1]])], 0)
