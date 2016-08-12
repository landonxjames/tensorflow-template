import tensorflow as tf
from tensorflow.python.ops.rnn_cell import _linear
from tensorflow.python.util import nest


def linear(args, output_size, bias, bias_start=0.0, scope=None):
    if args is None or (nest.is_sequence(args) and not args):
        raise ValueError("`args` must be specified")
    if not nest.is_sequence(args):
        args = [args]

    common_shape = [tf.shape(args[0])[i] for i in range(len(args[0].get_shape().as_list()[:-1]))]
    flat_args = [tf.reshape(arg, [-1, arg.get_shape().as_list()[-1]]) for arg in args]
    flat_out = _linear(flat_args, output_size, bias, bias_start=bias_start, scope=scope)
    out = tf.reshape(flat_out, common_shape + [output_size])
    return out
