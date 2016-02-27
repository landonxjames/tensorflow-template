from pprint import pprint
import os

import tensorflow as tf

from base_model import BaseModel
from read_data import read_data

flags = tf.app.flags

# File directories
flags.DEFINE_string("log_dir", "log", "Log directory [log]")
flags.DEFINE_string("save_dir", "save", "Save directory [save]")
flags.DEFINE_string("train_data_dir", 'data/train', "Train data directory [data/train]")
flags.DEFINE_string("val_data_dir", 'data/val', "Val data directory [data/val]")
flags.DEFINE_string("test_data_dir", 'data/eval', "Test data directory [data/eval]")

# Training parameters
flags.DEFINE_integer("batch_size", 100, "Batch size for the network [32]")
flags.DEFINE_float("init_mean", 0, "Initial weight mean [0]")
flags.DEFINE_float("init_std", 0.1, "Initial weight std [0.1]")
flags.DEFINE_integer("num_epochs", 100, "Total number of epochs for training [100]")

# Training and testing options
flags.DEFINE_boolean("train", False, "Train? Test if False [False]")
flags.DEFINE_integer("val_num_batches", 10, "Val num batches [10]")
flags.DEFINE_boolean("load", False, "Load from saved model? [False]")
flags.DEFINE_boolean("progress", True, "Show progress? [True]")
flags.DEFINE_boolean("gpu", False, 'Enable GPU? (Linux only) [False]')
flags.DEFINE_integer("val_period", 5, "Val period (for display purpose only) [5]")
flags.DEFINE_integer("save_period", 10, "Save period [10]")

# Debugging
flags.DEFINE_boolean("draft", False, "Draft? (quick build) [False]")

FLAGS = flags.FLAGS


def main(_):
    train_ds = read_data('train', FLAGS, FLAGS.train_data_dir)
    val_ds = read_data('val', FLAGS, FLAGS.val_data_dir)
    test_ds = read_data('test', FLAGS, FLAGS.test_data_dir)

    FLAGS.train_num_batches = train_ds.num_batches
    FLAGS.val_num_batches = FLAGS.val_num_batches
    FLAGS.test_num_batches = test_ds.num_batches

    if not os.path.exists(FLAGS.save_dir):
        os.mkdir(FLAGS.save_dir)

    if FLAGS.draft:
        # For quick build (deubgging).
        # Add any other parameter that requires a lot of computations
        FLAGS.train_num_batches = 1
        FLAGS.val_num_batches = 1
        FLAGS.test_num_batches = 1
        FLAGS.num_epochs = 1
        FLAGS.eval_period = 1
        FLAGS.save_period = 1

    pprint(FLAGS.__flags)
    print "training: %d, validation: %d, eval: %d" % (train_ds.num_examples, val_ds.num_examples, test_ds.num_examples)

    graph = tf.Graph()
    model = BaseModel(graph, FLAGS)
    with tf.Session(graph=graph) as sess:
        sess.run(tf.initialize_all_variables())
        if FLAGS.train:
            writer = tf.train.SummaryWriter(FLAGS.log_dir, sess.graph_def)
            if FLAGS.load:
                model.load(sess)
            model.train(sess, writer, train_ds, val_ds)
        else:
            model.load(sess)
            model.eval(sess, test_ds)

if __name__ == "__main__":
    tf.app.run()