import argparse
import json
import math
import os
import shutil

import tensorflow as tf
from tqdm import tqdm

from basic.evaluator import Evaluator
from basic.model import Model
from basic.trainer import Trainer

from basic.read_data import load_metadata, read_data


def main(config):
    if config.train:
        _train(config)
    else:
        _test(config)


def _train(config):
    load_metadata(config, 'train')  # this updates the config file according to metadata file
    set_dirs(config)
    train_data = read_data(config, 'train')
    dev_data = read_data(config, 'dev')

    # construct model graph and variables (using default graph)
    model = Model(config)
    trainer = Trainer(config, model)
    evaluator = Evaluator(config, model)

    # Variables
    sess = tf.Session()
    model.initialize(sess)

    # begin training
    num_steps = config.num_steps or int(config.num_epochs * train_data.num_examples / config.batch_size)
    for batch in tqdm(train_data.get_batches(config.batch_size, num_batches=num_steps), total=num_steps):
        global_step = sess.run(model.global_step) + 1  # +1 because all calculations are done after step
        trainer.step(sess, batch, write=(global_step % config.log_period == 0))

        # Occasional evaluation and saving
        if global_step % config.eval_period == 0:
            e = evaluator.get_evaluation_from_batches(sess, dev_data.get_batches(config.batch_size), write=True)
            e.dump(config.eval_dir)
            print(e)
        if global_step % config.save_period == 0:
            model.save(sess)


def _test(config):
    load_metadata(config, 'test')  # this updates the config file according to metadata file
    set_dirs(config)
    test_data = read_data(config, 'test')

    model = Model(config)
    evaluator = Evaluator(config, model)

    sess = tf.Session()
    model.initialize(sess)

    num_steps_per_epoch = math.ceil(test_data.num_examples / config.batch_size)
    e = evaluator.get_evaluation_from_batches(sess, tqdm(test_data.get_batches(config.batch_size), total=num_steps_per_epoch))
    e.dump(config.eval_dir)
    print(e)


def set_dirs(config):
    # create directories
    if not config.load and os.path.exists(config.out_dir):
        shutil.rmtree(config.out_dir)

    config.save_dir = os.path.join(config.out_dir, "save")
    config.log_dir = os.path.join(config.out_dir, "log")
    config.eval_dir = os.path.join(config.out_dir, "eval")
    if not os.path.exists(config.out_dir):
        os.makedirs(config.out_dir)
    if not os.path.exists(config.save_dir):
        os.mkdir(config.save_dir)
    if not os.path.exists(config.log_dir):
        os.mkdir(config.eval_dir)


def _get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path")
    return parser.parse_args()


class Config(object):
    def __init__(self, **entries):
        self.__dict__.update(entries)


def _run():
    args = _get_args()
    with open(args.config_path, 'r') as fh:
        config = Config(**json.load(fh))
        main(config)


if __name__ == "__main__":
    _run()
