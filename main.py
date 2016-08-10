import json
import math
import argparse
import os

import tensorflow as tf
from tqdm import tqdm

from trainer import Trainer
from read_data import load_metadata, read_data
from model import Model
from evaluator import Evaluator


def main(config):
    if config.train:
        _train(config)
    else:
        _test(config)


def _train(config):
    load_metadata(config, 'train')  # this updates the config file according to metadata file
    train_data = read_data(config, 'train')
    # dev_data = read_data(config, 'dev')

    # create directories
    if not os.path.exists(config.save_dir):
        os.mkdir(config.save_dir)

    # construct model graph and variables (using default graph)
    model = Model(config)
    trainer = Trainer(config, model)
    evaluator = Evaluator(config, model)

    # Variables
    sess = tf.Session()
    model.initialize(sess)

    # begin training
    num_steps = config.num_steps or int(config.num_epochs * train_data.num_examples / config.batch_size)
    num_steps_per_epoch = math.ceil(train_data.num_examples / config.batch_size)
    for i, batch in tqdm(enumerate(train_data.get_batches(config.batch_size, num_batches=num_steps)), total=num_steps):
        trainer.step(sess, batch, write=(i % config.log_period == 0))

        # Occasional evaluation and saving
        if i % config.eval_period == 0:
            e = evaluator.get_evaluation(sess, tqdm(train_data.get_batches(config.batch_size), total=num_steps_per_epoch), write=True)
            print(e)
        if i % config.save_period == 0:
            model.save(sess)


def _test(config):
    load_metadata(config, 'test')  # this updates the config file according to metadata file
    test_data = read_data(config, 'test')

    model = Model(config)
    evaluator = Evaluator(config, model)

    sess = tf.Session()
    model.initialize(sess)

    e = evaluator.get_evaluation(sess, tqdm(test_data.get_batches(config.batch_size)))


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
