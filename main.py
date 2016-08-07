import json
import argparse

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
    load_metadata(config)  # this updates the config file according to metadata file
    train_data = read_data(config, 'train')
    # dev_data = read_data(config, 'dev')

    # use default graph
    model = Model(config)
    trainer = Trainer(config, model)
    sess = tf.Session()

    for i, batch in tqdm(enumerate(train_data.get_batches(num_steps=config.num_steps))):
        # model_temp.get_loss(sess, batch)
        model.step(sess, trainer.get_train_op(), batch)

        if i % config.eval_period == 0:
            print(sum(model.eval(sess, batch, Evaluator) for batch in tqdm(train_data.get_batches())))
        if i % config.log_period == 0:
            model.log(sess)
        if i % config.save_period == 0:
            model.save(sess)


def _test(config):
    test_data = read_data(config, 'test')

    model = Model(config)
    sess = tf.Session()
    model.load(sess)
    print(model.eval(sess, test_data, Evaluator))


def _get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path")
    return parser.parse_args()


def _run():
    args = _get_args()
    with open(args.config_path, 'r') as fh:
        config = json.load(fh)
        main(config)


if __name__ == "__main__":
    _run()
