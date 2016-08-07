import json
import argparse

import tensorflow as tf
from tqdm import tqdm

from trainer import Trainer, MultiGPUTrainer
from read_data import load_metadata, read_data
from model import Model
from evaluator import Evaluator


def multi_gpu_main(config):
    if config.train:
        _train(config)
    else:
        _test(config)


def _train(config):
    load_metadata(config)  # this updates the config file according to metadata file
    train_data = read_data(config, 'train')
    # dev_data = read_data(config, 'dev')

    # use default graph
    # Define several models for each GPU
    models = []
    for i in range(config.num_gpus):
        with tf.name_scope("gpu_{}".format(i)), tf.device("/gpu:{}".format(i)):
            each_model = Model(config)
            models.append(each_model)
            tf.get_variable_scope().reuse_variables()
    model = models[0]

    trainer = MultiGPUTrainer(config, models)
    sess = tf.Session()
    if config.load:
        model.load(sess)

    for i, batches in tqdm(enumerate(train_data.get_multi_batches(num_steps=config.num_steps))):
        # model_temp.get_loss(sess, batch)
        trainer.step(sess, batches)

        # Occasional evaluation, logging, and saving
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
    print(sum(model.eval(sess, batch, Evaluator) for batch in tqdm(test_data.get_batches())))


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
