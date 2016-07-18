import argparse
import json
import os
import numpy as np
from collections import OrderedDict


def get_args():
    parser = argparse.ArgumentParser()
    home = os.path.expanduser("~")
    source_dir = os.path.join(home, "data", "model")
    target_dir = "data/model"
    parser.add_argument("--source_dir", default=source_dir)
    parser.add_argument("--target_dir", default=target_dir)
    # TODO : put more args here
    return parser.parse_args()


def prepro(args):
    source_dir = args.source_dir
    target_dir = args.target_dir
    # TODO : put something here; Fake data shown
    sizes = {'train': 700, 'dev': 100, 'test': 200}
    std = 0.1
    all_data = {}
    for mode, size in sizes.items():
        Y = [0] * size + [1] * size
        X = np.random.normal(0, std, size).tolist() + np.random.normal(1, std, size).tolist()
        all_data[mode] = {'X': X, 'Y': Y}

    for mode, each_data in all_data.items():
        _save(target_dir, mode, each_data)


def _save(target_dir, mode, data):
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    metadata_path = os.path.join(target_dir, "{}_metadata.json".format(mode))
    data_path = os.path.join(target_dir, "{}_data.json".format(mode))

    metadata = {'num_classes': len(set(data['Y']))}

    with open(metadata_path, 'w') as fh:
        json.dump(metadata, fh)
    with open(data_path, 'w') as fh:
        json.dump(data, fh)


def main():
    args = get_args()
    prepro(args)


if __name__ == "__main__":
    main()
