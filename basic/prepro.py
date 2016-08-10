import argparse
import json
import os
import itertools

import numpy as np


def get_args():
    parser = argparse.ArgumentParser()
    home = os.path.expanduser("~")
    source_dir = os.path.join(home, "data", "basic")
    target_dir = "data/basic"
    parser.add_argument("--source_dir", default=source_dir)
    parser.add_argument("--target_dir", default=target_dir)
    parser.add_argument("--size", default=1000, type=int)
    parser.add_argument("--std", default=0.5, type=float)
    # TODO : put more args here
    return parser.parse_args()


def prepro(args):
    target_dir = args.target_dir
    size = args.size
    std = args.std

    def gen(n, label):
        mean = [1, -1] if label == 0 else [1, 1]
        cov = [[std, 0], [0, std]]
        x = np.random.multivariate_normal(mean, cov, size=[n]).tolist()
        y = list(itertools.repeat(label, n))
        return x, y

    def create_data(mode, n, supervised=True):
        x0, y0 = gen(int(n/2), 0)
        x1, y1 = gen(int(n/2), 1)
        data = {'X': x0 + x1}
        if supervised:
            data['Y'] = y0 + y1
        metadata = {'num_classes': 2, 'dim': 2}
        data_path = os.path.join(target_dir, "data_{}.json".format(mode))
        metadata_path = os.path.join(target_dir, "metadata_{}.json".format(mode))
        with open(data_path, 'w') as fh:
            json.dump(data, fh)
        with open(metadata_path, 'w') as fh:
            json.dump(metadata, fh)

    if not os.path.exists(target_dir):
        os.mkdir(target_dir)

    create_data('train', int(size * 0.7))
    create_data('dev', int(size * 0.1))
    create_data('test', int(size * 0.2))
    create_data('forward', int(size * 0.5), supervised=False)


def main():
    args = get_args()
    prepro(args)

if __name__ == "__main__":
    main()
