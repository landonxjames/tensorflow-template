import argparse
import json
import os
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
    # TODO : put something here
    mode2idxs_dict = {}
    data = {}
    _save(mode2idxs_dict, data, target_dir)


def _save(mode2idxs_dict, data, target_dir):
    mode2idxs_path = os.path.join(target_dir, "mode2idxs.json")
    metadata_path = os.path.join(target_dir, "metadata.json")
    data_path = os.path.join(target_dir, "data.json")
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as fh:
            metadata = json.load(fh)
    else:
        metadata = OrderedDict()

    with open(mode2idxs_path, 'w') as fh:
        json.dump(fh, mode2idxs_dict)
    with open(metadata_path, 'w') as fh:
        json.dump(fh, metadata)
    with open(data_path, 'w') as fh:
        json.dump(fh, data)


def main():
    args = get_args()
    prepro(args)


if __name__ == "__main__":
    main()
