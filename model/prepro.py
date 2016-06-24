import argparse
import os


def get_args():
    parser = argparse.ArgumentParser()
    home = os.path.expanduser("~")
    source_dir = os.path.join(home, "data", "model")
    target_dir = "data/model"
    parser.add_argument("--source_dir", default=source_dir)
    parser.add_argument("--target_dir", default=target_dir)
    return parser.parse_args()


def prepro(args):
    # TODO : do something
    pass


def main():
    args = get_args()
    prepro(args)


if __name__ == "__main__":
    main()
