import os
from argparse import ArgumentParser

import yaml

from experiments.preprocessing.global_collect_relevances_and_activations import run_collect_relevances_and_activations
from experiments.preprocessing.cd_preprocessing import run_cd_preprocessing

def get_args():
    parser = ArgumentParser()
    parser.add_argument('--config_file', default=None)
    args = parser.parse_args()
    return args


def main():
    args = get_args()

    with open(args.config_file, "r") as stream:
        try:
            config = yaml.safe_load(stream)
            config["wandb_id"] = os.path.basename(args.config_file)[:-5]
        except yaml.YAMLError as exc:
            print(exc)
            config = {}

    config['config_file'] = args.config_file
    run_preprocessing(config)


def run_preprocessing(config):
    method = config['method']

    collect_relevances(config)

    if "CD" in method:
        run_cd_preprocessing(config)

    else:
        print("No further preprocessing..")


def collect_relevances(config):
    num_classes = 9 if "isic" in config['dataset_name'] else 5
    for class_id in range(num_classes):
        run_collect_relevances_and_activations({**config,
                                                'class_id': class_id})


if __name__ == "__main__":
    main()
