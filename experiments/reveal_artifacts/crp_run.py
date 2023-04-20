import os
from argparse import ArgumentParser

import torch.nn as nn
import yaml
from crp.attribution import CondAttribution
from crp.concepts import ChannelConcept
from crp.helper import get_layer_names
from crp.image import *
from crp.visualization import FeatureVisualization
from zennit.composites import EpsilonPlusFlat

from datasets import get_dataset
from models import get_fn_model_loader, get_canonizer


def get_parser(fixed_arguments: List[str] = []):
    parser = ArgumentParser(
        description='Run CRP preprocessing.', )

    parser.add_argument('--batch_size', default=64)
    parser.add_argument('--config_file',
                        default="config_files/correcting_isic/local/vgg16_Vanilla.yaml")
    args = parser.parse_args()

    with open(parser.parse_args().config_file, "r") as stream:
        try:
            config = yaml.safe_load(stream)
            config["config_name"] = os.path.basename(args.config_file)[:-5]
        except yaml.YAMLError as exc:
            print(exc)
            config = {}

    for k, v in config.items():
        if k not in fixed_arguments:
            setattr(args, k, v)

    return args


def main():
    args = get_parser()

    model_name = args.model_name
    dataset_name = args.dataset_name

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    dataset = get_dataset(dataset_name)(data_paths=args.data_paths, normalize_data=False, split="train")

    model = get_fn_model_loader(model_name)(n_class=len(dataset.class_names),
                                            ckpt_path=args.ckpt_path_corrected).to(device)
    model.eval()

    canonizers = get_canonizer(model_name)
    composite = EpsilonPlusFlat(canonizers)
    cc = ChannelConcept()
    layer_names = get_layer_names(model, [nn.Conv2d])
    layer_map = {layer: cc for layer in layer_names}

    attribution = CondAttribution(model)

    fv = FeatureVisualization(attribution, dataset, layer_map, preprocess_fn=dataset.normalize_fn,
                              path=f"crp_files/{args.config_name}", cache=None)

    fv.run(composite, 0, len(dataset), batch_size=args.batch_size)


if __name__ == "__main__":
    main()
