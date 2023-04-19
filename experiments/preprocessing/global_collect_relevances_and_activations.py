import logging
import os
from argparse import ArgumentParser

import numpy as np
import torch
import yaml
from crp.attribution import CondAttribution
from crp.concepts import ChannelConcept
from crp.helper import get_layer_names
from tqdm import tqdm
from zennit.composites import EpsilonPlusFlat

from datasets import get_dataset
from experiments.reveal_artifacts.run_spray import str2bool
from models import get_canonizer, get_fn_model_loader

logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)


def get_args():
    parser = ArgumentParser()
    parser.add_argument("--batch_size", default=None, type=int)
    parser.add_argument('--config_file',
                        default="config_files/correcting_isic/local/vgg16_Vanilla.yaml")
    parser.add_argument("--class_id", default=None, type=int)
    parser.add_argument("--results_dir", default="results")
    parser.add_argument('--all_layers', default=False)
    parser.add_argument("--split", default="all", choices=['train', 'val', 'all'])
    parser.add_argument('--use_corrected_ckpt', default=False, type=str2bool)

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

    batch_size = args.batch_size or config.get("batch_size", 32)

    config['config_file'] = args.config_file
    config['split'] = args.split
    config['batch_size'] = batch_size
    config['results_dir'] = args.results_dir
    config['use_corrected_ckpt'] = args.use_corrected_ckpt
    config['all_layers'] = args.all_layers
    config['class_id'] = args.class_id

    run_collect_relevances_and_activations(config)


def run_collect_relevances_and_activations(config):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    start_collection(
        config['config_file'],
        config['dataset_name'],
        config['data_paths'],
        config.get("img_size", 224),
        config.get("attacked_classes", []),
        config.get("p_artifact", 0),
        config.get("split", "all"),
        config['model_name'],
        config['ckpt_path'],
        config['batch_size'],
        config.get('results_dir', 'results'),
        config.get('use_corrected_ckpt', False),
        device,
        config.get('all_layers', False),
        config['layer_name'],
        config['class_id']
    )


def start_collection(
        config_file,
        dataset_name,
        data_paths,
        img_size,
        attacked_classes,
        p_artifact,
        split,
        model_name,
        ckpt_path,
        batch_size,
        results_dir,
        use_corrected_ckpt,
        device,
        all_layers,
        layer_name,
        class_id
):
    dataset = get_dataset(dataset_name)(data_paths=data_paths,
                                        normalize_data=True,
                                        image_size=img_size,
                                        attacked_classes=attacked_classes,
                                        p_artifact=p_artifact)

    if split != "all":

        if split == 'train':
            dataset_split = dataset.get_subset_by_idxs(dataset.idxs_train)
        elif split == 'val':
            dataset_split = dataset.get_subset_by_idxs(dataset.idxs_val)
        elif split == 'test':
            dataset_split = dataset.get_subset_by_idxs(dataset.idxs_test)

    else:
        dataset_split = dataset

    logger.info(f"Using split {split} ({len(dataset_split)} samples)")

    n_classes = len(dataset_split.class_names)

    config_name = os.path.basename(config_file)[:-5]

    if use_corrected_ckpt:
        ckpt_path = f"checkpoints/{config_name}/last.ckpt"

    model = get_fn_model_loader(model_name=model_name)(n_class=n_classes, ckpt_path=ckpt_path)
    model = model.to(device)
    model.eval()

    attribution = CondAttribution(model)
    canonizers = get_canonizer(model_name)
    composite = EpsilonPlusFlat(canonizers)

    cc = ChannelConcept()

    linear_layers = []
    if all_layers:
        layer_names = get_layer_names(model, [torch.nn.Conv2d, torch.nn.BatchNorm2d])
        conv_layers = get_layer_names(model, [torch.nn.Conv2d, torch.nn.BatchNorm2d])
    else:
        layer_names = [layer_name]
        conv_layers = [layer_name]

    samples = np.array(
        [i for i in range(len(dataset_split)) if ((class_id is None) or (dataset_split.get_target(i) == class_id))])
    logger.info(f"Found {len(samples)} samples of class {class_id}.")

    n_samples = len(samples)
    n_batches = int(np.ceil(n_samples / batch_size))

    crvs = dict(zip(layer_names, [[] for _ in layer_names]))
    relevances_all = dict(zip(layer_names, [[] for _ in layer_names]))
    cavs_max = dict(zip(layer_names, [[] for _ in layer_names]))
    cavs_mean = dict(zip(layer_names, [[] for _ in layer_names]))
    smpls = []
    output = []

    for i in tqdm(range(n_batches)):
        samples_batch = samples[i * batch_size:(i + 1) * batch_size]
        data = torch.stack([dataset[j][0] for j in samples_batch], dim=0).to(device).requires_grad_()
        out = model(data).detach().cpu()
        condition = [{"y": c_id} for c_id in out.argmax(1)]

        attr = attribution(data, condition, composite, record_layer=layer_names, init_rel=1)
        non_zero = ((attr.heatmap.sum((1, 2)).abs().detach().cpu() > 0) * (out.argmax(1) == class_id)).numpy()
        samples_nz = samples_batch[non_zero]
        output.append(out[non_zero])

        for l in layer_names:
            relevances_all[l] += cc.attribute(attr.relevances[l],
                                              abs_norm=True).detach().cpu()  # TODO: check if this is needed @Frederik

        if samples_nz.size:
            smpls += [s for s in samples_nz]
            rels = [cc.attribute(attr.relevances[layer][non_zero], abs_norm=True) for layer in layer_names]
            acts_max = ([attr.activations[layer][non_zero].flatten(start_dim=2).max(2)[0] for layer in conv_layers]
                        + [attr.activations[layer][non_zero] for layer in linear_layers])
            acts_mean = ([attr.activations[layer][non_zero].mean((2, 3)) for layer in conv_layers]
                         + [attr.activations[layer][non_zero] for layer in linear_layers])
            for l, r, amax, amean in zip(layer_names, rels, acts_max, acts_mean):
                crvs[l] += r.detach().cpu()
                cavs_max[l] += amax.detach().cpu()
                cavs_mean[l] += amean.detach().cpu()

    if use_corrected_ckpt:
        path = f"{results_dir}/global_relevances_and_activations/{config_name}"
    else:
        path = f"{results_dir}/global_relevances_and_activations/{dataset_name}/{model_name}"
    os.makedirs(path, exist_ok=True)

    str_class_id = 'all' if class_id is None else class_id
    torch.save({"samples": smpls,
                "output": output,
                "crvs": crvs,
                "relevances_all": relevances_all,
                "cavs_max": cavs_max,
                "cavs_mean": cavs_mean},
               f"{path}/class_{str_class_id}_{split}.pth")
    for layer in layer_names:
        torch.save({"samples": smpls,
                    "output": output,
                    "crvs": crvs[layer],
                    "relevances_all": relevances_all[layer],
                    "cavs_max": cavs_max[layer],
                    "cavs_mean": cavs_mean[layer]},
                   f"{path}/{layer}_class_{str_class_id}_{split}.pth")


if __name__ == "__main__":
    main()
