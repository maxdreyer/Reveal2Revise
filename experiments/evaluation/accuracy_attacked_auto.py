import os
import random
from argparse import ArgumentParser
from typing import List

import numpy as np
import torch
import torchvision
import wandb
import yaml
from tqdm import tqdm

from datasets import get_dataset
from experiments.evaluation.compute_metrics import compute_metrics
from experiments.evaluation.prepare_for_evaluation import prepare_model_for_evaluation
from models import get_fn_model_loader

torch.random.manual_seed(0)
np.random.seed(0)
random.seed(0)


def get_args():
    parser = ArgumentParser()
    parser.add_argument("--ckpt_path", default=None)
    parser.add_argument("--artifact", default="band_aid")
    parser.add_argument('--config_file',
                        default="config_files/fixing/local/vgg16_RRR_ExpMax_sgd_lr0.0001_lamb500_skin_marker.yaml")

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

    if config.get('wandb_api_key', None):
        os.environ["WANDB_API_KEY"] = config['wandb_api_key']
        wandb.init(id=config['wandb_id'], project=config['wandb_project_name'], resume=True)

    config['ckpt_path'] = args.ckpt_path
    config['config_file'] = args.config_file
    config['attack_artifact'] = args.artifact

    compute_accuracy_attacked(config)

def compute_accuracy_attacked(config):
    """
    Computes accuracy on attacked datasets (train/val/test), where artifacts are cropped from randomly picked training samples (using 
    automated artifact localization) and inserted into samples to be evaluated.

    Args:
        config (dict): exeriment config
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset_name = config['dataset_name']
    model_name = config['model_name']

    dataset = get_dataset(f"{dataset_name}_hm")(data_paths=config['data_paths'],
                                                normalize_data=True,
                                                artifact_ids_file=config['artifacts_file'],
                                                artifact=config['attack_artifact'])
    
    n_classes = len(dataset.class_names)

    ckpt_path = config['ckpt_path'] if config['ckpt_path'] else f"checkpoints/{os.path.basename(config['config_file'])[:-5]}/last.ckpt"

    rng = np.random.default_rng(0)

    model = get_fn_model_loader(model_name=model_name)(n_class=n_classes, ckpt_path=ckpt_path)
    model = prepare_model_for_evaluation(model, dataset, ckpt_path, device, config)

    gaussian = torchvision.transforms.GaussianBlur(kernel_size=41, sigma=5.0)

    ### COLLECT ARTIFACTS
    artifact_samples = dataset.sample_ids_by_artifact[config['attack_artifact']]
    masks = []
    artifacts = []
    batch_size = config['batch_size']
    print(f"There are {len(artifact_samples)} artifact samples")
    for k, samples in enumerate([artifact_samples]):

        n_samples = len(samples)
        n_batches = int(np.ceil(n_samples / batch_size))

        for i in tqdm(range(n_batches)):
            samples_batch = samples[i * batch_size:(i + 1) * batch_size]
            data = torch.stack([dataset[j][0] for j in samples_batch], dim=0)
            mask = torch.stack([dataset[j][2] for j in samples_batch])
            mask = gaussian(mask.clamp(min=0)) ** 1.0
            mask = mask / mask.abs().flatten(start_dim=1).max(1)[0][:, None, None]
            artifacts.append(data)
            masks.append(mask)

    masks = torch.cat(masks, 0)
    artifacts = torch.cat(artifacts, 0)

    val_set = dataset.idxs_val
    test_set = dataset.idxs_test

    sets = {
        "val": val_set,
        "test": test_set
    }

    for split in ['val', 'test']:
        split_set = sets[split]
        labels_set = ["all", "clean"]
        sample_sets = [split_set,
                       [x for x in split_set if (x not in artifact_samples)]]

        print("size of sample sets", [len(x) for x in sample_sets])

        y_pred_all, y_target_all = [], []
        for k, samples in enumerate(sample_sets):

            y_pred = []
            y_target = []
            samples = np.array(samples)
            n_samples = len(samples)
            n_batches = int(np.ceil(n_samples / batch_size))

            for _ in range(3):
                for i in tqdm(range(n_batches)):
                    samples_batch = samples[i * batch_size:(i + 1) * batch_size]
                    data = torch.stack([dataset[j][0] for j in samples_batch], dim=0)
                    pick = rng.choice(range(len(masks)), len(samples_batch))
                    m = masks[pick][:, None, :, :]
                    artifact = artifacts[pick]
                    data = data * (1 - m) + artifact * m

                    out = model(data.to(device)).detach().cpu()

                    targets = torch.tensor([dataset[j][1] for j in samples_batch])

                    y_pred.append(out)
                    y_target.append(targets)

            y_pred = torch.cat(y_pred, 0)
            y_target = torch.cat(y_target, 0)

            metrics = compute_metrics(y_pred, y_target, None, prefix=f"{split}_auto-attacked_{config['artifact']}_",
                                      suffix=f"_{labels_set[k].lower()}")
            print(metrics)
            y_pred_all.append(y_pred)
            y_target_all.append(y_target)

            if config.get('wandb_api_key', None):
                wandb.log(metrics)


if __name__ == "__main__":
    main()
