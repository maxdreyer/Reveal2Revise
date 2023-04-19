import os
from argparse import ArgumentParser

import torch
import wandb
import yaml
from torch.utils.data import DataLoader

from datasets import get_dataset
from experiments.evaluation.compute_metrics import compute_metrics, compute_model_scores
from experiments.evaluation.prepare_for_evaluation import prepare_model_for_evaluation
from models import get_fn_model_loader

torch.random.manual_seed(0)


def get_args():
    parser = ArgumentParser()
    parser.add_argument("--ckpt_path", default=None)
    parser.add_argument('--config_file', default="config_files/training/local/vgg_test.yaml")

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

    evaluate_by_subset_attacked(config)

def evaluate_by_subset_attacked(config):
    """ Run evaluations for each data split (train/val/test) on 3 variants of datasets:
            1. Same as training (one attacked class)
            2. Attacked (artifact in all classes)
            3. Clean (no artifacts)

    Args:
        config (dict): config for model correction run
    """

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset_name = config['dataset_name']
    model_name = config['model_name']

    data_paths = config['data_paths']
    batch_size = config['batch_size']
    
    dataset = get_dataset(dataset_name)(data_paths=data_paths,
                                        normalize_data=True,
                                        p_artifact=config['p_artifact'],
                                        attacked_classes=config['attacked_classes'])

    n_classes = len(dataset.class_names)
    ckpt_path = config['ckpt_path'] if config['ckpt_path'] else f"checkpoints/{os.path.basename(config['config_file'])[:-5]}/last.ckpt"
    model = get_fn_model_loader(model_name=model_name)(n_class=n_classes, ckpt_path=ckpt_path)
    model = prepare_model_for_evaluation(model, dataset, ckpt_path, device, config)

    sets = {
        'train': dataset.idxs_train,
        'val': dataset.idxs_val,
        'test': dataset.idxs_test,
    }
    
    dataset_clean = get_dataset(dataset_name)(data_paths=data_paths,
                                                normalize_data=True,
                                                p_artifact=1.0,
                                                attacked_classes=[])
    
    all_classes = dataset.class_names if "isic" in dataset_name else range(len(dataset.class_names))
    
    dataset_attacked = get_dataset(dataset_name)(data_paths=data_paths,
                                                normalize_data=True,
                                                p_artifact=1.0,
                                                attacked_classes=all_classes)
    for split in ['test', 'val']:
        split_set = sets[split]

        dataset_ch_split = dataset.get_subset_by_idxs(split_set)
        dataset_clean_split = dataset_clean.get_subset_by_idxs(split_set)
        dataset_attacked_split = dataset_attacked.get_subset_by_idxs(split_set)

        dl = DataLoader(dataset_ch_split, batch_size=batch_size, shuffle=False)
        model_outs, y_true = compute_model_scores(model, dl, device)

        dl_attacked = DataLoader(dataset_attacked_split, batch_size=batch_size, shuffle=False)
        model_outs_attacked, y_true_attacked = compute_model_scores(model, dl_attacked, device)

        dl_clean = DataLoader(dataset_clean_split, batch_size=batch_size, shuffle=False)
        model_outs_clean, y_true_clean = compute_model_scores(model, dl_clean, device)

        metrics = compute_metrics(model_outs, y_true, dataset.class_names, prefix=f"{split}_", suffix=f"_ch")

        metrics_attacked = compute_metrics(model_outs_attacked, y_true_attacked, dataset.class_names,
                                           prefix=f"{split}_", suffix=f"_attacked")
        metrics_clean = compute_metrics(model_outs_clean, y_true_clean, dataset.class_names, prefix=f"{split}_",
                                        suffix=f"_clean")

        if config.get('wandb_api_key', None):
            wandb.log({**metrics, **metrics_attacked, **metrics_clean})


if __name__ == "__main__":
    main()
