import os
import random
from argparse import ArgumentParser

import numpy as np
import torch
import wandb
import yaml
from crp.attribution import CondAttribution
from matplotlib import pyplot as plt
from zennit.composites import EpsilonPlusFlat

from datasets import get_dataset
from experiments.evaluation.prepare_for_evaluation import prepare_model_for_evaluation
from models import get_fn_model_loader, get_canonizer

torch.random.manual_seed(0)
np.random.seed(0)
random.seed(0)


def get_args():
    parser = ArgumentParser()
    parser.add_argument("--sample_ids", default="531,562,560", type=str)
    parser.add_argument('--config_file',
                        default="config_files/fixing_isic/local/vgg16_AClarc_sgd_lr0.0001_lamb1.0_features.14_band_aid.yaml")
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

    config['sample_ids'] = [int(i) for i in args.sample_ids.split(",")]

    plot_corrected_model(config)


def plot_corrected_model(config):
    dataset_name = config['dataset_name']

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    kwargs_data = {
        "p_artifact": 1.0,
        "attacked_classes": config['attacked_classes']
    } if config['artifact'] == 'artificial' else {
        "artifact_ids_file": config['artifacts_file']
    }

    dataset = get_dataset(dataset_name)(data_paths=config['data_paths'],
                                        normalize_data=True,
                                        **kwargs_data)

    if "attacked" in dataset_name:
        sample_ids = dataset.artifact_ids[sample_ids]
    data = torch.stack([dataset[j][0] for j in sample_ids], dim=0).to(device)
    target = torch.stack([dataset[j][1] for j in sample_ids], dim=0)
    ckpt_path_corrected = f"checkpoints/{os.path.basename(config['config_file'])[:-5]}/last.ckpt"
    ckpt_path_corrupted = config['ckpt_path']
    model_corrected = get_fn_model_loader(model_name=config['model_name'])(n_class=len(dataset.class_names),
                                                                           ckpt_path=ckpt_path_corrected)
    model_corrected = prepare_model_for_evaluation(model_corrected, dataset, ckpt_path_corrected, device, config)

    model_corrupted = get_fn_model_loader(model_name=config['model_name'])(n_class=len(dataset.class_names),
                                                                           ckpt_path=ckpt_path_corrupted)

    model_corrupted.eval()
    model_corrupted = model_corrupted.to(device)

    attribution_corrected = CondAttribution(model_corrected)
    attribution_corrupted = CondAttribution(model_corrupted)
    canonizers = get_canonizer(config['model_name'])
    composite = EpsilonPlusFlat(canonizers)

    condition = [{"y": c_id.item()} for c_id in target]
    attr_corrected = attribution_corrected(data.requires_grad_(), condition, composite, init_rel=1)
    heatmaps_corrected = attr_corrected.heatmap / attr_corrected.heatmap.flatten(start_dim=1).max(1,
                                                                                                  keepdim=True).values[
                                                  :, None]
    heatmaps_corrected = heatmaps_corrected.detach().cpu().numpy()

    # computed corrupted heatmaps
    condition_corrupted = [{"y": c_id.item()} for c_id in target]
    attr_corrupted = attribution_corrupted(data.requires_grad_(), condition_corrupted, composite, init_rel=1)
    heatmaps_corrupted = attr_corrupted.heatmap / attr_corrupted.heatmap.flatten(start_dim=1).max(1,
                                                                                                  keepdim=True).values[
                                                  :, None]
    heatmaps_corrupted = heatmaps_corrupted.detach().cpu().numpy()

    # plot input images and heatmaps in grid
    fig, axs = plt.subplots(3, len(sample_ids), figsize=(len(sample_ids) * 3, 3 * 3), dpi=300)

    for i, sample_id in enumerate(sample_ids):
        axs[0, i].imshow(dataset.reverse_normalization(dataset[sample_id][0]).permute(1, 2, 0) / 255)

        axs[0, i].set_xticks([])
        axs[0, i].set_yticks([])
        axs[0, i].set_title(f"Sample {sample_id}")

        axs[1, i].imshow(heatmaps_corrupted[i], vmin=-1, vmax=1, cmap="bwr")
        axs[1, i].set_xticks([])
        axs[1, i].set_yticks([])

        axs[2, i].imshow(heatmaps_corrected[i], vmin=-1, vmax=1, cmap="bwr")
        axs[2, i].set_xticks([])
        axs[2, i].set_yticks([])

        # make border thicker
        for ax in axs[:, i]:
            for spine in ax.spines.values():
                spine.set_linewidth(2)

        # set label for the first column
        if i == 0:
            axs[0, i].set_ylabel("Input")
            axs[1, i].set_ylabel("Vanilla")
            axs[2, i].set_ylabel(str(config['method']))

    plt.tight_layout()

    # save figure with and without labels as pdf
    path = f"results/plot_corrected_model"
    if not os.path.exists(path):
        os.makedirs(path)

    plt.savefig(f"{path}/{config['wandb_id']}.pdf", bbox_inches="tight", dpi=300)

    # disable labels
    for ax in axs.flatten():
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_title("")
    plt.savefig(f"{path}/{config['wandb_id']}_no_labels.pdf", bbox_inches="tight", pad_inches=0, dpi=300)
    plt.savefig(f"{path}/{config['wandb_id']}_no_labels.png", bbox_inches="tight", pad_inches=0, dpi=300)
    plt.show()

    # log png to wandb
    wandb.log({"corrected_model": wandb.Image(f"{path}/{config['wandb_id']}_no_labels.png")})

    print("Done.")


if __name__ == "__main__":
    main()
