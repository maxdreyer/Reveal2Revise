import os
import random
from argparse import ArgumentParser
from typing import List

import numpy as np
import torch
import torchvision
import yaml
from matplotlib import pyplot as plt

from datasets import get_dataset, get_bone_attacked, get_isic_artifact

torch.random.manual_seed(0)
np.random.seed(0)
random.seed(0)


def get_args(fixed_arguments: List[str] = []):
    parser = ArgumentParser()
    parser.add_argument("--artifact", default="band_aid")
    parser.add_argument("--sample_id", default=531, type=int)
    parser.add_argument("--attacked_sample_id", default=1222)
    parser.add_argument('--config_file',
                        default="config_files/fixing/local/vgg16_AClarc2_sgd_lr0.0001_lamb0.2_band_aid_local.yaml")

    args = parser.parse_args()

    with open(parser.parse_args().config_file, "r") as stream:
        try:
            config = yaml.safe_load(stream)
            config["wandb_id"] = os.path.basename(args.config_file)[:-5]
        except yaml.YAMLError as exc:
            print(exc)
            config = {}

    for k, v in config.items():
        if k not in fixed_arguments:
            setattr(args, k, v)

    return args


def main():
    args = get_args(fixed_arguments=[
        "artifact"
    ])
    dataset_name = args.dataset_name
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    attacked_classes = [0] if "bone" in dataset_name else ["MEL"]
    dirs_by_version = {'2019': "/media/pahde/Data/ISIC2019"} if "isic_combined" in dataset_name else {}
    p_artifact = 1
    dataset = get_dataset(dataset_name + "_hm")(dirs_by_version=dirs_by_version, path=args.data_path, preprocessing=True, split="test",
                                            attacked_classes=attacked_classes, fix_artifact=True,
                                            p_artifact=p_artifact)

    gaussian = torchvision.transforms.GaussianBlur(kernel_size=41, sigma=8.0)

    for sample_id in [1222]:
        data = torch.stack([dataset[j][0] for j in [sample_id]], dim=0)
        # data = torch.stack([dataset[j][0] for j in [args.sample_id]], dim=0)
        mask = torch.stack([dataset[j][2] for j in [sample_id]])
        # mask = torch.stack([dataset[j][2] for j in [args.sample_id]])

        fig, ax = plt.subplots(2, 5, figsize=(10, 4), dpi=300)

        ax[0][0].imshow(mask[0].detach().cpu(), vmin=-mask[0].max(), vmax=mask[0].max(), cmap="bwr")

        mask = 1.0 * (mask / mask.abs().flatten(start_dim=1).max(1)[0][:, None, None] > 0.1)
        mask = gaussian(mask.clamp(min=0)) ** 1.0
        mask = 1.0 * (mask / mask.abs().flatten(start_dim=1).max(1)[0][:, None, None] > 0.4)
        ax[0][1].imshow(1 / 255 * dataset.reverse_normalization(data[0]).detach().cpu().permute((1, 2, 0)))

        ax[0][2].imshow(mask[0].detach().cpu())

        img = 1 / 255 * dataset.reverse_normalization(data[0] * mask[0][None].to(data)).detach().cpu().permute(
            (1, 2, 0)) + (1 - mask[0][..., None]) * 1
        ax[0][3].imshow(img)

        data_ = torch.stack([dataset[j][0] for j in [args.attacked_sample_id]], dim=0)
        img = dataset.reverse_normalization(data[0]).detach().cpu().permute((1, 2, 0)) * mask[0][..., None] \
            + dataset.reverse_normalization(data_[0]).detach().cpu().permute((1, 2, 0)) * (1 - mask[0][..., None])
        ax[1][0].imshow(img / 255)

        img = dataset.reverse_normalization(data_[0]).detach().cpu().permute((1, 2, 0))
        ax[1][1].imshow(img / 255)

        if dataset_name == "":
            get_ds = get_bone_attacked
            path = args.data_path
            attacked_classes = [0, 1, 2, 3, 4]
            split = "train"
        elif "isic" in dataset_name:
            get_ds = get_isic_artifact
            path = {'2019': args.data_path, }
            attacked_classes = [1, 2, 3, 4, 5, 6, 7, 8, 9]
            split = "val"

        dataset_attacked = get_ds(path, split=split, preprocessing=False, attacked_classes=attacked_classes,
                        p_artifact=1.0, fix_artifact=True)
        data = torch.stack([dataset_attacked[j][0] for j in [sample_id]], dim=0)
        ax[1][2].imshow((data[0]).detach().cpu().permute((1, 2, 0)))

        dataset_attacked = get_ds(path, split=split, preprocessing=False, attacked_classes=attacked_classes,
                        p_artifact=0.0, fix_artifact=True)


    
        data__ = torch.stack([dataset_attacked[j][0] for j in [sample_id]], dim=0)
        # data__ = torch.stack([dataset[j][0] for j in [args.sample_id]], dim=0)
        ax[1][3].imshow((data__[0]).detach().cpu().permute((1, 2, 0)))

        mask = 1.0 * ((data - data__).sum(1) != 0)
        mask = gaussian(mask.clamp(min=0)) ** 1.0
        mask = 1.0 * (mask / mask.abs().flatten(start_dim=1).max(1)[0][:, None, None] > 0.3)
        img = (data[0] * mask.to(data).detach().cpu()).permute(
            (1, 2, 0)) + (1 - mask[0][..., None]) * 1
        ax[1][4].imshow(img)

        # remove ticks from all axes
        for i in range(2):
            for j in range(5):
                ax[i][j].set_xticks([])
                ax[i][j].set_yticks([])

        plt.tight_layout()

        # save figure with and without labels as pdf
        path = f"results/plot_masked_artifact"
        if not os.path.exists(path):
            os.makedirs(path)

        plt.savefig(f"{path}/{args.artifact}_{sample_id}.pdf", bbox_inches="tight", pad_inches=0, dpi=300)
        plt.savefig(f"{path}/{args.artifact}_{sample_id}.png", bbox_inches="tight", pad_inches=0, dpi=300)

        plt.show()


if __name__ == "__main__":
    main()
