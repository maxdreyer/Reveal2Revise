import os
from argparse import ArgumentParser

import numpy as np
import torch
import yaml
from scipy import ndimage
from tqdm import tqdm

from datasets import get_dataset
from models import get_fn_model_loader
from utils import cd


def get_args():
    parser = ArgumentParser()
    parser.add_argument('--config_file', default="config_files/fixing/local/vgg16_GClarc_sgd_lr0.0001_lamb1.yaml")

    args = parser.parse_args()

    return args


FEATURE_DIM = {
    'vgg16': 25088,
    'resnet18': 512
}

def compute_cd(blob, img_torch, model, model_name):
    if model_name == "vgg16":
        return cd.cd_vgg_features(blob, img_torch, model)
    elif model_name == "resnet18":
        return cd.cd_resnet_features(blob, img_torch, model)


def main():
    args = get_args()

    with open(args.config_file, "r") as stream:
        try:
            config = yaml.safe_load(stream)
            config["wandb_id"] = os.path.basename(args.config_file)[:-5]
        except yaml.YAMLError as exc:
            print(exc)
            config = {}

    run_cd_preprocessing(config)
    

def run_cd_preprocessing(config):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    cd_preprocessing(
        dataset_name=config['dataset_name'] + "_hm",
        data_paths=config['data_paths'],
        artifacts_file=config.get('artifacts_file', None),
        img_size=config['img_size'],
        segmentation_dir=config.get('segmentation_dir', None),
        artifact=config['artifact'],
        attacked_classes=config.get('attacked_classes', None),
        p_artifact=config.get('p_artifact', None),
        model_name=config['model_name'],
        ckpt_path=config['ckpt_path'],
        device=device
    )

def cd_preprocessing(
        dataset_name,
        data_paths,
        artifacts_file,
        img_size,
        segmentation_dir,
        artifact,
        attacked_classes,
        p_artifact,
        model_name,
        ckpt_path,
        device
):
    if "band" in artifact and segmentation_dir is not None:
        print("Using True segmentation masks for CD Features")
        dataset_name = "isic_seg_bandaid"
        dataset = get_dataset(dataset_name)(data_paths=data_paths, 
                                            normalize_data=True, 
                                            segmentation_dir=segmentation_dir, 
                                            image_size=img_size)
    else:
        dataset = get_dataset(dataset_name)(data_paths=data_paths, 
                                            normalize_data=True, 
                                            artifact=artifact, 
                                            image_size=img_size,
                                            artifact_ids_file=artifacts_file,
                                            attacked_classes=attacked_classes,
                                            p_artifact=p_artifact)

    n_classes = len(dataset.class_names)
    model = get_fn_model_loader(model_name=model_name)(n_class=n_classes, ckpt_path=ckpt_path)
    model = model.to(device)
    model.eval()

    cd_features = -np.ones((len(dataset), 2, FEATURE_DIM[model_name]))  # rel, irrel

    my_square = np.ones((20, 20), dtype=np.uint8)
    with torch.no_grad():
        for i in tqdm(dataset.sample_ids_by_artifact[artifact]):
            mask = dataset[i][2]
            if mask.sum() > 0:
                img_torch = dataset[i][0][None].to(device)

                seg = mask / mask.abs().max() > 0.1
                blob = dilation((np.asarray(seg)).astype(np.uint8), my_square).astype(np.float32)

                rel, irrel = compute_cd(blob, img_torch, model, model_name)

                cd_features[i, 0] = rel[0].cpu().numpy()
                cd_features[i, 1] = irrel[0].cpu().numpy()

    path = f"results/cd_preprocessing/{dataset_name}/{model_name}"

    os.makedirs(path, exist_ok=True)
    np.save(f"{path}/cd_features.npy", cd_features)


def dilation(image, footprint=None, out=None, shift_x=False, shift_y=False):
    footprint = np.array(footprint)
    footprint = _shift_footprint(footprint, shift_x, shift_y)

    footprint = _invert_footprint(footprint)
    if out is None:
        out = np.empty_like(image)
    ndimage.grey_dilation(image, footprint=footprint, output=out)
    return out


def _shift_footprint(footprint, shift_x, shift_y):
    if footprint.ndim != 2:
        # do nothing for 1D or 3D or higher footprints
        return footprint
    m, n = footprint.shape
    if m % 2 == 0:
        extra_row = np.zeros((1, n), footprint.dtype)
        if shift_x:
            footprint = np.vstack((footprint, extra_row))
        else:
            footprint = np.vstack((extra_row, footprint))
        m += 1
    if n % 2 == 0:
        extra_col = np.zeros((m, 1), footprint.dtype)
        if shift_y:
            footprint = np.hstack((footprint, extra_col))
        else:
            footprint = np.hstack((extra_col, footprint))
    return footprint


def _invert_footprint(footprint):
    inverted = footprint[(slice(None, None, -1),) * footprint.ndim]
    return inverted


if __name__ == "__main__":
    main()
