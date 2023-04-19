import os
import shutil
from argparse import ArgumentParser
from typing import Tuple, List

import numpy as np
import torch
import yaml
from PIL import Image
from crp.attribution import CondAttribution
from matplotlib import pyplot as plt
from sklearn.svm import LinearSVC
from tqdm import tqdm
from zennit.composites import EpsilonPlusFlat

from datasets import get_dataset
from models import get_canonizer, get_fn_model_loader


def get_args():
    parser = ArgumentParser()
    parser.add_argument("--batch_size", default=10, type=int)
    parser.add_argument("--split", default="all")
    parser.add_argument("--layer_name", default="last_conv")
    parser.add_argument("--mode", default="cavs_max", choices=['cavs_mean', 'cavs_max', 'crvs'])
    parser.add_argument("--artifact", default="band_aid", type=str,
                        choices=["band_aid", "ruler", "skin_marker", "big_l", "small_l"])
    parser.add_argument("--neurons", default=(), type=Tuple[int])
    parser.add_argument("--save_localization", default=True, type=bool)
    parser.add_argument("--save_examples", default=True, type=bool)
    parser.add_argument('--config_file',
                        default="config_files/fixing_isic/local/vgg16_Vanilla_sgd_lr0.0001_band_aid.yaml")
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

    config['batch_size'] = args.batch_size
    config['layer_name'] = args.layer_name
    config['artifact'] = args.artifact

    localize_artifacts(config,
                       split=args.split,
                       mode=args.mode,
                       neurons=args.neurons,
                       save_examples=args.save_examples,
                       save_localization=args.save_localization)
    

def localize_artifacts(config: dict, 
                       split: str, 
                       mode: str, 
                       neurons: List(int), 
                       save_examples: bool, 
                       save_localization: bool):
    """Spatially localize artifacts in input samples.

    Args:
        config (dict): experiment config
        split (str): data split to use
        mode (str): CAV mode
        neurons (List): List of neurons to consider (all if None)
        save_examples (bool): Store example images
        save_localization (bool): Store localization heatmaps
    """

    dataset_name = config['dataset_name']
    model_name = config['model_name']

    device = "cuda" if torch.cuda.is_available() else "cpu"
    artifacts_file = config.get('artifacts_file', None)

    kwargs_data = {
        "p_artifact": 1.0, 
        "attacked_classes": config['attacked_classes']
        } if config['artifact'] == 'artificial' else {}
    
    dataset = get_dataset(dataset_name)(data_paths=config['data_paths'],
                                        normalize_data=True,
                                        artifact_ids_file=artifacts_file,
                                        artifact=config['artifact'],
                                        **kwargs_data)
    
    assert config['artifact'] in dataset.sample_ids_by_artifact.keys(), f"Artifact {config['artifact']} unknown."
    n_classes = len(dataset.class_names)

    path = f"results/global_relevances_and_activations/{dataset_name}/{model_name}"

    vecs = []

    sample_ids = []
    for class_id in range(n_classes):
        data = torch.load(f"{path}/{config['layer_name']}_class_{class_id}_{split}.pth")
        if data['samples']:
            sample_ids += data['samples']
            vecs.append(torch.stack(data[mode], 0))

    vecs = torch.cat(vecs, 0).to(device)

    # choose only specific neurons
    if neurons:
        vecs = vecs[:, np.array(neurons)]

    sample_ids = np.array(sample_ids)
    artifact_ids = np.array(
        [id_ for id_ in dataset.sample_ids_by_artifact[config['artifact']] if np.argwhere(sample_ids == id_)])
    target_ids = np.array(
        [np.argwhere(sample_ids == id_)[0][0] for id_ in artifact_ids])
    print(f"Chose {len(target_ids)} target samples.")

    model = get_fn_model_loader(model_name=model_name)(n_class=len(dataset.class_names),
                                                       ckpt_path=config['ckpt_path'])
    model = model.to(device)
    model.eval()

    canonizers = get_canonizer(model_name)
    composite = EpsilonPlusFlat(canonizers)

    attribution = CondAttribution(model)

    img_to_plt = lambda x: dataset.reverse_normalization(x.detach().cpu()).permute((1, 2, 0)).int().numpy()
    hm_to_plt = lambda x: x.detach().cpu().numpy()

    linear = LinearSVC(random_state=0, 
                       fit_intercept=False, 
                       penalty="l1", 
                       loss="squared_hinge", 
                       dual=False)
    
    targets = np.array([1 * (j in target_ids) for j, x in enumerate(sample_ids)])
    num_targets = (targets == 1).sum()
    num_notargets = (targets == 0).sum()
    weights = (targets == 1) / num_targets + (targets == 0) * 1 / num_notargets
    weights = weights / weights.max()

    X = vecs.detach().cpu().clamp(min=0).numpy()
    print("Fitting linear model..")

    linear.fit(X, targets, sample_weight=weights)
    w = torch.Tensor(linear.coef_)[..., None, None].to(device)

    print("Training score:")
    print(linear.score(X, targets, sample_weight=weights))

    samples = [dataset[sample_ids[i]] for i in target_ids]
    data_sample = torch.stack([s[0] for s in samples]).to(device).requires_grad_()
    target = [s[1] for s in samples]

    conditions = [{"y": t.item()} for t in target]

    batch_size = 32
    num_batches = int(np.ceil(len(data_sample) / batch_size))

    heatmaps = []
    inp_imgs = []

    layer_name = config['layer_name']

    for b in tqdm(range(num_batches)):
        data = data_sample[batch_size * b: batch_size * (b + 1)]
        attr = attribution(data,
                           conditions[batch_size * b: batch_size * (b + 1)],
                           composite, record_layer=[layer_name])
        act = attr.activations[layer_name]

        inp_imgs.extend([img_to_plt(s.detach().cpu()) for s in data])

        attr = attribution(data, [{}], composite, start_layer=layer_name, init_rel=act.clamp(min=0) * w)
        heatmaps.extend([hm_to_plt(h.detach().cpu().clamp(min=0)) for h in attr.heatmap])

    if save_examples:
        num_imgs = min(len(inp_imgs), 72) * 2
        grid = int(np.ceil(np.sqrt(num_imgs) / 2) * 2)

        fig, axs_ = plt.subplots(grid, grid, dpi=150, figsize=(grid * 1.2, grid * 1.2))

        for j, axs in enumerate(axs_):
            ind = int(j * grid / 2)
            for i, ax in enumerate(axs[::2]):
                if len(inp_imgs) > ind + i:
                    ax.imshow(inp_imgs[ind + i])
                    ax.set_xlabel(f"sample {int(artifact_ids[ind + i])}", labelpad=1)
                ax.set_xticks([])
                ax.set_yticks([])

            for i, ax in enumerate(axs[1::2]):
                if len(inp_imgs) > ind + i:
                    max = np.abs(heatmaps[ind + i]).max()
                    ax.imshow(heatmaps[ind + i], cmap="bwr", vmin=-max, vmax=max)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_xlabel(f"artifact", labelpad=1)

        plt.tight_layout(h_pad=0.1, w_pad=0.0)
        # reduce spacing between plots
        # fig.subplots_adjust(hspace=0.55, wspace=0.01)
        os.makedirs(f"results/localization/{dataset_name}/{model_name}", exist_ok=True)
        plt.savefig(f"results/localization/{dataset_name}/{model_name}/{config['artifact']}_{layer_name}_{mode}.pdf")
        plt.show()

    if save_localization:
        path = f"data/localized_artifacts/{dataset_name}/{config['artifact']}"
        if os.path.isdir(path):
            shutil.rmtree(path)
        os.makedirs(path, exist_ok=True)
        for i in range(len(heatmaps)):
            sample_id = int(artifact_ids[i])
            heatmap = heatmaps[i]
            heatmap[heatmap < 0] = 0
            heatmap = heatmap / heatmap.max() * 255
            im = Image.fromarray(heatmap).convert("L")
            im.save(f"{path}/{sample_id}.png")


if __name__ == "__main__":
    main()
