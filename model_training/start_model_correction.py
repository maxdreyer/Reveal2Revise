import glob
import logging
import os
import random
import shutil
from argparse import ArgumentParser

import numpy as np
import torch
import yaml
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader

from datasets import get_dataset
from experiments.preprocessing.global_collect_relevances_and_activations import run_collect_relevances_and_activations
from model_training.correction_methods import get_correction_method
from models import get_fn_model_loader

torch.random.manual_seed(0)
np.random.seed(0)
random.seed(0)


def get_parser():
    parser = ArgumentParser(
        description='Train ISIC models.', )
    parser.add_argument('--config_file',
                        default="config_files/fixing_isic/local/vgg16_Vanilla_sgd_lr0.0001_band_aid.yaml")
    parser.add_argument('--num_gpu', default=1)
    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()
    config_file = args.config_file

    with open(config_file, "r") as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            logging.info(exc)
    config["config_file"] = config_file
    config_name = os.path.basename(config_file)[:-5]
    start_model_correction(config, config_name, args.num_gpu)


def start_model_correction(config, config_name, num_gpu):
    """ Starts model correction for given config file.

    Args:
        config (dict): Dictionary with config parameters for training.
        config_name (str): Name of given config
        num_gpu (int): Number of GPUs
    """

    dataset_name = config['dataset_name']
    data_paths = config['data_paths']

    model_name = config['model_name']
    ckpt_path = config['ckpt_path']
    num_epochs = config['num_epochs']
    batch_size = config['batch_size']
    optimizer_name = config['optimizer']
    loss_name = config['loss']
    lr = config['lr']

    # Real artifacts
    artifacts_file = config.get('artifacts_file', None)

    # Attack Details
    attacked_classes = config.get('attacked_classes', [])
    p_artifact = config.get('p_artifact', .5)
    artifact_type = config.get('artifact_type', "ch_text")
    img_size = config.get('img_size', 224)

    method = config["method"]

    # Clarc Parameters
    layer_name = config.get('layer_name', 'features.19')
    class_name = config.get('class_name', None)

    default_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = config.get('device', default_device)
    wandb_project_name = config.get('wandb_project_name', None)
    wandb_api_key = config.get('wandb_api_key', None)

    do_wandb_logging = wandb_project_name is not None

    # Initialize WandB
    if do_wandb_logging:
        assert wandb_api_key is not None, f"'wandb_api_key' required if 'wandb_project_name' is provided ({wandb_project_name})"
        os.environ["WANDB_API_KEY"] = wandb_api_key
        logger.info(f"Initialized wand. Logging to {wandb_project_name} / {config_name}...")
        logger_ = WandbLogger(project=wandb_project_name, name=f"{config_name}", id=f"{config_name}", config=config)
    
    # Load Data
    kwargs_data = {
        "p_artifact": p_artifact, 
        "attacked_classes": attacked_classes,
        "artifact_type": artifact_type
        } if config['artifact'] == 'artificial' else {}
    
    
    dataset = get_dataset(dataset_name)(data_paths=data_paths,
                                        normalize_data=True,
                                        image_size=img_size,
                                        artifact_ids_file=artifacts_file,
                                        **kwargs_data)

    # Load Model
    fn_model_loader = get_fn_model_loader(model_name)

    model = fn_model_loader(
        ckpt_path=ckpt_path,
        n_class=len(dataset.class_names)).to(device)

    # Construct correction kwargs
    kwargs_correction = {}
    if "clarcfullfeature" in method.lower():
        cav_dir = f"{os.path.dirname(ckpt_path)}/cavs_{os.path.basename(ckpt_path[:-4])}"
        os.makedirs(cav_dir, exist_ok=True)
        kwargs_correction['cav_dir'] = cav_dir
        kwargs_correction['dataset'] = dataset
        kwargs_correction['class_name'] = class_name
        kwargs_correction['artifact_name'] = config['artifact']
    elif "clarc" in method.lower():
        mode = "crvs" if "gclarc" in method.lower() else "cavs_max"
        kwargs_correction['n_classes'] = len(dataset.class_names)
        kwargs_correction['artifact_sample_ids'] = dataset.sample_ids_by_artifact[config['artifact']]
        kwargs_correction['mode'] = mode

    # Define Optimizer and Loss function
    correction_method = get_correction_method(method)
    model_corrected = correction_method(model, config, **kwargs_correction)

    # Define Optimizer and Loss function
    model_corrected.set_optimizer(optimizer_name, model_corrected.parameters(), lr)
    model_corrected.set_loss(loss_name, weights=dataset.weights)

    # Load Extended dataset if required (RRR/CDEP)
    load_extended_data = True
    if "RRR" in method:
        if ("band" in config['artifact'] and config.get("segmentation_dir", None) is not None):
            kwargs_data['segmentation_dir'] = config['segmentation_dir']
            dataset_name = "isic_seg_bandaid"
            logger.info("Using true segmentation masks for RRR.")
        else:
            kwargs_data['artifact'] = config['artifact']
            dataset_name = f"{dataset_name}_hm"
            
    elif "CDEP" in method:
        seg_mask_source = "isic_seg_bandaid" if ("band" in config['artifact'] \
                                                 and config.get("segmentation_dir", None) is not None) \
                                                    else dataset_name + "_hm"
        dataset_name = dataset_name + "_cd"
        kwargs_data['seg_mask_source'] = seg_mask_source
        kwargs_data['model_name'] = model_name
    else:
        load_extended_data = False
        
    if load_extended_data:
        dataset = get_dataset(dataset_name)(data_paths=data_paths, 
                                            normalize_data=True,
                                            image_size=img_size,
                                            artifact_ids_file=artifacts_file,
                                            **kwargs_data)
        
    # Split data into train/val
    idxs_train = dataset.idxs_train
    idxs_val = dataset.idxs_val

    dataset_train = dataset.get_subset_by_idxs(idxs_train)
    dataset_val = dataset.get_subset_by_idxs(idxs_val)
    dataset_train.do_augmentation = False  # Important!

    logger.info(f"Number of samples: {len(dataset_train)} (train) / {len(dataset_val)} (val)")

    dl_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=8)
    dl_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False, num_workers=8)

    checkpoint_callback = ModelCheckpoint(monitor="valid_acc",
                                          dirpath=f"checkpoints/{config_name}",
                                          filename="checkpt-{epoch:02d}-{valid_acc:.2f}",
                                          auto_insert_metric_name=False,
                                          save_last=True,
                                          save_weights_only=True,
                                          mode="max")

    trainer = Trainer(callbacks=[checkpoint_callback],
                      devices=num_gpu,
                      max_epochs=num_epochs,
                      accelerator="gpu",
                      logger=logger_)

    trainer.fit(model_corrected, dl_train, dl_val)

    # Store checkpoint when no finetuning is done
    if config['num_epochs'] == 0:
        os.makedirs(f"checkpoints/{config_name}", exist_ok=True)
        shutil.copy(ckpt_path, f"checkpoints/{config_name}/last.ckpt")

    # Store CAVs
    if "ClarcFullFeature" in method:
        os.makedirs(f"checkpoints/{config_name}/cavs_last", exist_ok=True)
        for file in glob.glob(f"{cav_dir}/*.npy"):
            shutil.copy(file, f"checkpoints/{config_name}/cavs_last/")

    # Re-compute relevances/activations for pclarced model
    if "PClarc" == method:
        for class_id in range(len(dataset.class_names)):
            run_collect_relevances_and_activations({**config,
                                                    'class_id': class_id,
                                                    'use_corrected_ckpt': True})
            

if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
    main()
