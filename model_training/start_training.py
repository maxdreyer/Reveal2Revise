import logging
import os
from argparse import ArgumentParser

import pandas as pd
import torch
import wandb
import yaml
from torch.utils.data import DataLoader

from datasets import get_dataset
from model_training.train_model import train_model
from model_training.training_utils import get_optimizer, get_loss
from models import get_fn_model_loader

torch.multiprocessing.set_sharing_strategy('file_system')


def get_parser():
    parser = ArgumentParser(
        description='Train models.', )
    parser.add_argument('--config_file')
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

    config_name = os.path.basename(config_file)[:-5]
    start_training(config, config_name)


def start_training(config, config_name):
    """ Starts training for given config file.

    Args:
        config (dict): Dictionary with config parameters for training.
        config_name (str): Name of given config
    """

    dataset_name = config['dataset_name']
    data_paths = config.get('data_paths', [])
    model_name = config['model_name']
    pretrained = config['pretrained']
    num_epochs = config['num_epochs']
    eval_every_n_epochs = config['eval_every_n_epochs']
    store_every_n_epochs = config['store_every_n_epochs']
    batch_size = config['batch_size']
    optimizer_name = config['optimizer']
    artifacts_file = config.get('artifacts_file', None)
    clean_samples_only = config.get('clean_samples_only', False)
    binary_target = config.get('binary_target', False)
    img_size = config.get('img_size', 224)
    loss_name = config['loss']
    lr = config['lr']
    model_savedir = config['model_savedir']

    # Attack Details
    attacked_classes = config.get('attacked_classes', [])
    p_artifact = config.get('p_artifact', .5)
    artifact_type = config.get("artifact_type", "ch_text")

    default_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = config.get('device', default_device)
    wandb_project_name = config.get('wandb_project_name', None)
    wandb_api_key = config.get('wandb_api_key', None)

    do_wandb_logging = wandb_project_name is not None

    # Initialize WandB
    if do_wandb_logging:
        assert wandb_api_key is not None, f"'wandb_api_key' required if 'wandb_project_name' is provided ({wandb_project_name})"
        os.environ["WANDB_API_KEY"] = wandb_api_key
        wandb.init(project=wandb_project_name, config=config)
        wandb.run.name = f"{config_name}-{wandb.run.name}"
        logger.info(f"Initialized wand. Logging to {wandb_project_name} / {wandb.run.name}...")

    # Load Data and Model
    dataset = get_dataset(dataset_name)(data_paths=data_paths,
                                        normalize_data=True,
                                        image_size=img_size,
                                        binary_target=binary_target, 
                                        attacked_classes=attacked_classes,
                                        p_artifact=p_artifact,
                                        artifact_type=artifact_type,
                                        artifact_ids_file=artifacts_file)

    fn_model_loader = get_fn_model_loader(model_name)

    num_classes = 2 if binary_target else len(dataset.class_names)

    model = fn_model_loader(
        ckpt_path=None,
        pretrained=pretrained,
        n_class=num_classes).to(device)

    # Define Optimizer and Loss function
    optimizer = get_optimizer(optimizer_name, model.parameters(), lr)
    criterion = get_loss(loss_name, weights=dataset.weights)


    dataset_train = dataset.get_subset_by_idxs(dataset.idxs_train)
    dataset_val = dataset.get_subset_by_idxs(dataset.idxs_val)

    logger.info(
        f"Splitting the data into train ({len(dataset_train)}) and val ({len(dataset_val)}), ignoring samples from test ({len(dataset.idxs_test)})")
    
    dataset_train.do_augmentation = True
    dataset_val.do_augmentation = False

    if clean_samples_only:
        logger.info(f"#Samples before filtering: {len(dataset_train)}")
        dataset_train = dataset_train.get_subset_by_idxs(dataset_train.clean_sample_ids)
        logger.info(f"#Samples after filtering: {len(dataset_train)}")

    logger.info(f"Number of samples: {len(dataset_train)} (train) / {len(dataset_val)} (val)")

    dl_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=8)
    dl_val_dict = {'val': DataLoader(dataset_val, batch_size=batch_size, shuffle=False, num_workers=8)}

    if len(attacked_classes) > 0:
        print("ADDING attacked / Clean Dataset")
        dataset_clean = get_dataset(dataset_name)(data_paths=data_paths,
                                                  normalize_data=True,
                                                  image_size=img_size,
                                                  binary_target=binary_target, 
                                                  attacked_classes=[],
                                                  artifact_type=artifact_type,
                                                  p_artifact=p_artifact)

        all_classes = dataset.class_names if "isic" in dataset_name else range(len(dataset.class_names))
        dataset_attacked = get_dataset(dataset_name)(data_paths=data_paths,
                                                     normalize_data=True,
                                                     image_size=img_size,
                                                     binary_target=binary_target, 
                                                     attacked_classes=all_classes,
                                                     artifact_type=artifact_type,
                                                     p_artifact=p_artifact)

        dataset_val_clean = dataset_clean.get_subset_by_idxs(dataset.idxs_val)
        dataset_val_attacked = dataset_attacked.get_subset_by_idxs(dataset.idxs_val)
        dl_val_dict['val_clean'] = DataLoader(dataset_val_clean, batch_size=batch_size, shuffle=False, num_workers=8)
        dl_val_dict['val_attacked'] = DataLoader(dataset_val_attacked, batch_size=batch_size, shuffle=False, num_workers=8)

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=[50, 80], gamma=0.1)

    # Start Training
    train_model(
        model,
        model_name,
        dl_train,
        dl_val_dict,
        criterion,
        optimizer,
        scheduler,
        num_epochs,
        eval_every_n_epochs,
        store_every_n_epochs,
        device,
        model_savedir,
        do_wandb_logging
    )

if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
    main()
