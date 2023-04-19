import os

import yaml

config_dir = "training_isic"

os.makedirs(f"{config_dir}/local", exist_ok=True)

base_config = {
    'num_epochs': 150,
    'device': 'cuda',
    'eval_every_n_epochs': 5,
    'store_every_n_epochs': 150,
    'dataset_name': 'isic_attacked',
    'loss': 'cross_entropy',
    'wandb_api_key': 'your_api_key',
    'wandb_project_name': 'your_project_name',
    'img_size': 224,
    'attacked_classes': ['MEL'],
    'p_artifact': 0.8
}


def store_config(config, config_name):
    config['batch_size'] = 64 if config['model_name'] == 'efficientnet_b4' else 128
    config['model_savedir'] = "/data/output"
    config['data_paths'] = ["/data/dataset_isic2019"]

    with open(f"{config_dir}/local/{config_name}.yaml", 'w') as outfile:
        yaml.dump(config, outfile, default_flow_style=False)


for model_name in [
    'vgg16',
    'efficientnet_b0',
    'resnet18',
]:
    for artifact_type in [
        "ch_text"
    ]:
        base_config['model_name'] = model_name
        base_config['artifact_type'] = artifact_type
        lrs = [0.001] if "efficientnet" in model_name else [0.005]
        for lr in lrs:
            base_config['lr'] = lr
            for pretrained in [
                True,
            ]:

                base_config['pretrained'] = pretrained
                optims = ["adam"] if "efficientnet" in model_name else ["sgd"]
                for optim_name in optims:
                    base_config['optimizer'] = optim_name
                    config_name = f"attacked-{artifact_type}_{model_name}_{optim_name}_lr{lr}_pretrained-{pretrained}"
                    store_config(base_config, config_name)
