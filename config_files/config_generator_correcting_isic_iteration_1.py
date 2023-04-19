import os

import yaml

config_dir = "iteratively_correcting_isic"
os.makedirs(f"{config_dir}/local", exist_ok=True)

base_config = {
    'num_epochs': 10,
    'device': 'cuda',
    'eval_every_n_epochs': 5,
    'store_every_n_epochs': 50,
    'dataset_name': 'isic',
    'loss': 'cross_entropy',
    'wandb_api_key': 'your_api_key',
    'wandb_project_name': 'your_project_name',
    'ckpt_path': 'checkpoints/checkpoint_vgg16_last.pth',
    'layer_name': 'features.28'
}


def store_configs(config, config_name):
    config['ckpt_path_corrected'] = f"checkpoints/{config_name}/last.ckpt"
    if "Vanilla" in config_name and config['num_epochs'] == 0:
        config['ckpt_path_corrected'] = config['ckpt_path']
    config['data_path'] = "/data/dataset_isic2019"
    config['batch_size'] = 64

    with open(f"{config_dir}/local/{config_name}.yaml", 'w') as outfile:
        yaml.dump(config, outfile, default_flow_style=False)


for model_name in [
    'vgg16',
]:
    base_config['model_name'] = model_name

    for artifact in [
        # "band_aid",
        "skin_marker",
        # "ruler"
    ]:
        base_config['artifact'] = artifact
        for lr in [
            0.0001,
        ]:
            base_config['lr'] = lr
            for optim_name in [
                'sgd',
                # 'adam'
            ]:
                base_config['optimizer'] = optim_name

                method = 'RRR_ExpMax'
                for lamb in [
                    500
                ]:
                    base_config['method'] = method
                    base_config['lamb'] = lamb
                    # base_config['num_epochs'] = 5
                    config_name = f"{model_name}_{method}_{optim_name}_lr{lr}_lamb{lamb}_{artifact}"

                    store_configs(base_config, config_name)
