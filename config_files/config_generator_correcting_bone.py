import copy
import os
import shutil

import yaml

config_dir = "correcting_bone"
shutil.rmtree(config_dir)
os.makedirs(f"{config_dir}/local", exist_ok=True)

base_config = {
    'num_epochs': 10,
    'device': 'cuda',
    'dataset_name': 'bone',
    'loss': 'cross_entropy',
    'wandb_api_key': 'your_api_key',
    'img_size': 224,
    'wandb_project_name': 'your_project_name',
    'artifacts_file': 'data/artifacts_bone.json'
}


def store_configs(config, config_name):
    model_name = config['model_name']
    config['ckpt_path'] = f"checkpoints/checkpoint_{model_name}_bone_last.pth"

    config['ckpt_path_corrected'] = f"checkpoints/{config_name}/last.ckpt"
    if "Vanilla" in config_name and config['num_epochs'] == 0:
        config['ckpt_path_corrected'] = config['ckpt_path']

    config['data_paths'] = ["/data/bone"]
    config['batch_size'] = 64

    with open(f"{config_dir}/local/{config_name}.yaml", 'w') as outfile:
        yaml.dump(config, outfile, default_flow_style=False)


layer_names_by_model = {
    'vgg16': [
        # 'features.0', 'features.2', 'features.5', 'features.7', 
        # 'features.10', 'features.12','features.14', 'features.17',
        'features.19',
        # 'features.21', 'features.24', 'features.26',
        'features.28'],
    'resnet18': [
        # "identity_0", "identity_1", "identity_2", 
        'last_conv'],
    'efficientnet_b0': [
        # "identity_0", "identity_1", "identity_2", "identity_3", 
        # "identity_4", "identity_6","identity_7",
        'last_conv']
}
for model_name, layer_name in [
    ('vgg16', 'features.28'),
    ('resnet18', 'last_conv'),
    ('efficientnet_b0', 'last_conv')
]:

    base_config['model_name'] = model_name
    base_config['layer_name'] = layer_name

    for artifact in [
        "big_l",
    ]:
        base_config['artifact'] = artifact
        for lr in [
            0.0001,
        ]:
            base_config['lr'] = lr
            optim_name = "sgd"

            base_config['optimizer'] = optim_name

            ### VANILLA
            config_vanilla = copy.deepcopy(base_config)
            method = 'Vanilla'
            config_vanilla['method'] = method
            config_vanilla['lamb'] = 0.0
            config_name = f"{model_name}_{method}_{optim_name}_lr{lr}_{artifact}"
            store_configs(config_vanilla, config_name)

            config_name = f"{model_name}_{method}"
            config_vanilla['num_epochs'] = 0
            store_configs(config_vanilla, config_name)

            ### CDEP
            if "efficientnet" not in model_name:
                method = 'CDEP'
                for lamb in [
                    50,
                ]:
                    base_config['method'] = method
                    base_config['lamb'] = lamb
                    # base_config['num_epochs'] = 5
                    config_name = f"{model_name}_{method}_{optim_name}_lr{lr}_lamb{lamb}_{artifact}"
                    store_configs(base_config, config_name)

            method = 'RRR_ExpMax'
            for lamb in [
                50,
            ]:
                base_config['method'] = method
                base_config['lamb'] = lamb
                # base_config['num_epochs'] = 5
                config_name = f"{model_name}_{method}_{optim_name}_lr{lr}_lamb{lamb}_{artifact}_{layer_name}"
                store_configs(base_config, config_name)

            # ClArC Implementations
            for layer_name in layer_names_by_model[model_name]:
                base_config['layer_name'] = layer_name

                method = 'AClarc'
                base_config['method'] = method
                for lamb in [
                    1
                ]:
                    base_config['lamb'] = lamb
                    config_name = f"{model_name}_{method}_{optim_name}_lr{lr}_lamb{lamb}_{artifact}_{layer_name}"
                    store_configs(base_config, config_name)

                method = 'PClarc'
                base_config['method'] = method
                for lamb in [
                    1
                ]:
                    base_config['lamb'] = lamb
                    config_name = f"{model_name}_{method}_{optim_name}_lr{lr}_lamb{lamb}_{artifact}_{layer_name}"
                    store_configs(base_config, config_name)
