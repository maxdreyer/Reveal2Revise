import os
import torch
from torch.utils.data import Dataset
from model_training.correction_methods import get_correction_method


def prepare_model_for_evaluation(
        model: torch.nn.Module, 
        dataset: Dataset, 
        ckpt_path: str, 
        device: str, 
        config: dict) -> torch.nn.Module:
    """ 
    Prepare corrected model for evaluation. Brings model to eval-mode and to the desired device.
    For P-ClArC methods (weights remain unchanged), the projection hook is added to the model.

    Args:
        model (torch.nn.Module): Model
        dataset (Dataset): Train Dataset
        ckpt_path (str): path to model checkpoint
        device (str): device name
        config (dict): config

    Returns:
        torch.nn.Module: Model to be evaluated
    """
    
    method = config['method']
    kwargs_correction = {}
    correction_method = get_correction_method(method)

    if method == "PClarcFullFeature":
        ckpt_name = ".".join(os.path.basename(ckpt_path).split(".")[:-1])
        cav_dir = f"{os.path.dirname(ckpt_path)}/cavs_{ckpt_name}"
        kwargs_correction['cav_dir'] = cav_dir
        kwargs_correction['dataset'] = dataset
        kwargs_correction['class_name'] = config.get('class_name', None)
        kwargs_correction['artifact_name'] = config['artifact']
        model = correction_method(model, config, **kwargs_correction)        
        print("CAV dir", cav_dir, ckpt_name)

    elif "pclarc" in method.lower():
        mode = "crvs" if "gclarc" in method.lower() else "cavs_max"
        kwargs_correction['n_classes'] = len(dataset.class_names)
        kwargs_correction['artifact_sample_ids'] =  dataset.sample_ids_by_artifact[config['artifact']]
        kwargs_correction['mode'] = mode

        model = correction_method(model, config, **kwargs_correction)

    model = model.to(device)
    model.eval()
    return model
