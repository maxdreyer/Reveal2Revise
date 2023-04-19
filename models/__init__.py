import logging

import torch

from models.efficientnet import get_efficientnet_b0, get_efficientnet_b4, get_efficientnet_canonizer
from models.resnet import get_resnet18, get_resnet34, get_resnet50, get_resnet_canonizer
from models.vgg import get_vgg16, get_vgg16_bn, get_vgg11, get_vgg11_bn, get_vgg13_bn, get_vgg13, get_vgg_canonizer

logger = logging.getLogger(__name__)

MODELS = {
    "vgg16": get_vgg16,
    "vgg16_bn": get_vgg16_bn,
    "vgg13": get_vgg13,
    "vgg13_bn": get_vgg13_bn,
    "vgg11": get_vgg11,
    "vgg11_bn": get_vgg11_bn,
    "efficientnet_b0": get_efficientnet_b0,
    "efficientnet_b4": get_efficientnet_b4,
    "resnet18": get_resnet18,
    "resnet34": get_resnet34,
    "resnet50": get_resnet50,
}

CANONIZERS = {
    "vgg16": get_vgg_canonizer,
    "vgg16_bn": get_vgg_canonizer,
    "vgg13": get_vgg_canonizer,
    "vgg13_bn": get_vgg_canonizer,
    "vgg11": get_vgg_canonizer,
    "vgg11_bn": get_vgg_canonizer,
    "efficientnet_b0": get_efficientnet_canonizer,
    "efficientnet_b4": get_efficientnet_canonizer,
    "resnet18": get_resnet_canonizer,
    "resnet34": get_resnet_canonizer,
    "resnet50": get_resnet_canonizer,
}


def get_canonizer(model_name):
    assert model_name in list(CANONIZERS.keys()), f"No canonizer for model '{model_name}' available"
    return [CANONIZERS[model_name]()]


def get_fn_model_loader(model_name: str) -> torch.nn.Module:
    if model_name in MODELS:
        fn_model_loader = MODELS[model_name]
        logger.info(f"Loading {model_name}")
        return fn_model_loader
    else:
        raise KeyError(f"Model {model_name} not available")
