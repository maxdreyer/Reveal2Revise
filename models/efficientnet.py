import torch
import torch.hub
from torch import Tensor
from torchvision.models import efficientnet_b0, efficientnet_b4

from utils.lrp_canonizers import EfficientNetBNCanonizer


def get_efficientnet_b0(ckpt_path=None, pretrained=True, n_class: int = None) -> torch.nn.Module:
    return get_efficientnet(efficientnet_b0, ckpt_path, pretrained, n_class)


def get_efficientnet_b4(ckpt_path=None, pretrained=True, n_class: int = None) -> torch.nn.Module:
    return get_efficientnet(efficientnet_b4, ckpt_path, pretrained, n_class)


def get_efficientnet(model_fn, ckpt_path=None, pretrained=True, n_class: int = None) -> torch.nn.Module:
    if pretrained:
        weights = "IMAGENET1K_V1"
    else:
        weights = None

    model = model_fn(weights=weights)

    if n_class:
        classifier = list(model.classifier.children())
        model.classifier = torch.nn.Sequential(*classifier[:-1])
        model.classifier.add_module('last', torch.nn.Linear(classifier[-1].in_features, n_class))
    if ckpt_path:
        checkpoint = torch.load(ckpt_path)
        if "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"]
        model.load_state_dict(checkpoint)

    for i in range(len(model.features) - 1):
        print("added identity", i)
        setattr(model, f"identity_{i}", torch.nn.Identity())
    model.last_conv = torch.nn.Identity()
    model.last_relu = torch.nn.ReLU(inplace=False)
    model._forward_impl = _forward_impl_modified.__get__(model)

    return model


def _forward_impl_modified(self, x: Tensor) -> Tensor:
    for i in range(len(self.features)):
        x = self.features[i](x)
        if hasattr(self, f"identity_{i}"):
            x = getattr(self, f"identity_{i}")(x)

    x = self.last_relu(self.last_conv(x))  # added identity

    x = self.avgpool(x)
    x = torch.flatten(x, 1)

    x = self.classifier(x)

    return x


def get_efficientnet_canonizer():
    return EfficientNetBNCanonizer()
