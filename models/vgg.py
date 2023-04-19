import torch
import torch.hub
from torchvision.models import vgg16, vgg16_bn, vgg11, vgg13, vgg13_bn, vgg11_bn
from zennit.torchvision import VGGCanonizer


def get_vgg16(ckpt_path=None, pretrained=True, n_class: int = None) -> torch.nn.Module:
    return get_vgg(vgg16, ckpt_path, pretrained, n_class)


def get_vgg16_bn(ckpt_path=None, pretrained=True, n_class=None) -> torch.nn.Module:
    return get_vgg(vgg16_bn, ckpt_path, pretrained, n_class)


def get_vgg13(ckpt_path=None, pretrained=True, n_class: int = None) -> torch.nn.Module:
    return get_vgg(vgg13, ckpt_path, pretrained, n_class)


def get_vgg13_bn(ckpt_path=None, pretrained=True, n_class: int = None) -> torch.nn.Module:
    return get_vgg(vgg13_bn, ckpt_path, pretrained, n_class)


def get_vgg11(ckpt_path=None, pretrained=True, n_class: int = None) -> torch.nn.Module:
    return get_vgg(vgg11, ckpt_path, pretrained, n_class)


def get_vgg11_bn(ckpt_path=None, pretrained=True, n_class: int = None) -> torch.nn.Module:
    return get_vgg(vgg11_bn, ckpt_path, pretrained, n_class)


def get_vgg(model_fn, ckpt_path=None, pretrained=True, n_class: int = None) -> torch.nn.Module:
    if pretrained:
        weights = "IMAGENET1K_V1"
    else:
        weights = None

    model = model_fn(weights=weights)

    if n_class:
        model.classifier[-1] = torch.nn.Linear(4096, n_class, bias=True)
    if ckpt_path:
        checkpoint = torch.load(ckpt_path)
        if "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"]
        if "module" in list(checkpoint.keys())[0]:
            checkpoint = {k.replace("module.", ""): v for k, v in checkpoint.items()}
        checkpoint = {k.replace("classifier.last", "classifier.6"): v for k, v in checkpoint.items()}  # ISIC MODEL
        model.load_state_dict(checkpoint)
    return model


def get_vgg_canonizer():
    return VGGCanonizer()
