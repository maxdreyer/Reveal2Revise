import numpy as np
import torch
import torchvision.transforms as T
from datasets.isic import isic_augmentation
from datasets.isic_attacked import ISICAttackedDataset

def get_isic_attacked_cd_dataset(data_paths, 
                                 normalize_data=True, 
                                 binary_target=False, 
                                 attacked_classes=[], 
                                 p_artifact=.5, 
                                 artifact_type="ch_text",
                                 image_size=224, 
                                 seg_mask_source=None,
                                 model_name=None,
                                 **kwargs):

    fns_transform = [
        T.Resize((image_size, image_size), interpolation=T.functional.InterpolationMode.BICUBIC),
        T.ToTensor()
    ]

    if normalize_data:
        fns_transform.append(T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]))

    transform = T.Compose(fns_transform)
    
    return ISICAttackedCdDataset(data_paths, train=True, transform=transform, augmentation=isic_augmentation,
                         binary_target=binary_target, attacked_classes=attacked_classes, p_artifact=p_artifact,
                         artifact_type=artifact_type, image_size=image_size, 
                         seg_mask_source=seg_mask_source, model_name=model_name)


class ISICAttackedCdDataset(ISICAttackedDataset):
    def __init__(self, 
                 data_paths, 
                 train=False, 
                 transform=None, 
                 augmentation=None,
                 binary_target=False, 
                 attacked_classes=[], 
                 p_artifact=.5,
                 artifact_type="ch_text",
                 image_size=224,
                 seg_mask_source="",
                 model_name=""
                 ):
        super().__init__(data_paths, train, transform, augmentation, binary_target, attacked_classes, 
                         p_artifact, artifact_type, image_size)
        path = f"results/cd_preprocessing/{seg_mask_source}/{model_name}"
        self.cd_features = np.load(f"{path}/cd_features.npy")

    def __getitem__(self, i):
        img, target = super().__getitem__(i)
        return img, target, torch.from_numpy(self.cd_features[i]).float()

    