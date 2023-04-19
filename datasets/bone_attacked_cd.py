import torch
import torchvision.transforms as T
from datasets.bone import bone_augmentation
from datasets.bone_attacked import BoneAttackedDataset, bone_augmentation
import numpy as np

def get_bone_attacked_cd_dataset(data_paths, 
                                 normalize_data=True, 
                                 image_size=224, 
                                 attacked_classes=[], 
                                 p_artifact=.5, 
                                 artifact_type="ch_text",
                                 seg_mask_source=None,
                                 model_name=None, 
                                 **kwargs):

    fns_transform = [
        T.Resize((image_size, image_size), interpolation=T.functional.InterpolationMode.BICUBIC),
        T.ToTensor()
    ]

    if normalize_data:
        fns_transform.append(T.Normalize([46.9 / 255.], [22.65 / 255.]))

    transform = T.Compose(fns_transform)

    return BoneAttackedCdDataset(data_paths, train=True, transform=transform, augmentation=bone_augmentation,
                                 attacked_classes=attacked_classes, p_artifact=p_artifact, artifact_type=artifact_type,
                                 seg_mask_source=seg_mask_source, model_name=model_name)

class BoneAttackedCdDataset(BoneAttackedDataset):
    def __init__(self, data_paths, train=True, transform=None, augmentation=None, 
                 attacked_classes=[], p_artifact=.5, artifact_type='ch_text', 
                 img_size=224, seg_mask_source=None, model_name=None):
        super().__init__(data_paths, train, transform, augmentation, attacked_classes,
                         p_artifact, artifact_type, img_size=img_size)

        path = f"results/cd_preprocessing/{seg_mask_source}/{model_name}"
        self.cd_features = np.load(f"{path}/cd_features.npy")

    def __getitem__(self, i):
        img, target = super().__getitem__(i)
        return img, target, torch.from_numpy(self.cd_features[i]).float()
