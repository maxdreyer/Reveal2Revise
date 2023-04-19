import numpy as np
import torch
import torchvision.transforms as T

from datasets.bone import BoneDataset, bone_augmentation


def get_bone_cd_dataset(data_paths, 
                        normalize_data=True, 
                        image_size=224, 
                        artifact_ids_file=None,
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

    return BoneCdDataset(data_paths, train=True, transform=transform, augmentation=bone_augmentation,
                         artifact_ids_file=artifact_ids_file, seg_mask_source=seg_mask_source, model_name=model_name)


class BoneCdDataset(BoneDataset):
    def __init__(self, 
                 data_paths, 
                 train=False, 
                 transform=None, 
                 augmentation=None,
                 artifact_ids_file=None,
                 seg_mask_source=None,
                 model_name=None
                 ):

        super().__init__(data_paths, train, transform, augmentation, artifact_ids_file)

        path = f"results/cd_preprocessing/{seg_mask_source}/{model_name}"
        # img_features = np.load(f"{path}/img_features.npy")
        self.cd_features = np.load(f"{path}/cd_features.npy")

    def __getitem__(self, item):
        img, target = super().__getitem__(item)
        return img, target, torch.from_numpy(self.cd_features[item]).float()
