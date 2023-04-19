import glob

import cv2
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

from datasets.bone import BoneDataset, bone_augmentation


def get_bone_hm_dataset(data_paths, normalize_data=True, image_size=224, artifact_ids_file=None, artifact=None, **kwargs):
    fns_transform = [
        T.Resize((image_size, image_size), interpolation=T.functional.InterpolationMode.BICUBIC),
        T.ToTensor()
    ]

    if normalize_data:
        fns_transform.append(T.Normalize([46.9 / 255.], [22.65 / 255.]))

    transform = T.Compose(fns_transform)

    return BoneHmDataset(data_paths, train=True, transform=transform, augmentation=bone_augmentation,
                         artifact_ids_file=artifact_ids_file, artifact=artifact)


class BoneHmDataset(BoneDataset):
    def __init__(self, 
                 data_paths, 
                 train=False, 
                 transform=None, 
                 augmentation=None,
                 artifact_ids_file=None,
                 artifact=None
                 ):

        super().__init__(data_paths, train, transform, augmentation, artifact_ids_file)

        self.hm_path = f"data/localized_artifacts/bone"
        artifact_paths = glob.glob(f"{self.hm_path}/{artifact}/*")
        artifact_sample_ids = np.array([int(x.split("/")[-1].split(".")[0]) for x in artifact_paths])
        self.artifact_ids = artifact_sample_ids
        self.hms = ["" for _ in range(len(self.metadata))]
        for i, j in enumerate(artifact_sample_ids):
            path = artifact_paths[i]
            if self.hms[j]:
                self.hms[j] += f",{path}"
            else:
                self.hms[j] += f"{path}"

        self.metadata["hms"] = self.hms

    def __getitem__(self, i):
        image, bone_age = super().__getitem__(i)

        if self.metadata["hms"].loc[i]:
            # print(self.hms[item].split(","))
            # TODO: LOOKS GOOD FOR MULTIPLE ARTIFACTS OR SHOULD NORMALIZE EACH?
            heatmaps = torch.stack(
                [torch.tensor(np.asarray(Image.open(hm))) for hm in self.metadata["hms"].loc[i].split(",")]).clamp(
                min=0).sum(0).float()
        else:
            heatmaps = torch.zeros_like(image[0])

        return image, bone_age, heatmaps