from PIL import Image
import torch
import torchvision.transforms as T
from utils.artificial_artifact import insert_artifact
import random
from datasets.bone import bone_augmentation
from datasets.bone_attacked import BoneAttackedDataset, bone_augmentation
import numpy as np

def get_bone_attacked_hm_dataset(data_paths, normalize_data=True, image_size=224, 
                                 attacked_classes=[], p_artifact=.5,  artifact_type="ch_text", **kwargs):

    fns_transform = [
        T.Resize((image_size, image_size), interpolation=T.functional.InterpolationMode.BICUBIC),
        T.ToTensor()
    ]

    if normalize_data:
        fns_transform.append(T.Normalize([46.9 / 255.], [22.65 / 255.]))

    transform = T.Compose(fns_transform)

    return BoneAttackedHmDataset(data_paths, train=True, transform=transform, augmentation=bone_augmentation,
                               attacked_classes=attacked_classes, p_artifact=p_artifact, artifact_type=artifact_type)

class BoneAttackedHmDataset(BoneAttackedDataset):
    def __init__(self, data_paths, train=True, transform=None, augmentation=None, 
                 attacked_classes=[], p_artifact=.5, artifact_type='ch_text', img_size=224):
        super().__init__(data_paths, train, transform, augmentation, attacked_classes,
                         p_artifact, artifact_type, img_size=img_size)


    def __getitem__(self, idx):
        img_name = f"{self.path}/{self.metadata.iloc[idx, 0]}.png"
        image =  Image.open(img_name).convert("RGB")
        # gender = np.atleast_1d(self.metadata.iloc[idx, 2])
        bone_age = torch.tensor(self.metadata.iloc[idx]["target"])

        image = self.transform_resize(image)

        if self.artifact_labels[idx]:
            image, mask = self.add_artifact(image, idx)
        else:
            mask = torch.zeros((self.img_size, self.img_size))

        if self.transform:
            image = self.transform(image)

        if self.do_augmentation:
            image = self.augmentation(image)

        return image.float(), bone_age, mask.type(torch.uint8)
