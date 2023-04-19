from PIL import Image
import torch
import torchvision.transforms as T
from utils.artificial_artifact import insert_artifact
import random
from datasets.bone import BoneDataset, bone_augmentation
import numpy as np

def get_bone_attacked_dataset(data_paths, normalize_data=True, image_size=224, 
                              attacked_classes=[], p_artifact=.5, artifact_type="ch_text", **kwargs):

    fns_transform = [
        T.Resize((image_size, image_size), interpolation=T.functional.InterpolationMode.BICUBIC),
        T.ToTensor()
    ]

    if normalize_data:
        fns_transform.append(T.Normalize([46.9 / 255.], [22.65 / 255.]))

    transform = T.Compose(fns_transform)

    return BoneAttackedDataset(data_paths, train=True, transform=transform, augmentation=bone_augmentation,
                               attacked_classes=attacked_classes, p_artifact=p_artifact, artifact_type=artifact_type, 
                               img_size=image_size)

class BoneAttackedDataset(BoneDataset):
    def __init__(self, data_paths, train=True, transform=None, augmentation=None, 
                 attacked_classes=[], p_artifact=.5, artifact_type='ch_text',
                 img_size=224):
        super().__init__(data_paths, train, transform, augmentation, None)

        self.attacked_classes = attacked_classes
        print(f"Attacking classes: {attacked_classes}")

        self.img_size = img_size
        self.transform_resize = T.Resize((img_size, img_size))
        self.p_artifact = p_artifact
        self.artifact_type = artifact_type

        np.random.seed(0)
        self.artifact_labels = np.array([torch.tensor(self.metadata.iloc[idx]["target"]) in self.attacked_classes and
                                        np.random.rand() < self.p_artifact
                                        for idx in range(len(self))])
        
        self.artifact_ids = np.where(self.artifact_labels)[0]

        self.sample_ids_by_artifact = {"artificial": self.artifact_ids}
        self.clean_sample_ids = [i for i in range(len(self)) if i not in self.artifact_ids]

        print(f"Artifacts ({len(self.artifact_ids)}) / Clean ({len(self.clean_sample_ids)})")
    
    def add_artifact(self, img, idx):
        random.seed(idx); torch.manual_seed(idx); np.random.seed(idx)
        kwargs = {
                'text': "Clever Hans",
                'fill': (0,0,0),
                'img_size': self.img_size
                } if self.artifact_type == "ch_text" else {
                'factor': 1.5
                }
        return insert_artifact(img, self.artifact_type, **kwargs)
    
    def __getitem__(self, idx):
        img_name = f"{self.path}/{self.metadata.iloc[idx, 0]}.png"
        image =  Image.open(img_name).convert("RGB")
        # gender = np.atleast_1d(self.metadata.iloc[idx, 2])
        bone_age = torch.tensor(self.metadata.iloc[idx]["target"])

        image = self.transform_resize(image)

        if self.artifact_labels[idx]:
            image, _ = self.add_artifact(image, idx)

        if self.transform:
            image = self.transform(image)

        if self.do_augmentation:
            image = self.augmentation(image)

        return image.float(), bone_age
