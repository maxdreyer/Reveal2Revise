import copy
import logging

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
from PIL import Image

from datasets.base_dataset import BaseDataset

bone_augmentation = T.Compose([
    # T.RandomHorizontalFlip(p=.25),
    T.RandomVerticalFlip(p=.5),
    # T.RandomApply(transforms=[T.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5))], p=.25),
    T.RandomApply(transforms=[T.RandomRotation(degrees=(0, 30))], p=5),
    T.RandomApply(transforms=[T.ColorJitter(brightness=.1, saturation=.1, hue=.1)], p=.25),
    T.RandomApply(transforms=[T.Pad(10, fill=-(46.9 / 255.) / (22.6 / 255.)), T.Resize(224)], p=.25)
])


def get_bone_dataset(data_paths, normalize_data=True, image_size=224, artifact_ids_file=None, **kwargs):
    fns_transform = [
        T.Resize((image_size, image_size), interpolation=T.functional.InterpolationMode.BICUBIC),
        T.ToTensor()
    ]

    if normalize_data:
        fns_transform.append(T.Normalize([46.9 / 255.], [22.65 / 255.]))

    transform = T.Compose(fns_transform)

    return BoneDataset(data_paths, train=True, transform=transform, augmentation=bone_augmentation,
                       artifact_ids_file=artifact_ids_file)


class BoneDataset(BaseDataset):
    def __init__(self, data_paths, train=True, transform=None, augmentation=None, artifact_ids_file=None):
        super().__init__(data_paths, train, transform, augmentation, artifact_ids_file)
        assert len(data_paths) == 1, "Only 1 path accepted for Bone Dataset"

        self.normalize_fn = T.Normalize([46.9 / 255.], [22.65 / 255.])
        self.path = f"{data_paths[0]}/boneage-training-dataset/"
        csv_path = f"{data_paths[0]}/train.csv"

        self.metadata = pd.read_csv(csv_path)

        self.metadata.iloc[:, 1:3] = self.metadata.iloc[:, 1:3].astype(np.float32)
        self.metadata['category'] = pd.cut(self.metadata['boneage'], 5)
        self.classes = [str(x) for x in self.metadata.groupby(['category']).count().index.values]
        self.class_names = self.classes
        self.metadata["target"] = self.metadata['category'].apply(lambda x: self.class_names.index(str(x))).astype(int)

        self.mean = torch.Tensor([46.9 / 255., 46.9 / 255., 46.9 / 255.])
        self.var = torch.Tensor([22.6 / 255., 22.6 / 255., 22.6 / 255.])
        dist = self.metadata.groupby(['category']).count()['id'].values
        self.weights = self.compute_weights(dist)

        logging.info(f'Creating dataset with {len(self.metadata)} examples')

        self.idxs_train, self.idxs_val, self.idxs_test = self.do_train_val_test_split(.1, .1)

        self.sample_ids_by_artifact = self.get_sample_ids_by_artifact()

        self.all_artifact_sample_ids = [sample_id for _, sample_ids in self.sample_ids_by_artifact.items() for sample_id
                                        in sample_ids]
        self.clean_sample_ids = list(set(np.arange(len(self))) - set(self.all_artifact_sample_ids))

    def get_all_ids(self):
        return list(self.metadata['id'].values)

    def __len__(self):
        return self.metadata.shape[0]

    def __getitem__(self, idx):
        img_name = f"{self.path}/{self.metadata.iloc[idx, 0]}.png"
        image = Image.open(img_name).convert("RGB")
        # gender = np.atleast_1d(self.metadata.iloc[idx, 2])
        bone_age = torch.tensor(self.metadata.iloc[idx]["target"])

        if self.transform:
            image = self.transform(image)

        if self.do_augmentation:
            image = self.augmentation(image)

        return image.float(), bone_age

    def get_sample_name(self, i):
        return self.metadata.iloc[i, 0]

    def get_target(self, i):
        target = torch.tensor(self.metadata.iloc[i]["target"])
        return target

    def get_subset_by_idxs(self, idxs):
        subset = copy.deepcopy(self)
        subset.metadata = subset.metadata.iloc[idxs].reset_index(drop=True)
        return subset
