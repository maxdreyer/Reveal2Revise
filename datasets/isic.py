import copy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
from PIL import Image

from datasets.base_dataset import BaseDataset

PATHS_BY_DATASET_VERSION = {
    '2019': {
        'train': 'Train',
        'test': 'Test'
    },
    '2020': {
        'train': 'train',
        'test': 'test'
    }
}

GROUND_TRUTH_FILES_BY_VERSION = {
    '2019': 'ISIC_2019_Training_GroundTruth.csv',
    '2020': 'ISIC_2020_Training_GroundTruth.csv'
}

isic_augmentation = T.Compose([
    T.RandomHorizontalFlip(p=.25),
    T.RandomVerticalFlip(p=.25),
    T.RandomApply(transforms=[T.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5))], p=.25),
    T.RandomApply(transforms=[T.RandomRotation(degrees=(0, 30))], p=.25),
    T.RandomApply(transforms=[T.ColorJitter(brightness=.1, saturation=.1, hue=.1)], p=.25)
])


def get_isic_dataset(data_paths,
                     normalize_data=True,
                     binary_target=False,
                     image_size=224,
                     artifact_ids_file=None,
                     **kwargs):
    fns_transform = [
        T.Resize((image_size, image_size), interpolation=T.functional.InterpolationMode.BICUBIC),
        T.ToTensor()
    ]

    if normalize_data:
        fns_transform.append(T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]))

    transform = T.Compose(fns_transform)

    return ISICDataset(data_paths, train=True, transform=transform, augmentation=isic_augmentation,
                       binary_target=binary_target, artifact_ids_file=artifact_ids_file)


def get_version(dir):
    if "2019" in dir:
        return "2019"
    elif "2020" in dir:
        return "2020"
    else:
        print("Unknown ISIC version. Default is 2019.")
        return "2019"


class ISICDataset(BaseDataset):
    def __init__(self,
                 data_paths,
                 train=False,
                 transform=None,
                 augmentation=None,
                 binary_target=False,
                 artifact_ids_file=None
                 ):
        super().__init__(data_paths, train, transform, augmentation, artifact_ids_file)
        self.classes = ['MEL', 'NV', 'BCC', 'AK', 'BKL', 'DF', 'VASC', 'SCC', 'UNK']
        self.normalize_fn = T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])

        paths_by_version = {get_version(data_path): data_path for data_path in data_paths}
        self.train_dirs_by_version = {version: dir / Path(PATHS_BY_DATASET_VERSION[version]['train'])
                                      for version, dir in paths_by_version.items()}
        self.test_dirs_by_version = {version: dir / Path(PATHS_BY_DATASET_VERSION[version]['test'])
                                     for version, dir in paths_by_version.items()}

        self.binary_target = binary_target

        self.metadata = self.construct_metadata(paths_by_version)

        # Set Class Names
        if self.binary_target:
            self.class_names = ['Benign', 'MEL']
            num_mel = self.metadata['MEL'].sum()
            dist = np.array([len(self.metadata) - num_mel, num_mel])
        else:
            self.class_names = ['MEL', 'NV', 'BCC', 'AK', 'BKL', 'DF', 'VASC', 'SCC', 'UNK']
            dist = np.array([float(x) for x in self.metadata.agg(sum).values[1:1 + len(self.class_names)]])

        self.weights = self.compute_weights(dist)

        self.idxs_train, self.idxs_val, self.idxs_test = self.do_train_val_test_split(.1, .1)
        self.sample_ids_by_artifact = self.get_sample_ids_by_artifact()

        self.all_artifact_sample_ids = [sample_id for _, sample_ids in self.sample_ids_by_artifact.items() for sample_id
                                        in sample_ids]
        self.clean_sample_ids = list(set(np.arange(len(self))) - set(self.all_artifact_sample_ids))

    def construct_metadata(self, dirs_by_version):
        tables = []
        for version, dir in dirs_by_version.items():
            data = pd.read_csv(dir / Path(GROUND_TRUTH_FILES_BY_VERSION[version]))
            data = self.prepare_metadata_by_version(version, data)
            data['version'] = version
            tables.append(data)

        data_combined = pd.concat(tables).reset_index(drop=True)
        data_combined['isic_id'] = data_combined.image.str.replace('_downsampled', '')
        data_combined = data_combined.drop_duplicates(subset=['isic_id'], keep='last').reset_index(drop=True)
        return data_combined

    def prepare_metadata_by_version(self, version, data):
        if version == '2019':
            return data
        elif version == '2020':

            diagnosis_map = {
                'nevus': 'NV',
                'melanoma': 'MEL',
                'seborrheic keratosis': 'BKL',
                'lentigo NOS': 'BKL',
                'lichenoid keratosis': 'BKL',
                'solar lentigo': 'BKL'
            }

            data['class_label'] = [diagnosis_map.get(x.diagnosis, 'UNK') for _, x in data.iterrows()]

            data_new = pd.DataFrame({'image': data.image_name.values,
                                     **{c: (data.class_label.values == c).astype(int) for c in self.classes}})

            return data_new
        else:
            raise ValueError(f"Unknown ISIC version ({version})")

    def get_all_ids(self):
        return list(self.metadata.image.values)

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, i):

        row = self.metadata.loc[i]

        path = self.train_dirs_by_version[row.version] if self.train else self.test_dirs_by_version[row.version]
        img = Image.open(path / Path(row['image'] + '.jpg'))

        img = self.transform(img)
        if self.do_augmentation:
            img = self.augmentation(img)

        columns = self.metadata.columns.to_list()

        if self.binary_target:
            # 1 = MEL, 0 = non-MEL
            target = (row['MEL'] == 1).astype(int)
        else:
            target = torch.Tensor([columns.index(row[row == 1.0].index[0]) - 1 if self.train else 0]).long()[0]

        return img, target

    def get_sample_name(self, i):
        return self.metadata.loc[i]["image"]

    def get_target(self, i):
        targets = self.metadata.loc[i]
        columns = self.metadata.columns.to_list()
        target = torch.Tensor([columns.index(targets[targets == 1.0].index[0]) - 1 if self.train else 0]).long()
        return target

    def get_subset_by_idxs(self, idxs):
        subset = copy.deepcopy(self)
        subset.metadata = subset.metadata.iloc[idxs].reset_index(drop=True)
        return subset
