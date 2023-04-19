import numpy as np
import torch
from PIL import Image
from pathlib import Path
import random
import torchvision.transforms as T
from datasets.isic import isic_augmentation
from datasets.isic_attacked import ISICAttackedDataset
from utils.artificial_artifact import insert_artifact

def get_isic_attacked_hm_dataset(data_paths, 
                                 normalize_data=True, 
                                 binary_target=False, 
                                 image_size=224, 
                                 p_artifact=None,
                                 attacked_classes=None,
                                 artifact_type="ch_text", 
                                 **kwargs):

    fns_transform = [
        T.Resize((image_size, image_size), interpolation=T.functional.InterpolationMode.BICUBIC),
        T.ToTensor()
    ]

    if normalize_data:
        fns_transform.append(T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]))

    transform = T.Compose(fns_transform)
    
    return ISICAttackedHmDataset(data_paths, train=True, transform=transform, augmentation=isic_augmentation,
                         binary_target=binary_target, p_artifact=p_artifact, attacked_classes=attacked_classes,
                         artifact_type=artifact_type, image_size=image_size)



class ISICAttackedHmDataset(ISICAttackedDataset):
    def __init__(self, 
                 data_paths, 
                 train=False, 
                 transform=None, 
                 augmentation=None,
                 binary_target=False, 
                 attacked_classes=[], 
                 p_artifact=.5,
                 artifact_type='ch_text',
                 image_size=224
                 ):
        super().__init__(data_paths, train, transform, augmentation, binary_target, 
                         attacked_classes, p_artifact, artifact_type, image_size)
        
    
    def __getitem__(self, i):
        row = self.metadata.loc[i]

        path = self.train_dirs_by_version[row.version] if self.train else self.test_dirs_by_version[row.version]
        img = Image.open(path / Path(row['image'] + '.jpg'))
        img = self.transform_resize(img)
        if self.artifact_labels[i]:
            img, mask = self.add_artifact(img, i)
        else:
            mask = torch.zeros((self.image_size, self.image_size))

        img = self.transform(img)
        columns = self.metadata.columns.to_list()
        target = torch.Tensor([columns.index(row[row == 1.0].index[0]) - 1 if self.train else 0]).long()[0]
        return img, target, mask.type(torch.uint8)

    