import numpy as np
import torch
import torchvision.transforms as T
from datasets.isic import ISICDataset, isic_augmentation

def get_isic_cd_dataset(data_paths, 
                        normalize_data=True, 
                        binary_target=False, 
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
        fns_transform.append(T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]))

    transform = T.Compose(fns_transform)
    
    return ISICCdDataset(data_paths, train=True, transform=transform, augmentation=isic_augmentation,
                         binary_target=binary_target, artifact_ids_file=artifact_ids_file,
                         seg_mask_source=seg_mask_source, model_name=model_name)


class ISICCdDataset(ISICDataset):
    def __init__(self, 
                 data_paths, 
                 train=False, 
                 transform=None, 
                 augmentation=None,
                 binary_target=False,
                 artifact_ids_file=None,
                 seg_mask_source="",
                 model_name=""
                 ):
        super().__init__(data_paths, train, transform, augmentation, binary_target, artifact_ids_file)
        
        path = f"results/cd_preprocessing/{seg_mask_source}/{model_name}"
        self.cd_features = np.load(f"{path}/cd_features.npy")

    def __getitem__(self, i):
        img, target = super().__getitem__(i)
        return img, target, torch.from_numpy(self.cd_features[i]).float()
