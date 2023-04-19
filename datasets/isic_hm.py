import numpy as np
import torch
import glob
from PIL import Image
import copy
import torchvision.transforms as T
from datasets.isic import ISICDataset, isic_augmentation

def get_isic_hm_dataset(data_paths, 
                        normalize_data=True, 
                        binary_target=False, 
                        image_size=224, 
                        artifact_ids_file=None,
                        artifact=None,
                        **kwargs):

    fns_transform = [
        T.Resize((image_size, image_size), interpolation=T.functional.InterpolationMode.BICUBIC),
        T.ToTensor()
    ]

    if normalize_data:
        fns_transform.append(T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]))

    transform = T.Compose(fns_transform)
    
    return ISICHmDataset(data_paths, train=True, transform=transform, augmentation=isic_augmentation,
                         binary_target=binary_target, artifact_ids_file=artifact_ids_file, artifact=artifact)



class ISICHmDataset(ISICDataset):
    def __init__(self, 
                 data_paths, 
                 train=False, 
                 transform=None, 
                 augmentation=None,
                 binary_target=False,
                 artifact_ids_file=None,
                 artifact=None
                 ):
        super().__init__(data_paths, train, transform, augmentation, binary_target, artifact_ids_file)
        
        # TODO: Refactor
        self.hm_path = f"data/localized_artifacts/isic"
        artifacts = artifact.split("-")
        artifact_paths = []
        for artifact in artifacts:
            print("LOADING", artifact)
            artifact_paths += glob.glob(f"{self.hm_path}/{artifact}/*")
        print(f"Localized artifacts: {len(artifact_paths)}")
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

        img, target = super().__getitem__(i)
        if self.metadata["hms"].loc[i]:
            # TODO: LOOKS GOOD FOR MULTIPLE ARTIFACTS OR SHOULD NORMALIZE EACH?
            heatmaps = torch.stack(
                [torch.tensor(np.asarray(Image.open(hm))) for hm in self.metadata["hms"].loc[i].split(",")]).clamp(
                min=0)
            # sum heatmaps after normalizing each one
            heatmaps = heatmaps / heatmaps.flatten(start_dim=1).max(dim=1).values[:, None, None]
            heatmaps = heatmaps.sum(dim=0).float()
        else:
            heatmaps = torch.zeros_like(img[0])

        return img, target, heatmaps
