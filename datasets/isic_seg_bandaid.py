import torch
import glob
from PIL import Image
import torchvision.transforms as T
from datasets.isic import ISICDataset, isic_augmentation

def get_isic_seg_bandaid_dataset(data_paths, 
                                 normalize_data=True, 
                                 binary_target=False, 
                                 image_size=224, 
                                 artifact_ids_file=None,
                                 **kwargs):

    segmentation_dir = kwargs['segmentation_dir']

    fns_transform = [
        T.Resize((image_size, image_size), interpolation=T.functional.InterpolationMode.BICUBIC),
        T.ToTensor()
    ]

    if normalize_data:
        fns_transform.append(T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]))

    transform = T.Compose(fns_transform)
    
    return ISICSegBandaidDataset(data_paths, train=True, transform=transform, augmentation=isic_augmentation,
                                 binary_target=binary_target, artifact_ids_file=artifact_ids_file, segmentation_dir=segmentation_dir)



class ISICSegBandaidDataset(ISICDataset):
    def __init__(self, 
                 data_paths, 
                 train=False, 
                 transform=None, 
                 augmentation=None,
                 binary_target=False,
                 artifact_ids_file=None,
                 segmentation_dir=""
                 ):
        super().__init__(data_paths, train, transform, augmentation, binary_target, artifact_ids_file)
        
        self.seg_dir = f"{segmentation_dir}/segmentation_masks"
        self.seg_names = glob.glob(f"{self.seg_dir}/*.jpg")
        print(f"Found {len(self.seg_names)} segmentation masks")
        self.transform = transform

        self.transform_mask = T.Compose([
            T.Resize((224, 224), interpolation=T.functional.InterpolationMode.BICUBIC),
            T.ToTensor()
            ])

        self.seg_ids = [name.split("/")[-1].replace(" - Copy", "").replace(".jpg", "") for name in self.seg_names]

    def get_mask(self, i):
        name = self.ids['image'][i].replace("_downsampled", "")
        if not name in self.seg_ids:
            mask = None
        else:
            name_mask = self.seg_names[self.seg_ids.index(name)]
            mask = Image.open(name_mask)
            mask = self.transform_mask(mask)
            mask = (mask.mean(0) > .5).type(torch.uint8)

        return mask
        
    
    def __getitem__(self, i):

        img, target = super().__getitem__(i)
        mask = self.get_mask(i)

        return img, target, mask
