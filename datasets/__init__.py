import logging
from typing import Callable

from datasets.bone import get_bone_dataset
from datasets.bone_attacked import get_bone_attacked_dataset
from datasets.bone_attacked_cd import get_bone_attacked_cd_dataset
from datasets.bone_attacked_hm import get_bone_attacked_hm_dataset
from datasets.bone_cd import get_bone_cd_dataset
from datasets.bone_hm import get_bone_hm_dataset

from datasets.isic import get_isic_dataset
from datasets.isic_hm import get_isic_hm_dataset
from datasets.isic_cd import get_isic_cd_dataset
from datasets.isic_attacked import get_isic_attacked_dataset
from datasets.isic_attacked_hm import get_isic_attacked_hm_dataset
from datasets.isic_attacked_cd import get_isic_attacked_cd_dataset
from datasets.isic_seg_bandaid import get_isic_seg_bandaid_dataset

logger = logging.getLogger(__name__)

DATASETS = {
    "isic": get_isic_dataset,
    "isic_hm": get_isic_hm_dataset,
    "isic_seg_bandaid": get_isic_seg_bandaid_dataset,
    "isic_cd": get_isic_cd_dataset,
    "isic_attacked": get_isic_attacked_dataset,
    "isic_attacked_hm": get_isic_attacked_hm_dataset,
    "isic_attacked_cd": get_isic_attacked_cd_dataset,
    "bone": get_bone_dataset,
    "bone_hm": get_bone_hm_dataset,
    "bone_cd": get_bone_cd_dataset,
    "bone_attacked": get_bone_attacked_dataset,
    "bone_attacked_hm": get_bone_attacked_hm_dataset,
    "bone_attacked_cd": get_bone_attacked_cd_dataset,
    
}


def get_dataset(dataset_name: str) -> Callable:
    """
    Get dataset by name.
    :param dataset_name: Name of the dataset.
    :return: Dataset.

    """
    if dataset_name in DATASETS:
        dataset = DATASETS[dataset_name]
        logger.info(f"Loading {dataset_name}")
        return dataset
    else:
        raise KeyError(f"DATASET {dataset_name} not defined.")
