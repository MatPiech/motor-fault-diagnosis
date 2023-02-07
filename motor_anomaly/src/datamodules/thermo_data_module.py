import itertools
from pathlib import Path
from typing import Optional, Tuple, Dict, Sequence

import albumentations as A
from albumentations.pytorch import ToTensorV2
import hydra
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset


class ThermoDataModule(LightningDataModule):
    def __init__(self,
                 data_path: Path,
                 dataset: Dataset,
                 dataset_distribution: Dict[str, Sequence[str]],
                 augment: bool,
                 batch_size: int,
                 image_size: Tuple[int, int],
                 padded_image_size: Tuple[int, int],
                 image_mean: Tuple[float, float, float],
                 image_std: Tuple[float, float, float],
                 number_of_workers: int,
                 number_of_splits: int,
                 current_split: int,
                 ):
        super().__init__()

        self._data_root = Path(data_path)
        self._dataset = dataset
        self._dataset_distribution = dataset_distribution
        self._augment = augment
        self._batch_size = batch_size
        self._number_of_workers = number_of_workers
        self._number_of_splits = number_of_splits
        self._current_split = current_split

        if self._dataset.split('.')[-1] == 'WorkswellThermoDataset':
            self._dataset_folder = 'workswell_wic_640'
        elif self._dataset.split('.')[-1] == 'LeptonThermoDataset':
            self._dataset_folder = 'flir_lepton_3_5'
        else:
            raise ValueError('Unknown dataset')

        self._train_dataset = None
        self._valid_dataset = None
        self._test_dataset = None

        self._transforms = A.Compose([
            # A.CenterCrop(image_size[1], image_size[0]),
            # A.PadIfNeeded(padded_image_size[1], padded_image_size[0], border_mode=cv2.BORDER_CONSTANT, value=0),
            # A.Normalize(mean=image_mean, std=image_std),
            ToTensorV2()
        ])

        self._augmentations = A.Compose([
            # geometry augmentations
            A.Affine(rotate=(-10, 10), translate_px=(-10, 10), scale=(0.9, 1.1)),
            A.HorizontalFlip(),
            # transforms
            A.RandomCrop(image_size[1], image_size[0]),
            # A.PadIfNeeded(padded_image_size[1], padded_image_size[0], border_mode=cv2.BORDER_CONSTANT, value=0),
            # A.Normalize(mean=image_mean, std=image_std),
            ToTensorV2()
        ])

    def setup(self, stage: Optional[str] = None):
        train_split = [list((self._data_root / dir_name / self._dataset_folder).glob('*.png')) for dir_name in self._dataset_distribution['train']]
        train_split = list(itertools.chain(*train_split))

        valid_split = [list((self._data_root / dir_name / self._dataset_folder).glob('*.png')) for dir_name in self._dataset_distribution['valid']]
        valid_split = list(itertools.chain(*valid_split))

        test_split = [list((self._data_root / dir_name / self._dataset_folder).glob('*.png')) for dir_name in self._dataset_distribution['test']]
        test_split = list(itertools.chain(*test_split))

        self._train_dataset: Dataset = hydra.utils.instantiate({
                '_target_': self._dataset,
                'images_list': train_split,
                'augmentations': self._augmentations if self._augment else self._transforms,
            })

        self._valid_dataset: Dataset = hydra.utils.instantiate({
                '_target_': self._dataset,
                'images_list': valid_split,
                'augmentations': self._transforms,
            })

        self._test_dataset: Dataset = hydra.utils.instantiate({
                '_target_': self._dataset,
                'images_list': test_split,
                'augmentations': self._transforms,
            })

    def train_dataloader(self):
        return DataLoader(
            self._train_dataset, batch_size=self._batch_size, num_workers=self._number_of_workers,
            pin_memory=True, drop_last=True, shuffle=True
        )

    def val_dataloader(self):
        return DataLoader(
            self._valid_dataset, batch_size=self._batch_size, num_workers=self._number_of_workers,
            pin_memory=True
        )

    def test_dataloader(self):
        return DataLoader(
            self._test_dataset, batch_size=self._batch_size, num_workers=self._number_of_workers,
            pin_memory=True
        )

    def predict_dataloader(self):
        return DataLoader(
            self._test_dataset, batch_size=self._batch_size, num_workers=self._number_of_workers,
            pin_memory=True
        )
