import itertools
from collections import deque
from pathlib import Path
from random import Random
from typing import Optional, Tuple, Sequence

import albumentations as A
from albumentations.pytorch import ToTensorV2
import hydra
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset


class ThermoDataModule(LightningDataModule):
    def __init__(
        self,
        data_path: Path,
        dataset: Dataset,
        datasets_list: Sequence[str],
        augment: bool,
        batch_size: int,
        image_size: Tuple[int, int],
        number_of_workers: int,
        number_of_splits: int,
        current_split: int,
        seed: int,
    ):
        super().__init__()

        self._data_root = Path(data_path)
        self._dataset = dataset
        self._datasets_list = datasets_list
        self._augment = augment
        self._batch_size = batch_size
        self._number_of_workers = number_of_workers
        self._number_of_splits = number_of_splits
        self._current_split = current_split
        self._seed = seed

        if self._dataset.split(".")[-1] == "WorkswellThermoDataset":
            self._dataset_folder = "workswell_wic_640"
        else:
            raise ValueError("Unknown dataset")

        self._train_dataset = None
        self._valid_dataset = None
        self._test_dataset = None

        self._transforms = A.Compose(
            [A.Resize(image_size[1], image_size[0]), ToTensorV2()]
        )

        self._augmentations = A.Compose(
            [
                # geometry augmentations
                A.Affine(rotate=(-10, 10), translate_px=(-10, 10), scale=(0.9, 1.1)),
                A.HorizontalFlip(),
                # transforms
                A.RandomCrop(image_size[1], image_size[0]),
                ToTensorV2(),
            ]
        )

    @staticmethod
    def partition_sequences(sequences: list[str], n: int, seed: int) -> list[list[str]]:
        sequences = sequences.copy()
        Random(seed).shuffle(sequences)
        return [sequences[i::n] for i in range(n)]

    @staticmethod
    def get_train_valid_test(splits: list[list[str]], current_split: int):
        splits = deque(splits)
        splits.rotate(current_split)
        splits = list(splits)

        return list(itertools.chain.from_iterable(splits[:-2])), splits[-2], splits[-1]

    def setup(self, stage: Optional[str] = None):
        splits = self.partition_sequences(
            self._datasets_list, self._number_of_splits, self._seed
        )
        train_split, valid_split, test_split = self.get_train_valid_test(
            splits, self._current_split
        )

        train_images_list = [
            list((self._data_root / dir_name / self._dataset_folder).glob("*.png"))
            for dir_name in train_split
        ]
        train_images_list = list(itertools.chain(*train_images_list))

        valid_images_list = [
            list((self._data_root / dir_name / self._dataset_folder).glob("*.png"))
            for dir_name in valid_split
        ]
        valid_images_list = list(itertools.chain(*valid_images_list))

        test_images_list = [
            list((self._data_root / dir_name / self._dataset_folder).glob("*.png"))
            for dir_name in test_split
        ]
        test_images_list = list(itertools.chain(*test_images_list))

        self._train_dataset: Dataset = hydra.utils.instantiate(
            {
                "_target_": self._dataset,
                "images_list": train_images_list,
                "augmentations": self._augmentations if self._augment else self._transforms,
            }
        )

        self._valid_dataset: Dataset = hydra.utils.instantiate(
            {
                "_target_": self._dataset,
                "images_list": valid_images_list,
                "augmentations": self._transforms,
            }
        )

        self._test_dataset: Dataset = hydra.utils.instantiate(
            {
                "_target_": self._dataset,
                "images_list": test_images_list,
                "augmentations": self._transforms,
            }
        )

    def train_dataloader(self):
        return DataLoader(
            self._train_dataset,
            batch_size=self._batch_size,
            num_workers=self._number_of_workers,
            pin_memory=True,
            drop_last=True,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self._valid_dataset,
            batch_size=self._batch_size,
            num_workers=self._number_of_workers,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self._test_dataset,
            batch_size=self._batch_size,
            num_workers=self._number_of_workers,
            pin_memory=True,
        )

    def predict_dataloader(self):
        return DataLoader(
            self._test_dataset,
            batch_size=self._batch_size,
            num_workers=self._number_of_workers,
            pin_memory=True,
        )
