import torch
from pangaea.datasets.base import RawGeoFMDataset
import pandas as pd
import numpy as np
from pathlib import Path

import xarray as xr
import zarr

class ZeroFlood(RawGeoFMDataset):
    def __init__(
        self,
        split: str,
        dataset_name: str,
        multi_modal: bool,
        multi_temporal: int,
        root_path: str,
        classes: list,
        num_classes: int,
        ignore_index: int,
        img_size: int,
        bands: dict[str, list[str]],
        distribution: list[int],
        data_mean: dict[str, list[str]],
        data_std: dict[str, list[str]],
        data_min: dict[str, list[str]],
        data_max: dict[str, list[str]],
        download_url: str,
        auto_download: bool,
        # ADDED 
        constant_scale: int,
    ):
        super(ZeroFlood, self).__init__(
            split=split,
            dataset_name=dataset_name,
            multi_modal=multi_modal,
            multi_temporal=multi_temporal,
            root_path=root_path,
            classes=classes,
            num_classes=num_classes,
            ignore_index=ignore_index,
            img_size=img_size,
            bands=bands,
            distribution=distribution,
            data_mean=data_mean,
            data_std=data_std,
            data_min=data_min,
            data_max=data_max,
            download_url=download_url,
            auto_download=auto_download,
        )

        self.root_path = Path(root_path)
        self.classes = classes
        self.split = split
        self.bands = bands
        self.constant_scale = constant_scale

        # Initialize file lists or data structures here
        if self.split in ['train', 'val']:
            tmp_split_folder = 'train'
            valid_files = pd.read_csv(
                self.root_path / 'metadata' / f"info_train.csv", header=0, delimiter='\t'
                ).iloc[:,0].to_list()
            val_len = len(valid_files)//5       # Train:Val = 8:2
            train_len = len(valid_files) - val_len
            if self.split == 'train':
                self.valid_files = valid_files[:train_len]
            else:
                self.valid_files = valid_files[-val_len:]
        else:
            tmp_split_folder = 'val'
            self.valid_files = pd.read_csv(
                self.root_path / 'metadata' / f"info_val.csv", header=0, delimiter='\t'
                ).iloc[:,0].to_list()

        if 'sar' in self.bands.keys():
            img_modality = 'S1RTC'
        elif 'optical' in self.bands.keys():
            img_modality = 'S2L2A'
        else:
            raise ValueError(f"Not supported bands: {self.bands}")

        self.image_list = [str(self.root_path / tmp_split_folder / img_modality / f"{f}.zarr.zip") for f in self.valid_files]
        self.target_list = [str(self.root_path / tmp_split_folder / 'MASK' / f"{f}.zarr.zip") for f in self.valid_files]

    def __len__(self):
        # Return the total number of samples
        return len(self.image_list)

    def _load_file(self, path:Path):

        store = zarr.storage.ZipStore(path, mode="r")
        ds = xr.open_zarr(store, consolidated=True)
        data = ds['bands'].values

        return data
    
    def __getitem__(self, index):
        """Returns the i-th item of the dataset.

        Args:
            i (int): index of the item

        Raises:
            NotImplementedError: raise if the method is not implemented

        Returns:
            dict[str, torch.Tensor | dict[str, torch.Tensor]]: output dictionary follwing the format
            {"image":
                {
                "optical": torch.Tensor of shape (C T H W) (where T=1 if single-temporal dataset),
                "sar": torch.Tensor of shape (C T H W) (where T=1 if single-temporal dataset),
                },
            "target": torch.Tensor of shape (H W) of type torch.int64 for segmentation, torch.float for
            regression datasets.,
            "metadata": dict}.
        """
        # Load your data and labels here
        image = self._load_file(self.image_list[index]) # CHW
        target = self._load_file(self.target_list[index])[0] # HW

        # Convert to tensors
        image = torch.tensor(image, dtype=torch.float32) * self.constant_scale
        image = image.unsqueeze(1)  # (C, H, W) -> (C, 1, H, W)
        target = torch.tensor(target, dtype=torch.long)
        return {
            'image': {list(self.bands.keys())[0] : image},
            'target': target,
            'metadata': {
                'filename': self.image_list[index]
            }
        }

    @staticmethod
    def download(self, silent=False):
        # Implement if your dataset requires downloading
        pass
