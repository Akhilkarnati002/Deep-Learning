import os
import random
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from .transformers import transform_pipeline


class IRImageDataset(Dataset):
    """
    Dataset class for loading Low_Resolution and High_Resolution IR images.
    Supports paired and unpaired datasets.
    """

    def __init__(self, low_dir, high_dir, transform=transform_pipeline, paired=False):
        allowed_exts = {"png", "jpg", "jpeg", "tif", "tiff"}
        self.low_paths = sorted(
            [p for p in Path(low_dir).glob("*") if p.suffix.lower().lstrip(".") in allowed_exts]
        )
        self.high_paths = sorted(
            [p for p in Path(high_dir).glob("*") if p.suffix.lower().lstrip(".") in allowed_exts]
        )
        assert len(self.low_paths) > 0 and len(self.high_paths) > 0, "Data folders must not be empty"

        self.transform = transform
        self.paired = paired

    def __len__(self):
        return max(len(self.low_paths), len(self.high_paths))

    def __getitem__(self, index):
        # Load Low-Resolution Image
        low_path = self.low_paths[index % len(self.low_paths)]
        low_img = Image.open(low_path).convert("RGB")

        if self.paired:
            # assume filenames match or same index
            high_path = self.high_paths[index % len(self.high_paths)]
        else:
            # unpaired mode: pick random high-res image
            high_path = random.choice(self.high_paths)
        high_img = Image.open(high_path).convert("RGB")

        # Applying transformers
        low_tensor = self.transform(low_img)
        high_tensor = self.transform(high_img)

        return {
            "low_res": low_tensor,
            "high_res": high_tensor,
            "low_path": str(low_path),
            "high_path": str(high_path),
        }


class SuperResolutionDataset(Dataset):
    """
    Super-resolution dataset built from high-resolution images only.

    For each HR image:
      - Resize to hr_size (H, W)
      - Downsample by `scale_factor` with bicubic to create LR
      - Upsample LR back to hr_size (standard SR residual setup)
      - Return tensors in [-1, 1]:
            lr: upsampled low-res input
            hr: original high-res target
    """

    def __init__(self, hr_dir, hr_size=(256, 512), scale_factor=4):
        allowed_exts = {"png", "jpg", "jpeg", "tif", "tiff"}
        self.hr_paths = sorted(
            [p for p in Path(hr_dir).glob("*") if p.suffix.lower().lstrip(".") in allowed_exts]
        )
        assert len(self.hr_paths) > 0, "High-resolution data folder must not be empty"

        self.hr_size = hr_size
        self.scale_factor = scale_factor

        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])

    def __len__(self):
        return len(self.hr_paths)

    def __getitem__(self, index):
        hr_path = self.hr_paths[index]
        hr_img = Image.open(hr_path).convert("RGB")

        # Resize HR to fixed size
        hr_img = hr_img.resize(self.hr_size, Image.BICUBIC)

        # Create LR by downsampling then upsampling back to hr_size
        lr_size = (self.hr_size[1] // self.scale_factor, self.hr_size[0] // self.scale_factor)
        lr_small = hr_img.resize(lr_size, Image.BICUBIC)
        lr_up = lr_small.resize(self.hr_size, Image.BICUBIC)

        hr_tensor = self.normalize(self.to_tensor(hr_img))
        lr_tensor = self.normalize(self.to_tensor(lr_up))

        return {
            "lr": lr_tensor,
            "hr": hr_tensor,
            "path": str(hr_path),
        }
