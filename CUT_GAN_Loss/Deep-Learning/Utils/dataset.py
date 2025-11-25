import os
import random
from pathlib import Path
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from .transformers import transform_pipeline


def load_ir_image(path):
    """
    Safely load TIFF / PNG / JPG infrared images.
    Automatically handles:
    - 8-bit images
    - 16-bit thermal TIFF images
    - multi-frame TIFFs (use first frame)
    - normalization
    """

    img = Image.open(path)

    # If TIFF with multiple frames, pick the first
    try:
        img.seek(0)
    except:
        pass

    arr = np.array(img)

    # If it's 16-bit, normalize to 0–255
    if arr.dtype == np.uint16 or arr.max() > 255:
        arr = arr.astype(np.float32)
        arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-6)
        arr = (arr * 255).astype(np.uint8)

    # Convert to PIL 8-bit grayscale
    img = Image.fromarray(arr).convert("L")
    return img


class IRImageDataset(Dataset):
    """
    Dataset class for loading Low_Resolution and High_Resolution IR images.
    Supports TIFF, PNG, JPG, etc.
    Handles paired and unpaired modes.
    """

    def __init__(self, low_dir, high_dir, transform=transform_pipeline, paired=False):
        allowed_exts = {"png", "jpg", "jpeg", "tif", "tiff"}

        self.low_paths = sorted([
            p for p in Path(low_dir).glob("*")
            if p.suffix.lower().lstrip(".") in allowed_exts
        ])
        self.high_paths = sorted([
            p for p in Path(high_dir).glob("*")
            if p.suffix.lower().lstrip(".") in allowed_exts
        ])

        assert len(self.low_paths) > 0, "No LR images found!"
        assert len(self.high_paths) > 0, "No HR images found!"

        self.transform = transform
        self.paired = paired

    def __len__(self):
        return max(len(self.low_paths), len(self.high_paths))

    def __getitem__(self, index):
        # Load LR
        low_path = self.low_paths[index % len(self.low_paths)]
        low_img = load_ir_image(low_path)

        # Paired or unpaired HR selection
        if self.paired:
            high_path = self.high_paths[index % len(self.high_paths)]
        else:
            high_path = random.choice(self.high_paths)

        high_img = load_ir_image(high_path)

        # Apply transforms
        low_tensor = self.transform(low_img)
        high_tensor = self.transform(high_img)

        return {
            "low_res": low_tensor,
            "high_res": high_tensor,
            "low_path": str(low_path),
            "high_path": str(high_path),
        }
