import os
import random
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
from  .transformers import transform_pipeline



class IRImageDataset(Dataset):
    """
    Dataset class for loading Low_Resolution and High_Resolution IR images
    Supports paired and unpaired datasets .
    """

    def __init__(self, low_dir, high_dir, transform= transform_pipeline, paired = False):
        allowed_exts = {'png', 'jpg', 'jpeg', 'tif', 'tiff'}
        self.low_paths = sorted([p for p in Path(low_dir).glob('*') if p.suffix.lower().lstrip('.') in allowed_exts])
        self.high_paths = sorted([p for p in Path(high_dir).glob('*') if p.suffix.lower().lstrip('.') in allowed_exts])
        assert len(self.low_paths) > 0 and len(self.high_paths) > 0, "Data folders must not be empty"

        self.transform = transform
        self.paired = paired


    def __len__(self):
        return max(len(self.low_paths), len(self.high_paths))

    def __getitem__(self, index):
        # Load Low-Resolution Image
        low_path = self.low_paths[index % len(self.low_paths)]    
        low_img = Image.open(low_path).convert('L')     # single channel IR

        if self.paired:
            # assume filenames match or same index
            high_path = self.high_paths[index % len(self.high_paths)]
        else:
            # unpaired mode: pick random high-res image
            high_path = random.choice(self.high_paths)
        high_img = Image.open(high_path).convert('L')   # single channel IR

        # Applying transformers
        low_tensor = self.transform(low_img)
        high_tensor = self.transform(high_img)

        return {
            'low_res': low_tensor,
            'high_res': high_tensor,
            'low_path': str(low_path),
            'high_path': str(high_path)
        }