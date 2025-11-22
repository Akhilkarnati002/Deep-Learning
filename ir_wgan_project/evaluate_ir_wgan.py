#!/usr/bin/env python3
import argparse
from pathlib import Path
import math

import numpy as np
from PIL import Image

import torch
from torchvision import transforms

from skimage.metrics import structural_similarity as ssim

from ir_wgan_project.train_ir_wgan_deprecated import ResNetGenerator, denorm, pseudo_low_from_high


def parse_args():
    p = argparse.ArgumentParser("Evaluate IR WGAN (consistency PSNR/SSIM)")
    p.add_argument("--ckpt", type=str, required=True,
                   help="Checkpoint path, e.g. runs/train/.../ir_wgan_epoch_100.pth")
    p.add_argument("--low_dir", type=str, required=True,
                   help="Folder with low-res images for evaluation (e.g. ./dataset/low_val)")
    p.add_argument("--image_size", type=int, default=256)
    return p.parse_args()


def tensor_preprocess(image_size):
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])


def psnr(img1, img2):
    """img1, img2: numpy arrays in [0,1]"""
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float("inf")
    return 10 * math.log10(1.0 / mse)


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    # --- load generator ---
    ckpt = torch.load(args.ckpt, map_location=device)
    G = ResNetGenerator(in_channels=1, out_channels=1, num_filters=64, num_blocks=6).to(device)
    G.load_state_dict(ckpt["G"])
    G.eval()

    # --- dataset ---
    low_dir = Path(args.low_dir)
    low_paths = sorted([
        p for p in low_dir.iterdir()
        if p.suffix.lower() in [".png", ".jpg", ".jpeg", ".tif", ".tiff"]
    ])
    if len(low_paths) == 0:
        raise RuntimeError(f"No images in {low_dir}")

    transform = tensor_preprocess(args.image_size)

    psnr_list = []
    ssim_list = []

    for path in low_paths:
        img = Image.open(path).convert("L")
        x = transform(img).unsqueeze(0).to(device)   # [1,1,H,W]

        with torch.no_grad():
            y_fake = G(x)                            # high prediction [-1,1]
            y_low_hat = pseudo_low_from_high(y_fake, scale=2)  # simulated low
            x_dn = denorm(x).clamp(0, 1)            # [0,1]
            y_dn = denorm(y_low_hat).clamp(0, 1)

        x_np = x_dn.squeeze().cpu().numpy()
        y_np = y_dn.squeeze().cpu().numpy()

        psnr_val = psnr(x_np, y_np)
        ssim_val = ssim(x_np, y_np, data_range=1.0)

        psnr_list.append(psnr_val)
        ssim_list.append(ssim_val)

    print(f"Evaluated on {len(low_paths)} images from {low_dir}")
    print(f"Mean PSNR (pseudo-low vs real low):  {np.mean(psnr_list):.2f} dB")
    print(f"Mean SSIM (pseudo-low vs real low):  {np.mean(ssim_list):.4f}")


if __name__ == "__main__":
    main()
