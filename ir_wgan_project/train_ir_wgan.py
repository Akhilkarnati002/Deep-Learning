#!/usr/bin/env python3
"""
Unpaired IR Image Translation: Low-res -> High-res (WGAN-GP + Downsampling Consistency)

- Unpaired dataset: low_dir and high_dir are separate folders.
- Generator: ResNet-style (similar to CycleGAN/CUT), with CoordConv (IR + x + y).
- Discriminator: Multi-scale PatchGAN critic with WGAN-GP.
- Losses:
    * WGAN-GP adversarial loss on high domain
    * Downsampling consistency: pseudo_low(G(low)) ~ low
    * Gradient consistency on pseudo-low vs low
    * Identity loss: G(high) ~ high

Usage example (on HPC):

    python3 train_ir_wgan.py \
        --low_dir dataset/low_cropped \
        --high_dir dataset/high_cropped \
        --epochs 100 \
        --batch_size 4 \
        --out_dir runs/train/exp1/checkpoints \
        --sample_dir runs/train/exp1/samples
"""

import argparse
import os
import random
from pathlib import Path

import numpy as np
from PIL import Image

import torch
from torch import nn, autograd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import save_image
import torch.nn.functional as F

# Optional: for color-mapped sample images (inferno, etc.)
try:
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


# -----------------------------
# Dataset
# -----------------------------

class UnpairedIRDataset(Dataset):
    """
    Unpaired IR dataset:
      - low_dir: low-res IR images (cropped/processed)
      - high_dir: high-res IR images (cropped/processed)

    At each __getitem__, returns one low image and one high image that are NOT a true pair.
    """

    def __init__(
        self,
        low_dir,
        high_dir,
        image_size=256,
        augment=True,
    ):
        self.low_dir = Path(low_dir)
        self.high_dir = Path(high_dir)

        self.low_paths = sorted(
            [p for p in self.low_dir.iterdir()
             if p.suffix.lower() in [".png", ".jpg", ".jpeg", ".tif", ".tiff"]]
        )
        self.high_paths = sorted(
            [p for p in self.high_dir.iterdir()
             if p.suffix.lower() in [".png", ".jpg", ".jpeg", ".tif", ".tiff"]]
        )

        if len(self.low_paths) == 0:
            raise RuntimeError(f"No images found in low_dir: {low_dir}")
        if len(self.high_paths) == 0:
            raise RuntimeError(f"No images found in high_dir: {high_dir}")

        base_transforms = [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),                  # (1, H, W) for 'L' images
            transforms.Normalize([0.5], [0.5]),     # to [-1, 1]
        ]

        if augment:
            self.transform = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.Grayscale(num_output_channels=1),
                *base_transforms,
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Grayscale(num_output_channels=1),
                *base_transforms,
            ])

    def __len__(self):
        # Use max so we keep cycling through smaller domain
        return max(len(self.low_paths), len(self.high_paths))

    def __getitem__(self, idx):
        low_path = self.low_paths[idx % len(self.low_paths)]
        high_path = random.choice(self.high_paths)  # unpaired: random pick from high domain

        low_img = Image.open(low_path).convert("L")
        high_img = Image.open(high_path).convert("L")

        low_tensor = self.transform(low_img)
        high_tensor = self.transform(high_img)

        return {
            "low": low_tensor,
            "high": high_tensor,
            "low_path": str(low_path),
            "high_path": str(high_path),
        }


# -----------------------------
# CoordConv helper
# -----------------------------

def add_coords(x: torch.Tensor) -> torch.Tensor:
    """
    Add normalized (x, y) coordinate channels to input.
    x: (B, 1, H, W) -> (B, 3, H, W)
    """
    B, _, H, W = x.shape
    yy, xx = torch.meshgrid(
        torch.linspace(-1, 1, H, device=x.device),
        torch.linspace(-1, 1, W, device=x.device),
        indexing="ij",
    )
    xx = xx.expand(B, 1, H, W)
    yy = yy.expand(B, 1, H, W)
    return torch.cat([x, xx, yy], dim=1)


# -----------------------------
# Model: Generator (ResNet)
# -----------------------------

class ResnetBlock(nn.Module):
    """Standard ResNet block with two 3x3 convs and instance norm."""

    def __init__(self, dim):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3, padding=0, bias=False),
            nn.InstanceNorm2d(dim),
            nn.ReLU(inplace=True),

            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3, padding=0, bias=False),
            nn.InstanceNorm2d(dim),
        )

    def forward(self, x):
        return x + self.conv_block(x)


class ResNetGenerator(nn.Module):
    """
    ResNet-based generator (similar to CycleGAN/CUT) for IR images.

    Input:  (B, 3, H, W)  -> IR + x + y
    Output: (B, 1, H, W) in [-1, 1]
    """

    def __init__(self, in_channels=3, out_channels=1, num_filters=64, num_blocks=6):
        super().__init__()

        # Initial 7x7 conv
        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels, num_filters, kernel_size=7, padding=0, bias=False),
            nn.InstanceNorm2d(num_filters),
            nn.ReLU(inplace=True),
        ]

        # Downsampling (x2, x4)
        in_f = num_filters
        out_f = in_f * 2
        for _ in range(2):
            model += [
                nn.Conv2d(in_f, out_f, kernel_size=3, stride=2, padding=1, bias=False),
                nn.InstanceNorm2d(out_f),
                nn.ReLU(inplace=True),
            ]
            in_f = out_f
            out_f = in_f * 2

        # ResNet blocks
        for _ in range(num_blocks):
            model += [ResnetBlock(in_f)]

        # Upsampling (x2, x4)
        out_f = in_f // 2
        for _ in range(2):
            model += [
                nn.ConvTranspose2d(in_f, out_f, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
                nn.InstanceNorm2d(out_f),
                nn.ReLU(inplace=True),
            ]
            in_f = out_f
            out_f = in_f // 2

        # Final 7x7 conv
        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_f, out_channels, kernel_size=7, padding=0),
            nn.Tanh(),  # output in [-1, 1]
        ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


# -----------------------------
# Model: Discriminators
# -----------------------------

class PatchGANDiscriminator(nn.Module):
    """
    PatchGAN-style critic for WGAN-GP.

    Input: (B, 1, H, W) IR image
    Output: (B, 1) scalar per image (mean over patch map).
    """

    def __init__(self, in_channels=1, num_filters=64):
        super().__init__()

        def conv_block(in_c, out_c, stride, norm=True):
            layers = [
                nn.Conv2d(in_c, out_c, kernel_size=4, stride=stride, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
            ]
            if norm:
                layers.append(nn.InstanceNorm2d(out_c))
            return layers

        layers = []
        # C64
        layers += conv_block(in_channels, num_filters, stride=2, norm=False)
        # C128
        layers += conv_block(num_filters, num_filters * 2, stride=2)
        # C256
        layers += conv_block(num_filters * 2, num_filters * 4, stride=2)
        # C512
        layers += conv_block(num_filters * 4, num_filters * 8, stride=1)
        # output 1-channel patch map
        layers += [nn.Conv2d(num_filters * 8, 1, kernel_size=4, stride=1, padding=1)]

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        out = self.model(x)       # (B, 1, H', W')
        return out.view(out.size(0), -1).mean(dim=1, keepdim=True)  # (B, 1)


class MultiScaleDiscriminator(nn.Module):
    """
    Multi-scale PatchGAN critic:
      D0: full resolution
      D1: downsampled by 2

    Returns a list of scores, one per scale.
    """

    def __init__(self, in_channels=1, num_filters=64, num_D=2):
        super().__init__()
        self.num_D = num_D
        self.discs = nn.ModuleList(
            [PatchGANDiscriminator(in_channels, num_filters) for _ in range(num_D)]
        )

    def forward(self, x):
        outputs = []
        cur = x
        for D in self.discs:
            outputs.append(D(cur))
            # downsample for next scale
            cur = F.avg_pool2d(cur, kernel_size=2, stride=2)
        return outputs


# -----------------------------
# WGAN-GP utilities
# -----------------------------

def gradient_penalty(critic, real, fake, device):
    """Compute WGAN-GP gradient penalty (on highest scale)."""
    batch_size = real.size(0)
    epsilon = torch.rand(batch_size, 1, 1, 1, device=device)
    interpolated = epsilon * real + (1.0 - epsilon) * fake
    interpolated.requires_grad_(True)

    mixed_scores = critic(interpolated)
    # If multi-scale, take highest-res scale
    if isinstance(mixed_scores, (list, tuple)):
        mixed_scores = mixed_scores[0]

    grad_outputs = torch.ones_like(mixed_scores, device=device)
    gradients = autograd.grad(
        outputs=mixed_scores,
        inputs=interpolated,
        grad_outputs=grad_outputs,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    gradients = gradients.view(batch_size, -1)
    gradient_norm = gradients.norm(2, dim=1)
    gp = ((gradient_norm - 1.0) ** 2).mean()
    return gp


def gaussian_blur(x, kernel_size=7, sigma=1.5):
    """Depthwise Gaussian blur."""
    coords = torch.arange(kernel_size, device=x.device) - (kernel_size - 1) / 2
    g = torch.exp(-0.5 * (coords / sigma) ** 2)
    g = g / g.sum()
    g2d = g[:, None] * g[None, :]
    g2d = g2d / g2d.sum()
    g2d = g2d.view(1, 1, kernel_size, kernel_size)
    g2d = g2d.repeat(x.shape[1], 1, 1, 1)  # depthwise
    return F.conv2d(x, g2d, padding=kernel_size // 2, groups=x.shape[1])


def pseudo_low_from_high(high, scale=2):
    """
    Approximate the low-res camera by:
      1) Gaussian blur
      2) average pooling with stride=scale
      3) upsampling back to original size (bilinear)
    """
    if scale <= 1:
        return high
    high_blur = gaussian_blur(high, kernel_size=9, sigma=2.0)
    pooled = F.avg_pool2d(high_blur, kernel_size=scale, stride=scale)
    up = F.interpolate(pooled, size=high.shape[-2:], mode="bilinear", align_corners=False)
    return up


def gradient_xy(x):
    """Simple finite-difference gradients in x and y."""
    dx = x[:, :, :, 1:] - x[:, :, :, :-1]
    dy = x[:, :, 1:, :] - x[:, :, :-1, :]
    return dx, dy


def denorm(x):
    """[-1, 1] -> [0, 1]"""
    return (x + 1.0) / 2.0


# -----------------------------
# Sample saving
# -----------------------------

def save_samples(low, fake_high, real_high, step, sample_dir, max_n=4):
    """Save grayscale triplets (low, fake_high, real_high) for quick inspection."""
    os.makedirs(sample_dir, exist_ok=True)

    n = min(max_n, low.size(0))
    low = denorm(low[:n])
    fake = denorm(fake_high[:n])
    real = denorm(real_high[:n])

    # Stack as [low1..n, fake1..n, real1..n]; 3*n images, n columns
    grid = torch.cat([low, fake, real], dim=0)
    path = os.path.join(sample_dir, f"step_{step:06d}_triplets.png")
    save_image(grid, path, nrow=n)

    # Optional: color-mapped version of fake_high[0]
    if HAS_MPL:
        y = fake_high[0:1]  # (1, 1, H, W)
        y_np = denorm(y).squeeze().detach().cpu().numpy()
        y_np = np.clip(y_np, 0.0, 1.0)
        cm_path = os.path.join(sample_dir, f"step_{step:06d}_fake_colormap.png")
        plt.imsave(cm_path, y_np, cmap="inferno")


# -----------------------------
# Training
# -----------------------------

def train(
    low_dir,
    high_dir,
    image_size=256,
    epochs=100,
    batch_size=4,
    lr=1e-4,
    betas=(0.5, 0.999),
    lambda_gp=10.0,
    lambda_recon=20.0,
    lambda_id=2.0,
    lambda_grad=5.0,
    downscale_factor=2,
    n_critic=5,
    out_dir="checkpoints",
    sample_dir="samples",
    num_workers=2,
    log_interval=20,
    sample_interval=200,
    resume=None,
    seed=42,
    device=None,
):
    # Device & seeds
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True

    # Dirs
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(sample_dir, exist_ok=True)

    # Dataset & dataloader
    dataset = UnpairedIRDataset(
        low_dir=low_dir,
        high_dir=high_dir,
        image_size=image_size,
        augment=True,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device == "cuda"),
        drop_last=True,
    )

    # Models
    G = ResNetGenerator(in_channels=3, out_channels=1, num_filters=64, num_blocks=6).to(device)
    D = MultiScaleDiscriminator(in_channels=1, num_filters=64, num_D=2).to(device)

    # Optimizers
    opt_G = torch.optim.Adam(G.parameters(), lr=lr, betas=betas)
    opt_D = torch.optim.Adam(D.parameters(), lr=lr, betas=betas)

    # Optionally resume
    start_epoch = 1
    global_step = 0
    if resume is not None and os.path.isfile(resume):
        print(f"Resuming from checkpoint: {resume}")
        ckpt = torch.load(resume, map_location=device)
        G.load_state_dict(ckpt["G"])
        D.load_state_dict(ckpt["D"])
        opt_G.load_state_dict(ckpt["opt_G"])
        opt_D.load_state_dict(ckpt["opt_D"])
        start_epoch = ckpt.get("epoch", 1)
        global_step = ckpt.get("global_step", 0)
        print(f"Resumed at epoch {start_epoch}, global_step {global_step}")

    l1 = nn.L1Loss()

    # Training loop
    for epoch in range(start_epoch, epochs + 1):
        # LR decay after 50 epochs
        if epoch == 51:
            for param_group in opt_G.param_groups:
                param_group["lr"] = 5e-5
            for param_group in opt_D.param_groups:
                param_group["lr"] = 5e-5

        for batch in dataloader:
            low = batch["low"].to(device)          # domain L (low-res)
            real_high = batch["high"].to(device)   # domain H (high-res)

            # CoordConv inputs
            low_coord = add_coords(low)
            real_high_coord = add_coords(real_high)

            # ---------------------------
            # Train critic D (n_critic steps)
            # ---------------------------
            for _ in range(n_critic):
                fake_high = G(low_coord).detach()

                real_list = D(real_high)
                fake_list = D(fake_high)

                # WGAN-GP loss across scales
                loss_D_adv = 0.0
                for sr, sf in zip(real_list, fake_list):
                    loss_D_adv += (sf.mean() - sr.mean())
                loss_D_adv /= len(real_list)

                gp = gradient_penalty(D, real_high, fake_high, device=device)
                loss_D = loss_D_adv + lambda_gp * gp

                opt_D.zero_grad()
                loss_D.backward()
                opt_D.step()

            # ---------------------------
            # Train generator G
            # ---------------------------
            fake_high = G(low_coord)
            fake_list = D(fake_high)

            adv_loss = 0.0
            for sf in fake_list:
                adv_loss += -sf.mean()
            adv_loss /= len(fake_list)

            # Downsampling consistency: approximate low-res camera
            pseudo_low = pseudo_low_from_high(fake_high, scale=downscale_factor)
            recon_loss = l1(pseudo_low, low)

            # Gradient consistency on pseudo-low vs low
            dx_fake, dy_fake = gradient_xy(pseudo_low)
            dx_low, dy_low = gradient_xy(low)
            grad_loss = l1(dx_fake, dx_low) + l1(dy_fake, dy_low)

            # Identity loss: G(high) ~ high
            id_high = G(real_high_coord)
            id_loss = l1(id_high, real_high)

            loss_G = (
                adv_loss
                + lambda_recon * recon_loss
                + lambda_id * id_loss
                + lambda_grad * grad_loss
            )

            opt_G.zero_grad()
            loss_G.backward()
            opt_G.step()

            # ---------------------------
            # Logging & samples
            # ---------------------------
            if global_step % log_interval == 0:
                print(
                    f"Epoch [{epoch}/{epochs}] Step {global_step} "
                    f"Loss_D: {loss_D.item():.4f} "
                    f"Loss_G: {loss_G.item():.4f} "
                    f"(adv: {adv_loss.item():.4f}, "
                    f"recon: {recon_loss.item():.4f}, "
                    f"id: {id_loss.item():.4f}, "
                    f"grad: {grad_loss.item():.4f})"
                )

            if global_step % sample_interval == 0:
                with torch.no_grad():
                    low_vis = low[:4]
                    low_vis_coord = add_coords(low_vis)
                    fake_vis = G(low_vis_coord)
                    real_vis = real_high[:4]
                    save_samples(low_vis, fake_vis, real_vis, global_step, sample_dir)

            global_step += 1

        # Save checkpoint each epoch
        ckpt_path = os.path.join(out_dir, f"ir_wgan_epoch_{epoch:03d}.pth")
        torch.save(
            {
                "epoch": epoch,
                "global_step": global_step,
                "G": G.state_dict(),
                "D": D.state_dict(),
                "opt_G": opt_G.state_dict(),
                "opt_D": opt_D.state_dict(),
                "low_dir": str(low_dir),
                "high_dir": str(high_dir),
                "image_size": image_size,
            },
            ckpt_path,
        )
        print(f"Saved checkpoint: {ckpt_path}")


# -----------------------------
# CLI
# -----------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Unpaired IR WGAN-GP (low -> high) with CoordConv and multi-scale discriminator"
    )
    parser.add_argument("--low_dir", type=str, required=True,
                        help="Folder with preprocessed low-res IR images")
    parser.add_argument("--high_dir", type=str, required=True,
                        help="Folder with preprocessed high-res IR images")
    parser.add_argument("--image_size", type=int, default=256,
                        help="Resize all images to this size (square)")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--beta1", type=float, default=0.5)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--lambda_gp", type=float, default=10.0,
                        help="Gradient penalty weight")
    parser.add_argument("--lambda_recon", type=float, default=20.0,
                        help="Downsampling consistency weight")
    parser.add_argument("--lambda_id", type=float, default=2.0,
                        help="Identity loss weight")
    parser.add_argument("--lambda_grad", type=float, default=5.0,
                        help="Gradient consistency loss weight")
    parser.add_argument("--downscale_factor", type=int, default=2,
                        help="Pseudo low-res downscale factor")
    parser.add_argument("--n_critic", type=int, default=5,
                        help="Critic steps per generator step")
    parser.add_argument("--out_dir", type=str, default="checkpoints",
                        help="Directory to save checkpoints")
    parser.add_argument("--sample_dir", type=str, default="samples",
                        help="Directory to save generated samples")
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--log_interval", type=int, default=20)
    parser.add_argument("--sample_interval", type=int, default=200)
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    train(
        low_dir=args.low_dir,
        high_dir=args.high_dir,
        image_size=args.image_size,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        betas=(args.beta1, args.beta2),
        lambda_gp=args.lambda_gp,
        lambda_recon=args.lambda_recon,
        lambda_id=args.lambda_id,
        lambda_grad=args.lambda_grad,
        downscale_factor=args.downscale_factor,
        n_critic=args.n_critic,
        out_dir=args.out_dir,
        sample_dir=args.sample_dir,
        num_workers=args.num_workers,
        log_interval=args.log_interval,
        sample_interval=args.sample_interval,
        resume=args.resume,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
