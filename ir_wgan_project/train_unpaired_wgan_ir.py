import argparse
import os
import random
from pathlib import Path

import torch
from torch import nn, autograd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image


# =========================
#       DATASET
# =========================

class UnpairedIRDataset(Dataset):
    """
    Unpaired IR dataset:
    - A domain: low-res images (cropped/processed)
    - B domain: high-res images (cropped/processed)

    At each __getitem__, returns one A image and one B image.
    They are NOT assumed to be pairs.
    """

    def __init__(self, low_dir, high_dir, image_size=256, augment=True):
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

        assert len(self.low_paths) > 0, "No images found in low_dir"
        assert len(self.high_paths) > 0, "No images found in high_dir"

        base_transforms = [
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),  # to [-1, 1]
        ]

        if augment:
            self.transform = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                # You can add more mild augmentations if needed
                *base_transforms,
            ])
        else:
            self.transform = transforms.Compose(base_transforms)

    def __len__(self):
        # Use max so we keep cycling through smaller domain
        return max(len(self.low_paths), len(self.high_paths))

    def __getitem__(self, idx):
        low_path = self.low_paths[idx % len(self.low_paths)]
        high_path = random.choice(self.high_paths)  # unpaired: random pick

        low_img = Image.open(low_path)
        high_img = Image.open(high_path)

        low_tensor = self.transform(low_img)
        high_tensor = self.transform(high_img)

        return {
            "low": low_tensor,
            "high": high_tensor,
            "low_path": str(low_path),
            "high_path": str(high_path),
        }


# =========================
#      MODELS
# =========================

class GeneratorUNet(nn.Module):
    """
    U-Net style generator for 1-channel IR images.
    Maps (B,1,H,W) -> (B,1,H,W), output in [-1,1] via tanh.
    """

    def __init__(self, in_channels=1, out_channels=1, features=64):
        super().__init__()

        self.down1 = self._down_block(in_channels, features, normalize=False)  # 256->128
        self.down2 = self._down_block(features, features * 2)                  # 128->64
        self.down3 = self._down_block(features * 2, features * 4)              # 64->32
        self.down4 = self._down_block(features * 4, features * 8)              # 32->16

        self.bottleneck = nn.Sequential(
            nn.Conv2d(features * 8, features * 8, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )  # 16->8

        self.up1 = self._up_block(features * 8, features * 8)         # 8->16
        self.up2 = self._up_block(features * 8 * 2, features * 4)     # 16->32
        self.up3 = self._up_block(features * 4 * 2, features * 2)     # 32->64
        self.up4 = self._up_block(features * 2 * 2, features)         # 64->128

        self.final_conv = nn.ConvTranspose2d(
            features * 2, out_channels, kernel_size=4, stride=2, padding=1
        )  # 128->256
        self.tanh = nn.Tanh()

    @staticmethod
    def _down_block(in_channels, out_channels, normalize=True):
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=4,
                      stride=2, padding=1, bias=False),
        ]
        if normalize:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        return nn.Sequential(*layers)

    @staticmethod
    def _up_block(in_channels, out_channels, dropout=False):
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4,
                               stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout(0.5))
        return nn.Sequential(*layers)

    def forward(self, x):
        # Encoder
        d1 = self.down1(x)   # 128
        d2 = self.down2(d1)  # 64
        d3 = self.down3(d2)  # 32
        d4 = self.down4(d3)  # 16

        bottleneck = self.bottleneck(d4)  # 8

        # Decoder with skip connections
        u1 = self.up1(bottleneck)         # 16
        u1 = torch.cat([u1, d4], dim=1)
        u2 = self.up2(u1)                 # 32
        u2 = torch.cat([u2, d3], dim=1)
        u3 = self.up3(u2)                 # 64
        u3 = torch.cat([u3, d2], dim=1)
        u4 = self.up4(u3)                 # 128
        u4 = torch.cat([u4, d1], dim=1)

        out = self.final_conv(u4)         # 256
        return self.tanh(out)


class Critic(nn.Module):
    """
    Patch-based critic for WGAN-GP.
    Used separately for low domain and high domain (in_channels=1).
    """

    def __init__(self, in_channels=1, features=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, features, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(features, features * 2, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(features * 2, features * 4, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(features * 4, features * 8, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(features * 8, 1, kernel_size=4, stride=1, padding=0),
        )

    def forward(self, x):
        out = self.net(x)
        return out.view(out.size(0), -1).mean(dim=1, keepdim=True)


# =========================
#   WGAN-GP & UTILITIES
# =========================

def gradient_penalty(critic, real, fake, device):
    batch_size = real.size(0)
    epsilon = torch.rand(batch_size, 1, 1, 1, device=device)
    interpolated = epsilon * real + (1 - epsilon) * fake
    interpolated.requires_grad_(True)

    mixed_scores = critic(interpolated)

    grad_outputs = torch.ones_like(mixed_scores, device=device)
    gradients = autograd.grad(
        inputs=interpolated,
        outputs=mixed_scores,
        grad_outputs=grad_outputs,
        retain_graph=True,
        create_graph=True,
        only_inputs=True,
    )[0]

    gradients = gradients.view(batch_size, -1)
    gradient_norm = gradients.norm(2, dim=1)
    gp = ((gradient_norm - 1.0) ** 2).mean()
    return gp


def denorm(t):
    # [-1,1] -> [0,1]
    return (t + 1) / 2.0


def save_sample(low, fake_high, step, sample_dir, max_n=4):
    os.makedirs(sample_dir, exist_ok=True)
    n = min(max_n, low.size(0))
    grid = torch.cat([denorm(low[:n]), denorm(fake_high[:n])], dim=0)
    save_image(grid, os.path.join(sample_dir, f"low2high_{step:06d}.png"), nrow=n)


# =========================
#       TRAIN LOOP
# =========================

def train(
    low_dir,
    high_dir,
    epochs=200,
    batch_size=4,
    lr=1e-4,
    n_critic=5,
    lambda_gp=10.0,
    lambda_cycle=10.0,
    lambda_id=5.0,
    image_size=256,
    out_dir="checkpoints_unpaired",
    sample_dir="samples_unpaired",
    device=None,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    dataset = UnpairedIRDataset(low_dir, high_dir, image_size=image_size, augment=True)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                            num_workers=2, pin_memory=True)

    # Generators
    G = GeneratorUNet(in_channels=1, out_channels=1).to(device)  # low -> high
    F = GeneratorUNet(in_channels=1, out_channels=1).to(device)  # high -> low

    # Critics
    D_low = Critic(in_channels=1).to(device)
    D_high = Critic(in_channels=1).to(device)

    opt_G = torch.optim.Adam(
        list(G.parameters()) + list(F.parameters()),
        lr=lr, betas=(0.5, 0.999)
    )
    opt_D = torch.optim.Adam(
        list(D_low.parameters()) + list(D_high.parameters()),
        lr=lr, betas=(0.5, 0.999)
    )

    l1 = nn.L1Loss()

    step = 0
    os.makedirs(out_dir, exist_ok=True)

    for epoch in range(1, epochs + 1):
        for batch in dataloader:
            low = batch["low"].to(device)    # domain A
            high = batch["high"].to(device)  # domain B

            # ============================
            #    TRAIN CRITICS (WGAN-GP)
            # ============================
            for _ in range(n_critic):
                fake_high = G(low).detach()
                fake_low = F(high).detach()

                # High-domain critic
                d_real_high = D_high(high)
                d_fake_high = D_high(fake_high)
                gp_high = gradient_penalty(D_high, high, fake_high, device)
                loss_D_high = (d_fake_high.mean() - d_real_high.mean()) + lambda_gp * gp_high

                # Low-domain critic
                d_real_low = D_low(low)
                d_fake_low = D_low(fake_low)
                gp_low = gradient_penalty(D_low, low, fake_low, device)
                loss_D_low = (d_fake_low.mean() - d_real_low.mean()) + lambda_gp * gp_low

                loss_D = (loss_D_high + loss_D_low) / 2.0

                opt_D.zero_grad()
                loss_D.backward()
                opt_D.step()

            # ============================
            #      TRAIN GENERATORS
            # ============================
            fake_high = G(low)
            fake_low = F(high)

            # Adversarial (want critics to think fakes are real)
            adv_G = -D_high(fake_high).mean()
            adv_F = -D_low(fake_low).mean()

            # Cycle consistency
            rec_low = F(fake_high)
            rec_high = G(fake_low)
            loss_cycle_low = l1(rec_low, low)
            loss_cycle_high = l1(rec_high, high)
            loss_cycle = loss_cycle_low + loss_cycle_high

            # Identity (optional but stabilizing)
            id_high = G(high)
            id_low = F(low)
            loss_id = l1(id_high, high) + l1(id_low, low)

            loss_G_total = adv_G + adv_F + lambda_cycle * loss_cycle + lambda_id * loss_id

            opt_G.zero_grad()
            loss_G_total.backward()
            opt_G.step()

            if step % 20 == 0:
                print(
                    f"Epoch [{epoch}/{epochs}] Step {step} "
                    f"Loss_D: {loss_D.item():.4f} "
                    f"Loss_G: {loss_G_total.item():.4f} "
                    f"(adv_G: {adv_G.item():.4f}, adv_F: {adv_F.item():.4f}, "
                    f"cycle: {loss_cycle.item():.4f}, id: {loss_id.item():.4f})"
                )

            if step % 200 == 0:
                with torch.no_grad():
                    low_vis = low[:4]
                    fake_high_vis = G(low_vis)
                    save_sample(low_vis, fake_high_vis, step, sample_dir)

            step += 1

        # Save checkpoint each epoch
        torch.save(
            {
                "epoch": epoch,
                "G_state_dict": G.state_dict(),     # low -> high
                "F_state_dict": F.state_dict(),     # high -> low
                "D_low_state_dict": D_low.state_dict(),
                "D_high_state_dict": D_high.state_dict(),
                "opt_G_state_dict": opt_G.state_dict(),
                "opt_D_state_dict": opt_D.state_dict(),
            },
            os.path.join(out_dir, f"unpaired_wgan_ir_epoch_{epoch:03d}.pth"),
        )


# =========================
#          CLI
# =========================

def main():
    parser = argparse.ArgumentParser(
        description="Unpaired WGAN-GP (Cycle-style) for IR low->high translation"
    )
    parser.add_argument("--low_dir", type=str, required=True,
                        help="Folder with preprocessed low-res images")
    parser.add_argument("--high_dir", type=str, required=True,
                        help="Folder with preprocessed high-res images")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--n_critic", type=int, default=5)
    parser.add_argument("--lambda_gp", type=float, default=10.0)
    parser.add_argument("--lambda_cycle", type=float, default=10.0)
    parser.add_argument("--lambda_id", type=float, default=5.0)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--out_dir", type=str, default="checkpoints_unpaired")
    parser.add_argument("--sample_dir", type=str, default="samples_unpaired")
    args = parser.parse_args()

    torch.manual_seed(42)
    random.seed(42)

    train(
        low_dir=args.low_dir,
        high_dir=args.high_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        n_critic=args.n_critic,
        lambda_gp=args.lambda_gp,
        lambda_cycle=args.lambda_cycle,
        lambda_id=args.lambda_id,
        image_size=args.image_size,
        out_dir=args.out_dir,
        sample_dir=args.sample_dir,
    )


if __name__ == "__main__":
    main()
