import os
import argparse
import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from datetime import datetime
from torchvision.utils import make_grid
import torch.nn.functional as F
from Utils.dataset import IRImageDataset
from Utils.transformers import transform_pipeline
from Models.CUT import CUTModel
import sys
import matplotlib.pyplot as plt
import math

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def compute_psnr(img1, img2):
    """Compute PSNR given two normalized tensors in range [-1,1]."""
    img1 = (img1 + 1) / 2
    img2 = (img2 + 1) / 2

    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100  # perfect score
    return 10 * torch.log10(1 / mse)


class CUTTrainer:
    """Trainer class for the CUT model."""

    def __init__(self, config):
        # Device Selection
        self.device = torch.device("cpu") if config["training"].get("use_cpu", False) \
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load Dataset
        dataset = IRImageDataset(
            low_dir=config["data"]["low_res_dir"],
            high_dir=config["data"]["high_res_dir"],
            transform=transform_pipeline,
            paired=config["data"].get("paired", False)
        )

        self.dataloader = DataLoader(
            dataset,
            batch_size=config["training"]["batch_size"],
            shuffle=True,
            num_workers=config["training"].get("num_workers", 0)
        )

        # Initialize CUT Model
        print("DEBUG: Initializing CUTModel...")
        self.model = CUTModel(config)
        self.model.device = self.device
        if hasattr(self.model, "move_networks_to_device"):
            self.model.move_networks_to_device()
        else:
            raise RuntimeError("CUTModel missing move_networks_to_device() method")
        print("DEBUG: CUTModel initialized successfully.")

        self.num_epochs = config["training"]["num_epochs"]

        # Results folder setup
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.results_dir = os.path.join("results", f"CUTModel_{timestamp}")
        os.makedirs(self.results_dir, exist_ok=True)

        # Subfolders
        self.fake_B_dir = os.path.join(self.results_dir, "fake_B")
        self.real_A_dir = os.path.join(self.results_dir, "real_A")
        self.real_B_dir = os.path.join(self.results_dir, "real_B")
        os.makedirs(self.fake_B_dir, exist_ok=True)
        os.makedirs(self.real_A_dir, exist_ok=True)
        os.makedirs(self.real_B_dir, exist_ok=True)

        # PSNR storage
        self.psnr_history = []


    def save_triplet(self, epoch, batch_idx):
        visuals = self.model.get_current_visuals()

        real_A = visuals.get("real_A")
        fake_B = visuals.get("fake_B")
        real_B = visuals.get("real_B")

        if real_A is None or fake_B is None or real_B is None:
            return

        # Normalize from [-1,1] to [0,1]
        real_A = (real_A + 1) / 2
        fake_B = (fake_B + 1) / 2
        real_B = (real_B + 1) / 2

        # 🔥 Resize real_A to match fake_B resolution
        real_A_up = F.interpolate(real_A, size=fake_B.shape[2:], mode='bilinear', align_corners=False)
        real_B_up = F.interpolate(real_B, size=fake_B.shape[2:], mode='bilinear', align_corners=False)

        triplet = torch.cat([real_A_up, fake_B, real_B_up], dim=3)

        out_path = os.path.join(self.results_dir, f"triplet_epoch{epoch+1}.png")
        save_image(triplet, out_path)
        print(f"Saved triplet → {out_path}")


    # --------------------------------------------------------------------
    # Save generated images
    # --------------------------------------------------------------------
    def save_generated_images(self, epoch, batch_idx):
        visuals = self.model.get_current_visuals()
        for label, tensor in visuals.items():
            folder = self.real_A_dir if label == "real_A" else self.real_B_dir if label == "real_B" else self.fake_B_dir
            for i in range(tensor.size(0)):
                img_filename = f"{label}_epoch{epoch+1}_batch{batch_idx+1}_img{i+1}.png"
                img_path = os.path.join(folder, img_filename)
                save_image((tensor[i] + 1) / 2, img_path)


    # --------------------------------------------------------------------
    # Training Loop
    # --------------------------------------------------------------------
    def train(self):
        print("DEBUG: Starting training loop...")
        for epoch in range(self.num_epochs):
            epoch_psnr = []

            print(f"Epoch [{epoch+1}/{self.num_epochs}]")

            for batch_idx, batch in enumerate(self.dataloader):
                low_res_tensor = batch.get("low_res")
                high_res_tensor = batch.get("high_res")

                if low_res_tensor is None or high_res_tensor is None or low_res_tensor.size(0) == 0:
                    continue

                low = low_res_tensor.to(self.device)
                high = high_res_tensor.to(self.device)

                input_dict = {
                    "A": low,
                    "B": high,
                    "A_paths": batch.get("low_path"),
                    "B_paths": batch.get("high_path")
                }
                self.model.set_input(input_dict)

                # random noise robustness
                batch['high_res'] = batch['high_res'] + 0.02 * torch.randn_like(batch['high_res'])
                batch['low_res'] = batch['low_res'] + 0.02 * torch.randn_like(batch['low_res'])

                self.model.optimize_parameters()

                # 🔥 PSNR calculation
                visuals = self.model.get_current_visuals()
                if visuals.get("fake_B") is not None and visuals.get("real_B") is not None:
                    psnr_val = compute_psnr(visuals["fake_B"], visuals["real_B"])
                    epoch_psnr.append(psnr_val.item())

                # Save final batch triplet only
                if batch_idx == len(self.dataloader) - 1:
                    self.save_triplet(epoch, batch_idx)

            # Average PSNR for epoch
            avg_psnr = sum(epoch_psnr) / len(epoch_psnr)
            self.psnr_history.append(avg_psnr)

            # Log losses
            losses = self.model.get_current_losses()
            g_loss = losses.get("G", losses.get("G_GAN", 0.0))
            d_loss = losses.get("D_real", 0.0) + losses.get("D_fake", 0.0)
            nce_loss = losses.get("NCE", losses.get("G_idt", 0.0))

            print(f"Epoch [{epoch+1}/{self.num_epochs}] "
                  f"G_loss: {g_loss:.4f}, D_loss: {d_loss:.4f}, NCE/IDT_loss: {nce_loss:.4f}, PSNR: {avg_psnr:.2f} dB")

        # Save PSNR log + graph
        self.save_psnr_results()
        print(f"DEBUG: Training complete. Results saved in {self.results_dir}")


    # --------------------------------------------------------------------
    # Save PSNR stats + Plot
    # --------------------------------------------------------------------
    def save_psnr_results(self):
        file_path = os.path.join(self.results_dir, "PSNR_log.txt")
        with open(file_path, "w") as f:
            for i, val in enumerate(self.psnr_history):
                f.write(f"Epoch {i+1}: {val:.2f} dB\n")

        plt.figure(figsize=(8,4))
        plt.plot(self.psnr_history, marker='o')
        plt.title("PSNR over Epochs")
        plt.xlabel("Epoch")
        plt.ylabel("PSNR (dB)")
        plt.grid(True)
        plt.savefig(os.path.join(self.results_dir, "PSNR_curve.png"))
        plt.close()
        print("📈 PSNR graph saved!")


# --------------------------------------------------------------------
# Entry point  
# --------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CUT model")
    parser.add_argument("--low", default="/zhome/e5/7/219270/Deep-Learning/Data/Cropped_dataset/Low_res_cropped_specimen", help="Low resolution data folder")
    parser.add_argument("--high", default="/zhome/e5/7/219270/Deep-Learning/Data/Cropped_dataset/High_res_cropped_specimen", help="High resolution data folder")
    parser.add_argument("--paired", action="store_true", help="Use paired dataset")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=0.0002)
    parser.add_argument("--num-epochs", type=int, default=50)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--cpu", action="store_true", help="Force CPU usage")
    parser.add_argument("--no-nce", action="store_true", help="Disable PatchNCE loss")
    parser.add_argument("--dry-run", action="store_true", help="Only initialize trainer, don't train")

    args = parser.parse_args()

    config = {
    "data": { "low_res_dir": args.low, "high_res_dir": args.high, "paired": args.paired },
    "training": {
        "batch_size": args.batch_size,
        "lr": 8e-5,
        "lr_D": 8e-6,
        "num_epochs": 150,
        "lambda_idt": 10.0,
        "lambda_L1": 5.0,
    },
    "cut": {
        "lambda_NCE": 1.0,
        "nce_T": 0.07,
        "num_patches": 192,
        "nce_layers": "4,8,12,16"
    },
    "input_nc": 1, "output_nc": 1, "lambda_FM": 2.0
    }

    print("Dataset paths:")
    print("  Low-res:", config["data"]["low_res_dir"])
    print("  High-res:", config["data"]["high_res_dir"])
    print(f"PatchNCE lambda: {config['cut']['lambda_NCE']} ({'DISABLED' if args.no_nce else 'ENABLED'})")

    if args.dry_run:
        print("Dry run complete. Trainer initialized successfully.")
    else:
        trainer = CUTTrainer(config)
        trainer.train()
