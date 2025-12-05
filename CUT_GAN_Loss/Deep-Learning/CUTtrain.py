import os
import argparse
import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from datetime import datetime
import torch.nn.functional as F
import matplotlib.pyplot as plt
from math import log10
from Utils.dataset import IRImageDataset
from Utils.transformers import transform_pipeline
from Models.CUT import CUTModel
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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

        # PSNR logging
        self.psnr_values = []


    #  PSNR COMPUTATION 
    def calculate_psnr(self, fake, real):
        """ Compute PSNR between Fake_B and Real_B """
        mse = torch.mean((fake - real) ** 2)
        if mse == 0:
            return 100
        psnr = 20 * log10(1.0 / torch.sqrt(mse))  # image range normalized [0,1]
        return psnr


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

        # Resize real_A and real_B to match fake_B resolution
        real_A_up = F.interpolate(real_A, size=fake_B.shape[2:], mode='bilinear', align_corners=False)
        real_B_up = F.interpolate(real_B, size=fake_B.shape[2:], mode='bilinear', align_corners=False)

        # Now concatenate
        triplet = torch.cat([real_A_up, fake_B, real_B_up], dim=3)

        out_path = os.path.join(self.results_dir, f"triplet_epoch{epoch+1}.png")
        save_image(triplet, out_path)
        print(f"Saved triplet → {out_path}")


    def save_generated_images(self, epoch, batch_idx):
        visuals = self.model.get_current_visuals()
        for label, tensor in visuals.items():
            folder = self.real_A_dir if label == "real_A" else self.real_B_dir if label == "real_B" else self.fake_B_dir
            for i in range(tensor.size(0)):
                img_filename = f"{label}_epoch{epoch+1}_batch{batch_idx+1}_img{i+1}.png"
                img_path = os.path.join(folder, img_filename)
                save_image((tensor[i] + 1) / 2, img_path)  # scale [-1,1] to [0,1]


    
    # Training Loop + PSNR Logging
    
    def train(self):
        print("DEBUG: Starting training loop...")
        for epoch in range(self.num_epochs):
            print(f"Epoch [{epoch+1}/{self.num_epochs}]")
            epoch_psnr = []

            for batch_idx, batch in enumerate(self.dataloader):
                low_res_tensor = batch.get("low_res")
                high_res_tensor = batch.get("high_res")

                if low_res_tensor is None or high_res_tensor is None or low_res_tensor.size(0) == 0:
                    continue

                # Move tensors to device
                low = low_res_tensor.to(self.device)
                high = high_res_tensor.to(self.device)

                # Add noise (stabilizing)
                batch['high_res'] = batch['high_res'] + 0.02 * torch.randn_like(batch['high_res'])
                batch['low_res'] = batch['low_res'] + 0.02 * torch.randn_like(batch['low_res'])

                # Set input for model
                self.model.set_input({
                    "A": low,
                    "B": high,
                    "A_paths": batch.get("low_path"),
                    "B_paths": batch.get("high_path")
                })

                # Run training step
                self.model.optimize_parameters()

                # ---- PSNR CALCULATION ----
                visuals = self.model.get_current_visuals()
                fake_B = (visuals["fake_B"] + 1) / 2
                real_B = (visuals["real_B"] + 1) / 2
                psnr = self.calculate_psnr(fake_B, real_B)
                epoch_psnr.append(psnr)

                # Save triplet at last batch
                if batch_idx == len(self.dataloader) - 1:
                    self.save_triplet(epoch, batch_idx)

            # Mean PSNR for epoch
            mean_psnr = sum(epoch_psnr) / len(epoch_psnr)
            self.psnr_values.append(mean_psnr)
            print(f"Epoch [{epoch+1}] Mean PSNR: {mean_psnr:.4f} dB")

            # Log losses
            losses = self.model.get_current_losses()
            g_loss = losses.get("G", losses.get("G_GAN", 0.0))
            d_loss = losses.get("D_real", 0.0) + losses.get("D_fake", 0.0)
            nce_loss = losses.get("NCE", losses.get("G_idt", 0.0))
            print(f"G_loss: {g_loss:.4f}, D_loss: {d_loss:.4f}, NCE/IDT_loss: {nce_loss:.4f}")

        #Save PSNR graph after training 
        self.plot_psnr_curve()
        print(f"DEBUG: Training complete. Results saved in {self.results_dir}")


    #  PLOT PSNR 
    def plot_psnr_curve(self):
        plt.figure(figsize=(10, 5))
        plt.plot(range(1, len(self.psnr_values) + 1), self.psnr_values, marker='o')
        plt.title("PSNR over Epochs")
        plt.xlabel("Epoch")
        plt.ylabel("PSNR (dB)")
        plt.grid(True)
        img_path = os.path.join(self.results_dir, "psnr_curve.png")
        plt.savefig(img_path)
        plt.close()
        print(f"PSNR curve saved → {img_path}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CUT model")
    parser.add_argument("--low", default="/zhome/e5/7/219270/Deep-Learning/CUT_GAN_Loss/Deep-Learning/Data/Cropped_dataset/Low_res_cropped_specimen", help="Low resolution data folder")
    parser.add_argument("--high", default="/zhome/e5/7/219270/Deep-Learning/CUT_GAN_Loss/Deep-Learning/Data/Cropped_dataset/High_res_cropped_specimen", help="High resolution data folder")
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
        "data": {
            "low_res_dir": args.low,
            "high_res_dir": args.high,
            "paired": args.paired,
        },
        "training": {
            "batch_size": args.batch_size,
            "lr": 0.0001,
            "lr_D": 0.00001,
            "num_epochs": args.num_epochs,
            "num_workers": args.num_workers,
            "use_cpu": args.cpu,
            "lambda_idt": 10,
        },
        "cut": {
            "lambda_NCE": 0.5,
            "nce_T": 0.1,
            "num_patches": 96,
            "nce_layers": "4,8,12"
        },
        "use_simplified": True,
        "input_nc": 1,
        "output_nc": 1,
        "lambda_L1": 10.0,
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
