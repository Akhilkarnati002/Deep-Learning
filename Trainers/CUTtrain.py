import torch
from torch.utils.data import DataLoader

from Utils.dataset import IRImageDataset
from Utils.transformers import transform_pipeline
from Models.CUT import CUTModel


class CUTTrainer:
    """ Trainer class for the CUT model. """

    def __init__(self, config):

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # -----------------------------
        # Load Dataset
        # -----------------------------
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
            num_workers=config["training"].get("num_workers", 4)
        )

        # -----------------------------
        # Initialize CUT Model
        # -----------------------------
        self.model = CUTModel(config).to(self.device)

        # -----------------------------
        # Optimizers
        # -----------------------------
        lr = config["training"]["lr"]

        self.optimizer_G = torch.optim.Adam(
            self.model.generator.parameters(),
            lr=lr,
            betas=(0.5, 0.999)
        )

        self.optimizer_D = torch.optim.Adam(
            self.model.discriminator.parameters(),
            lr=lr,
            betas=(0.5, 0.999)
        )

        self.num_epochs = config["training"]["num_epochs"]

    # --------------------------------------------------------------------
    # Training Loop
    # --------------------------------------------------------------------
    def train(self):

        for epoch in range(self.num_epochs):
            for batch in self.dataloader:

                low = batch["low_res"].to(self.device)
                high = batch["high_res"].to(self.device)

                # ---- Train Generator ----
                self.optimizer_G.zero_grad()
                losses_G = self.model.compute_G_loss(low, high)
                losses_G["G_total"].backward()
                self.optimizer_G.step()

                # ---- Train Discriminator ----
                self.optimizer_D.zero_grad()
                losses_D = self.model.compute_D_loss(low, high)
                losses_D["D_total"].backward()
                self.optimizer_D.step()

            print(f"Epoch [{epoch+1}/{self.num_epochs}] "
                  f"G_loss: {losses_G['G_total'].item():.4f}, "
                  f"D_loss: {losses_D['D_total'].item():.4f}")


# --------------------------------------------------------------------
# Run training directly
# --------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser(description="Train CUT model")
    parser.add_argument("--low", help="Low resolution data folder", default="Data/Low_resolution/CFRP_60_low")
    parser.add_argument("--high", help="High resolution data folder", default="Data/High_resolution/CFRP_60_high")
    parser.add_argument("--paired", action="store_true", help="Use paired dataset (default: False)")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.0002)
    parser.add_argument("--num-epochs", type=int, default=50)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--dry-run", action="store_true", help="Only instantiate trainer and print diagnostics, don't run training")

    args = parser.parse_args()

    config = {
        "data": {
            "low_res_dir": args.low,
            "high_res_dir": args.high,
            "paired": args.paired,
        },
        "training": {
            "batch_size": args.batch_size,
            "lr": args.lr,
            "num_epochs": args.num_epochs,
            "num_workers": args.num_workers,
        },
        "cut": {
            "lambda_NCE": 1.0,
            "num_patches": 256,
        },
    }

    def count_files(path):
        try:
            return sum(1 for _ in os.listdir(path) if os.path.isfile(os.path.join(path, _)))
        except Exception:
            return 0

    print("Dataset paths:")
    print("  low:", config["data"]["low_res_dir"], "-> files:", count_files(config["data"]["low_res_dir"]))
    print("  high:", config["data"]["high_res_dir"], "-> files:", count_files(config["data"]["high_res_dir"]))

    if args.dry_run:
        # perform a quick dataset/dataloader sanity check without constructing the model
        from Utils.dataset import IRImageDataset
        from Utils.transformers import transform_pipeline

        ds = IRImageDataset(low_dir=config["data"]["low_res_dir"],
                            high_dir=config["data"]["high_res_dir"],
                            transform=transform_pipeline,
                            paired=config["data"].get("paired", False))
        print("Dry-run: dataset length:", len(ds))
        sample = ds[0]
        print("Dry-run: sample keys:", list(sample.keys()))
    else:
        trainer = CUTTrainer(config)
        trainer.train()