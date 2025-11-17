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
        # Initialize CUT Model (model manages its own device/optimizers)
        # -----------------------------
        self.model = CUTModel(config)

        # number of epochs
        self.num_epochs = config["training"]["num_epochs"]

    # --------------------------------------------------------------------
    # Training Loop
    # --------------------------------------------------------------------
    def train(self):

        for epoch in range(self.num_epochs):
            for batch in self.dataloader:

                low = batch["low_res"].to(self.model.device)
                high = batch["high_res"].to(self.model.device)

                # Feed inputs to model (CUTModel expects keys 'A' and 'B')
                input_dict = {
                    'A': low,
                    'B': high,
                    'A_paths': batch.get('low_path'),
                    'B_paths': batch.get('high_path')
                }
                self.model.set_input(input_dict)

                # Run a training step using model's API
                self.model.optimize_parameters()

            # after epoch, collect and print current losses (may be last batch's values)
            losses = self.model.get_current_losses()
            g_loss = losses.get('G', losses.get('G_GAN', 0.0))
            d_loss = losses.get('D_real', 0.0) + losses.get('D_fake', 0.0)
            print(f"Epoch [{epoch+1}/{self.num_epochs}] G_loss: {g_loss:.4f}, D_loss: {d_loss:.4f}")


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

    if args.dry_run :
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

        trainer = CUTTrainer(config)
        trainer.train()
    else:
        trainer = CUTTrainer(config)
        trainer.train()