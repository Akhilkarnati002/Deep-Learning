import torch
from torch.utils.data import DataLoader
from types import SimpleNamespace

from Utils.dataset import SuperResolutionDataset
from Models.SR import SRModel


class SRTrainer:
    """Trainer for supervised super-resolution using SRModel."""

    def __init__(self, config):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Dataset / DataLoader
        data_cfg = config["data"]
        hr_dir = data_cfg["hr_dir"]
        hr_size = data_cfg.get("hr_size", (256, 512))
        scale_factor = data_cfg.get("scale_factor", 4)

        dataset = SuperResolutionDataset(
            hr_dir=hr_dir,
            hr_size=hr_size,
            scale_factor=scale_factor,
        )

        self.dataloader = DataLoader(
            dataset,
            batch_size=config["training"]["batch_size"],
            shuffle=True,
            num_workers=config["training"].get("num_workers", 0),
        )

        # Build opt namespace for SRModel
        sr_opt = self._build_sr_opt(config)
        self.model = SRModel(sr_opt)

        self.num_epochs = config["training"]["num_epochs"]

    def _build_sr_opt(self, config):
        model_cfg = config.get("model", {})
        train_cfg = config.get("training", {})

        opt = SimpleNamespace()

        if torch.cuda.is_available():
            opt.gpu_ids = model_cfg.get("gpu_ids", [0])
        else:
            opt.gpu_ids = []

        opt.input_nc = model_cfg.get("input_nc", 3)
        opt.output_nc = model_cfg.get("output_nc", 3)
        opt.ngf = model_cfg.get("ngf", 32)

        opt.lr = train_cfg.get("lr", 1e-4)
        opt.beta1 = train_cfg.get("beta1", 0.9)
        opt.beta2 = train_cfg.get("beta2", 0.999)

        return opt

    def train(self):
        print("Starting SR training...")
        for epoch in range(self.num_epochs):
            print(f"Epoch [{epoch+1}/{self.num_epochs}]")
            for batch in self.dataloader:
                self.model.set_input(batch)
                self.model.optimize_parameters()

            losses = self.model.get_current_losses()
            print(f"Epoch [{epoch+1}/{self.num_epochs}] L1: {losses['L1']:.4f}")
