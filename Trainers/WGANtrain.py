import torch
from torch.utils.data import DataLoader
from types import SimpleNamespace

from Utils.dataset import IRImageDataset
from Utils.transformers import transform_pipeline
from Models.WGAN import WGANModel


class WGANTrainer:
    """Trainer for the WGAN-GP model on your IR dataset."""

    def __init__(self, config):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Dataset / DataLoader
        dataset = IRImageDataset(
            low_dir=config["data"]["low_res_dir"],
            high_dir=config["data"]["high_res_dir"],
            transform=transform_pipeline,
            paired=config["data"].get("paired", True),
        )

        self.dataloader = DataLoader(
            dataset,
            batch_size=config["training"]["batch_size"],
            shuffle=True,
            num_workers=config["training"].get("num_workers", 0),
        )

        # Build opt namespace for WGANModel
        wgan_opt = self._build_wgan_opt(config)
        self.model = WGANModel(wgan_opt)

        self.num_epochs = config["training"]["num_epochs"]
        self.n_critic = wgan_opt.n_critic

    def _build_wgan_opt(self, config):
        """
        Convert config dict into opt namespace for WGANModel.

        Expected config sections:
          - config["model"]: input_nc, output_nc, ngf, ndf
          - config["training"]: lr, beta1, beta2, num_epochs, batch_size
          - config["wgan"]: lambda_L1, lambda_gp, n_critic, gpu_ids
        """
        model_cfg = config.get("model", {})
        train_cfg = config.get("training", {})
        wgan_cfg = config.get("wgan", {})

        opt = SimpleNamespace()

        # GPU ids
        if torch.cuda.is_available():
            opt.gpu_ids = wgan_cfg.get("gpu_ids", [0])
        else:
            opt.gpu_ids = []

        # Model channels
        opt.input_nc = model_cfg.get("input_nc", 1)
        opt.output_nc = model_cfg.get("output_nc", 1)
        opt.ngf = model_cfg.get("ngf", 64)
        opt.ndf = model_cfg.get("ndf", 64)

        # Optimizer params
        opt.lr = train_cfg.get("lr", 1e-4)
        opt.beta1 = train_cfg.get("beta1", 0.5)
        opt.beta2 = train_cfg.get("beta2", 0.9)

        # WGAN-specific
        opt.lambda_L1 = wgan_cfg.get("lambda_L1", 10.0)
        opt.lambda_gp = wgan_cfg.get("lambda_gp", 10.0)
        opt.n_critic = wgan_cfg.get("n_critic", 5)

        return opt

    def train(self):
        print("Starting WGAN-GP training...")
        for epoch in range(self.num_epochs):
            print(f"Epoch [{epoch+1}/{self.num_epochs}]")
            for i, batch in enumerate(self.dataloader):
                self.model.train_mode()
                self.model.set_input(batch)

                # Critic updates
                for _ in range(self.n_critic):
                    self.model.optimize_critic()

                # Generator update
                self.model.optimize_generator()

            losses = self.model.get_current_losses()
            print(
                f"Epoch [{epoch+1}/{self.num_epochs}] "
                f"G: {losses['G']:.4f} (adv {losses['G_adv']:.4f}, L1 {losses['G_L1']:.4f}), "
                f"D: {losses['D']:.4f}, GP: {losses['GP']:.4f}"
            )
