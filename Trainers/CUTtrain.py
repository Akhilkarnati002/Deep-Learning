import torch
from torch.utils.data import DataLoader
from types import SimpleNamespace

from Utils.dataset import IRImageDataset
from Utils.transformers import transform_pipeline
from Models.CUT import CUTModel


class CUTTrainer:
    """ Trainer class for the CUT model. """

    def __init__(self, config):

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
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
        # Initialize CUT Model (adapt dict config to CUTModel opt object)
        # -----------------------------
        cut_opt = self._build_cut_opt(config)
        self.model = CUTModel(cut_opt)

        # number of epochs
        self.num_epochs = config["training"]["num_epochs"]

    def _build_cut_opt(self, config):
        """
        Convert nested `config` dict into an attribute-style `opt`
        object expected by `CUTModel`.

        Expected `config` structure (with reasonable defaults):
            config["model"]: channels and base nets
            config["training"]: lr, betas, epochs, etc.
            config["cut"]: CUT-specific hyperparameters
        """
        model_cfg = config.get("model", {})
        train_cfg = config.get("training", {})
        cut_cfg = config.get("cut", {})

        opt = SimpleNamespace()

        # GPU / device
        if torch.cuda.is_available():
            opt.gpu_ids = cut_cfg.get("gpu_ids", [0])
        else:
            opt.gpu_ids = []

        # Generator / discriminator channels
        opt.input_nc = model_cfg.get("input_nc", 1)
        opt.output_nc = model_cfg.get("output_nc", 1)
        opt.ngf = model_cfg.get("ngf", 64)
        opt.ndf = model_cfg.get("ndf", 64)

        # Training hyperparameters
        opt.lr = train_cfg.get("lr", 2e-4)
        opt.beta1 = train_cfg.get("beta1", 0.5)
        opt.beta2 = train_cfg.get("beta2", 0.999)

        # CUT / loss hyperparameters
        opt.lambda_GAN = cut_cfg.get("lambda_GAN", 1.0)
        opt.lambda_NCE = cut_cfg.get("lambda_NCE", 1.0)
        opt.lambda_idt = cut_cfg.get("lambda_idt", 0.5)
        opt.num_patches = cut_cfg.get("num_patches", 256)
        opt.nce_layers = cut_cfg.get("nce_layers", "0,4,8,12,16")
        opt.nce_T = cut_cfg.get("nce_T", 0.07)
        opt.use_simplified = cut_cfg.get("use_simplified", False)

        # Misc / bookkeeping
        opt.checkpoints_dir = cut_cfg.get("checkpoints_dir", "./checkpoints")
        opt.name = cut_cfg.get("name", "cut_experiment")
        opt.direction = cut_cfg.get("direction", "AtoB")

        return opt

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
