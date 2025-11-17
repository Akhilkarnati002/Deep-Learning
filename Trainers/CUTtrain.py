import torch
from torch.utils.data import DataLoader
from Utils.dataset import IRImageDataset
from Utils.transformers import transform_pipeline
from Models.CUT import CUTModel



class CUTtrainer:
    """
    Trainer class for the CUT model.
    """

    def __init__(self, config):

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


        #load dataset
        dataset = IRImageDataset(
            low_dir=config['data']['low_res_dir'],
            high_dir=config['data']['high_res_dir'],
            transform=transform_pipeline,
            paired=config['data'].get('paired', False)
        )

        self.dataloader = DataLoader(
            dataset,
            batch_size=config['training']['batch_size'],
            shuffle=True,
            num_workers=config['training'].get('num_workers', 4)
        )


        # Initialize CUT model
        self.model = CUTModel(config).to(self.device)   

        # Optimizers
        self.optimizer_G = torch.optim.Adam(
            self.model.generator.parameters(),
            lr=config['training']['lr'],
            betas=(0.5, 0.999)
        )

        self.optimizer_D = torch.optim.Adam(
            self.model.discriminator.parameters(),
            lr=config['training']['lr'],
            betas=(0.5, 0.999)
        )

        self.num_epochs = config['training']['num_epochs']

    def train(self):
        """
        Training loop for the CUT model.
        """
     
        for batch in self.dataloader:
            low = batch["low_res"].to(self.device)
            high = batch["high_res"].to(self.device)


            # --- Train Generator ---
            self.optimizer_G.zero_grad()
            losses_G = self.model.compute_G_loss(low, high)
            losses_G["G_total"].backward()
            self.optimizer_G.step()


            # --- Train Discriminator ---
            self.optimizer_D.zero_grad()
            losses_D = self.model.compute_D_loss(low, high)
            losses_D["D_total"].backward()
            self.optimizer_D.step()

        print(f"Epoch [{epoch+1}/{self.epochs}] - G: {losses_G['G_total'].item()} , D: {losses_D['D_total'].item()}")
    




if __name__ == "__main__":
    config = {
    "low_dir": "data/low_res",
    "high_dir": "data/high_res",
    "batch_size": 4,
    "lr": 0.0002,
    "epochs": 50,
    "paired": True,
    "lambda_NCE": 1.0,
    "num_patches": 256,
    }

    trainer = CUTTrainer(config)
    trainer.train()
