import torch
import torch.nn as nn
import torch.autograd as autograd

from Models.Network import define_G, define_D


class WGANModel:
    """
    Conditional WGAN-GP for low_res -> high_res on your IR dataset.

    - G: maps real_A (low_res) -> fake_B (high_res)
    - D (critic): sees concatenated [real_A, real_or_fake_B]
    - Losses:
        * Critic: E[D(A,fake_B)] - E[D(A,real_B)] + lambda_gp * GP
        * Generator: -E[D(A,fake_B)] + lambda_L1 * |fake_B - real_B|
    """

    def __init__(self, opt):
        """
        Expected opt attributes (with defaults):
          opt.input_nc, opt.output_nc, opt.ngf, opt.ndf
          opt.lr, opt.beta1, opt.beta2
          opt.lambda_L1, opt.lambda_gp, opt.n_critic
          opt.gpu_ids (list)
        """
        self.opt = opt
        self.device = torch.device(
            f"cuda:{opt.gpu_ids[0]}" if getattr(opt, "gpu_ids", []) else "cpu"
        )

        in_nc = getattr(opt, "input_nc", 1)
        out_nc = getattr(opt, "output_nc", 1)
        ngf = getattr(opt, "ngf", 64)
        ndf = getattr(opt, "ndf", 64)

        # Generator: low_res -> high_res
        self.netG = define_G(in_nc, out_nc, ngf).to(self.device)
        # Critic: sees concatenated [low_res, high_res] -> patch scores
        self.netD = define_D(in_nc + out_nc, ndf).to(self.device)

        # Optimizers
        lr = getattr(opt, "lr", 1e-4)
        beta1 = getattr(opt, "beta1", 0.5)
        beta2 = getattr(opt, "beta2", 0.9)  # common for WGAN-GP

        self.optimizer_G = torch.optim.Adam(
            self.netG.parameters(), lr=lr, betas=(beta1, beta2)
        )
        self.optimizer_D = torch.optim.Adam(
            self.netD.parameters(), lr=lr, betas=(beta1, beta2)
        )

        # Hyperparameters
        self.lambda_L1 = getattr(opt, "lambda_L1", 10.0)
        self.lambda_gp = getattr(opt, "lambda_gp", 10.0)
        self.n_critic = getattr(opt, "n_critic", 5)

        # Placeholders
        self.real_A = None
        self.real_B = None
        self.fake_B = None

        # For logging
        self.loss_D = 0.0
        self.loss_G = 0.0
        self.loss_G_adv = 0.0
        self.loss_G_L1 = 0.0
        self.loss_gp = 0.0
        self.loss_D_real = 0.0
        self.loss_D_fake = 0.0

    # -------------------------
    # Data / forward
    # -------------------------
    def set_input(self, batch):
        """
        batch from IRImageDataset:
          {
            'low_res':  tensor [B,1,H,W],
            'high_res': tensor [B,1,H,W],
            'low_path': str,
            'high_path': str
          }
        """
        self.real_A = batch["low_res"].to(self.device)
        self.real_B = batch["high_res"].to(self.device)

    def forward_G(self):
        self.fake_B = self.netG(self.real_A)

    # -------------------------
    # Gradient penalty
    # -------------------------
    def _gradient_penalty(self, real_pairs, fake_pairs):
        """
        real_pairs, fake_pairs: [B, C_concat, H, W]
        """
        batch_size = real_pairs.size(0)
        alpha = torch.rand(batch_size, 1, 1, 1, device=self.device)
        interpolates = alpha * real_pairs + (1 - alpha) * fake_pairs
        interpolates.requires_grad_(True)

        d_interpolates = self.netD(interpolates)
        # For patch output, sum over spatial dims to get scalar per sample
        d_interpolates = d_interpolates.view(batch_size, -1).mean(dim=1)

        gradients = autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=torch.ones_like(d_interpolates),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        gradients = gradients.view(batch_size, -1)
        grad_norm = gradients.norm(2, dim=1)
        gp = ((grad_norm - 1.0) ** 2).mean()
        return gp

    # -------------------------
    # Critic / generator steps
    # -------------------------
    def optimize_critic(self):
        """One critic (D) update step."""
        self.forward_G()
        fake_B = self.fake_B.detach()

        real_pairs = torch.cat([self.real_A, self.real_B], dim=1)
        fake_pairs = torch.cat([self.real_A, fake_B], dim=1)

        self.optimizer_D.zero_grad()

        d_real = self.netD(real_pairs)
        d_fake = self.netD(fake_pairs)

        # Mean over all outputs
        d_real_mean = d_real.view(d_real.size(0), -1).mean(dim=1)
        d_fake_mean = d_fake.view(d_fake.size(0), -1).mean(dim=1)

        loss_D_real = -d_real_mean.mean()
        loss_D_fake = d_fake_mean.mean()

        gp = self._gradient_penalty(real_pairs, fake_pairs)

        loss_D = loss_D_real + loss_D_fake + self.lambda_gp * gp
        loss_D.backward()
        self.optimizer_D.step()

        # Logging
        self.loss_D_real = float(d_real_mean.mean().detach().cpu())
        self.loss_D_fake = float(d_fake_mean.mean().detach().cpu())
        self.loss_gp = float(gp.detach().cpu())
        self.loss_D = float(loss_D.detach().cpu())

    def optimize_generator(self):
        """One generator (G) update step."""
        self.forward_G()
        fake_pairs = torch.cat([self.real_A, self.fake_B], dim=1)

        self.optimizer_G.zero_grad()

        d_fake = self.netD(fake_pairs)
        d_fake_mean = d_fake.view(d_fake.size(0), -1).mean(dim=1)

        loss_adv = -d_fake_mean.mean()
        loss_l1 = nn.functional.l1_loss(self.fake_B, self.real_B) * self.lambda_L1

        loss_G = loss_adv + loss_l1
        loss_G.backward()
        self.optimizer_G.step()

        # Logging
        self.loss_G_adv = float(loss_adv.detach().cpu())
        self.loss_G_L1 = float(loss_l1.detach().cpu())
        self.loss_G = float(loss_G.detach().cpu())

    # -------------------------
    # Helpers for trainer
    # -------------------------
    def get_current_losses(self):
        return {
            "D": self.loss_D,
            "D_real": self.loss_D_real,
            "D_fake": self.loss_D_fake,
            "GP": self.loss_gp,
            "G": self.loss_G,
            "G_adv": self.loss_G_adv,
            "G_L1": self.loss_G_L1,
        }

    def eval(self):
        self.netG.eval()
        self.netD.eval()

    def train_mode(self):
        self.netG.train()
        self.netD.train()
