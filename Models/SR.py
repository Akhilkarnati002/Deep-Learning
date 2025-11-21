import torch
import torch.nn as nn

from Models.Network import define_G


class SRModel:
    """
    Simple supervised super-resolution model using the existing ResentGenerator.

    - Input: lr (upsampled low-res) in [-1, 1]
    - Output: fake_hr, same shape as hr
    - Loss: L1(fake_hr, hr)
    """

    def __init__(self, opt):
        """
        Expected opt attributes (with defaults):
          opt.input_nc, opt.output_nc, opt.ngf
          opt.lr, opt.beta1, opt.beta2
          opt.gpu_ids (list)
        """
        self.opt = opt
        self.device = torch.device(
            f"cuda:{opt.gpu_ids[0]}" if getattr(opt, "gpu_ids", []) else "cpu"
        )

        in_nc = getattr(opt, "input_nc", 1)
        out_nc = getattr(opt, "output_nc", 1)
        ngf = getattr(opt, "ngf", 64)

        self.netG = define_G(in_nc, out_nc, ngf).to(self.device)

        lr = getattr(opt, "lr", 1e-4)
        beta1 = getattr(opt, "beta1", 0.9)
        beta2 = getattr(opt, "beta2", 0.999)

        self.optimizer_G = torch.optim.Adam(
            self.netG.parameters(), lr=lr, betas=(beta1, beta2)
        )

        self.criterionL1 = nn.L1Loss()

        self.lr = None
        self.hr = None
        self.fake_hr = None
        self.loss_L1 = 0.0

    def set_input(self, batch):
        self.lr = batch["lr"].to(self.device)
        self.hr = batch["hr"].to(self.device)

    def optimize_parameters(self):
        self.netG.train()
        self.optimizer_G.zero_grad()
        self.fake_hr = self.netG(self.lr)
        loss = self.criterionL1(self.fake_hr, self.hr)
        loss.backward()
        self.optimizer_G.step()

        self.loss_L1 = float(loss.detach().cpu())

    def eval(self):
        self.netG.eval()

    def get_current_losses(self):
        return {"L1": self.loss_L1}

