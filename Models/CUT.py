import torch
import torch.nn as nn
import torch.nn.functional as F
import random

from Models.Network import define_G, define_D
from Losses.NCE_losses import PatchNCELoss


class PatchSampler:
    """
    Simple patch sampler: given a feature map BxCxHxW, flatten spatial dims and
    sample num_patches indices (same indices used for query & key).
    Returns tensors shaped [B, C, num_patches].
    """
    def __init__(self, num_patches=256):
        self.num_patches = num_patches

    def sample(self, feat):
        # feat: B x C x H x W
        B, C, H, W = feat.shape
        N = H * W
        feat_flat = feat.view(B, C, N)               # B x C x N

        if self.num_patches >= N:
            # take all (or pad by sampling with replacement)
            if self.num_patches == N:
                return feat_flat
            else:
                # sample with replacement to reach desired number
                idx = torch.randint(0, N, (self.num_patches,), device=feat.device)
                idx = idx.unsqueeze(0).expand(B, -1)   # B x num_patches
        else:
            # sample without replacement
            perm = torch.randperm(N, device=feat.device)[: self.num_patches]
            idx = perm.unsqueeze(0).expand(B, -1)    # B x num_patches

        # gather: for each sample in batch select same spatial indices
        # feat_flat: B x C x N  -> permute -> B x N x C -> gather -> B x num_patches x C -> permute -> B x C x num_patches
        feat_t = feat_flat.permute(0, 2, 1)           # B x N x C
        sampled = torch.gather(feat_t, 1, idx.unsqueeze(-1).expand(-1, -1, C))  # B x num_patches x C
        sampled = sampled.permute(0, 2, 1)            # B x C x num_patches
        return sampled


class CUTModel:
    """
    CUTModel supporting:
      - full CUT (PatchNCE + GAN) when opt.use_simplified == False
      - simplified GAN+L1 when opt.use_simplified == True

    Required opt attributes (defaults provided if missing):
       opt.input_nc (channels in)
       opt.output_nc
       opt.ngf, opt.ndf
       opt.lr
       opt.lambda_GAN
       opt.lambda_NCE
       opt.num_patches
       opt.nce_layers  -- comma-separated indices like "0,4,8"
       opt.use_simplified (bool)
       opt.gpu_ids (list)
       opt.checkpoints_dir, opt.name, opt.epoch (for save/load)
       opt.lr_policy etc. (used by setup if you integrate scheduler)
    """

    def __init__(self, opt):
        self.opt = opt
        self.device = torch.device('cuda:{}'.format(opt.gpu_ids[0]) if getattr(opt, 'gpu_ids', []) else 'cpu')

        # read options with safe defaults
        in_nc = getattr(opt, 'input_nc', 1)
        out_nc = getattr(opt, 'output_nc', 1)
        ngf = getattr(opt, 'ngf', 64)
        ndf = getattr(opt, 'ndf', 64)
        self.use_simplified = getattr(opt, 'use_simplified', False)

        # create networks from your networks.py
        self.netG = define_G(in_nc, out_nc, ngf).to(self.device)
        self.netD = define_D(out_nc, ndf).to(self.device)

        # Patch sampler & NCE loss (only used in full CUT)
        self.num_patches = getattr(opt, 'num_patches', 256)
        self.sampler = PatchSampler(self.num_patches)
        self.nce_layers = [int(x) for x in getattr(opt, 'nce_layers', '0,4,8,12,16').split(',')]\
                              if not self.use_simplified else []
        self.criterionNCE = PatchNCELoss(temperature=getattr(opt, 'nce_T', 0.07),
                                         num_patches=self.num_patches).to(self.device) if not self.use_simplified else None

        # GAN loss: use LSGAN style (MSE) for stability
        self.criterionGAN = nn.MSELoss()

        # Identity / reconstruction loss (for simplified mode / optional)
        self.criterionIdt = nn.L1Loss()

        # loss weights
        self.lambda_GAN = getattr(opt, 'lambda_GAN', 1.0)
        self.lambda_NCE = getattr(opt, 'lambda_NCE', 1.0) if not self.use_simplified else 0.0
        self.lambda_idt = getattr(opt, 'lambda_idt', 0.5) if self.use_simplified else 0.0

        # optimizers
        lr = getattr(opt, 'lr', 2e-4)
        beta1 = getattr(opt, 'beta1', 0.5)
        beta2 = getattr(opt, 'beta2', 0.999)

        if self.use_simplified:
            # simplified: only generator and discriminator
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=lr, betas=(beta1, beta2))
        else:
            # full CUT: we will update G and (if needed) separate projector layers inside loss - keep G optimizer
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=lr, betas=(beta1, beta2))
        self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=lr, betas=(beta1, beta2))

        # bookkeeping
        self.loss_names = []
        self.visual_names = []
        self.model_names = ['G', 'D']
        self.optimizers = [self.optimizer_G, self.optimizer_D]
        self.schedulers = []  # created in setup if desired

        # placeholders set during training
        self.real_A = None
        self.real_B = None
        self.fake_B = None
        self.flipped_for_equivariance = False

    # -------------------------
    # Utilities required by training loop
    # -------------------------
    def set_input(self, input):
        """Unpack input dict from dataloader and send to device.
           Expects keys 'A' and 'B' (low-res, high-res) or adapt accordingly.
        """
        AtoB = getattr(self.opt, 'direction', 'AtoB') == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input.get('A_paths', None)

    def data_dependent_initialize(self, data):
        """Optional: can be called on the first batch to set up anything that
           depends on input size (not required here). Kept for API compatibility."""
        # no special data-dependent layers in this simplified implementation
        pass

    def setup(self, opt):
        """Create schedulers if you want. Keep minimal here."""
        # Optionally create LR schedulers by user code, e.g.:
        # from util.lr_scheduler import get_scheduler
        # self.schedulers = [get_scheduler(opt, optimizer) ...]
        pass

    def parallelize(self):
        if getattr(self.opt, 'gpu_ids', []):
            if torch.cuda.device_count() > 1:
                self.netG = nn.DataParallel(self.netG, self.opt.gpu_ids)
                self.netD = nn.DataParallel(self.netD, self.opt.gpu_ids)

    def eval(self):
        self.netG.eval()
        self.netD.eval()

    def test(self):
        with torch.no_grad():
            self.fake_B = self.netG(self.real_A)
            self.compute_visuals()

    def compute_visuals(self):
        # prepare tensors for visualization; keep values in [-1,1] as generator outputs Tanh
        self.visuals = {
            'real_A': self.real_A.detach(),
            'fake_B': self.fake_B.detach() if self.fake_B is not None else None,
            'real_B': self.real_B.detach()
        }

    def get_current_visuals(self):
        return getattr(self, 'visuals', {})

    def get_image_paths(self):
        return getattr(self, 'image_paths', [])

    def get_current_losses(self):
        """Return scalar floats for logging. Keys are used by trainer to print."""
        losses = {}
        # GAN losses
        losses['G_GAN'] = float(getattr(self, 'loss_G_GAN', 0.0))
        losses['D_real'] = float(getattr(self, 'loss_D_real', 0.0))
        losses['D_fake'] = float(getattr(self, 'loss_D_fake', 0.0))
        # NCE / idt
        if not self.use_simplified:
            losses['NCE'] = float(getattr(self, 'loss_NCE', 0.0))
        else:
            losses['G_idt'] = float(getattr(self, 'loss_idt', 0.0))
        # combined G loss
        losses['G'] = float(getattr(self, 'loss_G_total', 0.0))
        return losses

    def save_networks(self, epoch_label):
        save_dir = getattr(self.opt, 'checkpoints_dir', './checkpoints')
        name = getattr(self.opt, 'name', 'experiment')
        import os
        os.makedirs(save_dir, exist_ok=True)

        G_path = os.path.join(save_dir, f'{epoch_label}_net_G.pth')
        D_path = os.path.join(save_dir, f'{epoch_label}_net_D.pth')
        torch.save(self.netG.state_dict(), G_path)
        torch.save(self.netD.state_dict(), D_path)

    def load_networks(self, epoch_label):
        save_dir = getattr(self.opt, 'checkpoints_dir', './checkpoints')
        G_path = os.path.join(save_dir, f'{epoch_label}_net_G.pth')
        D_path = os.path.join(save_dir, f'{epoch_label}_net_D.pth')
        self.netG.load_state_dict(torch.load(G_path, map_location=self.device))
        self.netD.load_state_dict(torch.load(D_path, map_location=self.device))

    def update_learning_rate(self):
        for scheduler in getattr(self, 'schedulers', []):
            scheduler.step()
        # expose lr
        if len(self.optimizers) > 0:
            return self.optimizers[0].param_groups[0]['lr']
        return None

    # -------------------------
    # Feature extraction for NCE: find features from generator layers
    # We look up features by layer index in the generator's sequential model.
    # -------------------------
    def _extract_generator_features(self, x, layer_ids):
        """
        Run x through self.netG.model (the sequential container) and return
        a dict of features for indices in layer_ids.
        layer_ids: list of integer indices (matching sequential indices)
        returns dict: {layer_idx: feature_tensor(B,C,H,W)}
        """
        feats = {}
        current = x
        # self.netG.model is nn.Sequential as in your networks.py
        for idx, layer in enumerate(self.netG.model):
            current = layer(current)
            if idx in layer_ids:
                feats[idx] = current
        return feats

    # -------------------------
    # NCE calculation
    # -------------------------
    def calculate_NCE_loss(self, src, tgt):
        """Compute NCE loss between source (real_A) and target (fake_B).
           src,tgt are tensors image-space: BxCxHxW
        """
        # extract required features
        # note: layer indices used must be in range of self.netG.model
        layer_ids = [i for i in self.nce_layers if i < len(self.netG.model)]
        if len(layer_ids) == 0:
            # fallback: use last conv output
            layer_ids = [len(self.netG.model) - 1]

        feat_q_dict = self._extract_generator_features(tgt, layer_ids)  # fake
        feat_k_dict = self._extract_generator_features(src, layer_ids)  # real

        total_loss = 0.0
        for lid in layer_ids:
            fq = feat_q_dict[lid]  # BxCxHxW
            fk = feat_k_dict[lid]

            # sample patches -> B x C x num_patches
            q_samples = self.sampler.sample(fq)
            k_samples = self.sampler.sample(fk)

            # make sure shapes align
            # convert from BxCxP to the expected [B, C, P] form used by PatchNCELoss
            loss_layer = self.criterionNCE(q_samples, k_samples)
            total_loss += loss_layer

        total_loss = total_loss / len(layer_ids)
        return total_loss

    # -------------------------
    # GAN helper (LSGAN style)
    # -------------------------
    def _compute_G_gan_loss(self, fake):
        pred_fake = self.netD(fake)
        target_real = torch.ones_like(pred_fake, device=pred_fake.device)
        return self.criterionGAN(pred_fake, target_real)

    def _compute_D_loss(self, fake, real):
        pred_fake = self.netD(fake.detach())
        pred_real = self.netD(real)
        target_fake = torch.zeros_like(pred_fake, device=pred_fake.device)
        target_real = torch.ones_like(pred_real, device=pred_real.device)
        loss_fake = self.criterionGAN(pred_fake, target_fake)
        loss_real = self.criterionGAN(pred_real, target_real)
        # store components for logging
        self.loss_D_fake = float(loss_fake.detach().cpu())
        self.loss_D_real = float(loss_real.detach().cpu())
        return 0.5 * (loss_fake + loss_real)

    # -------------------------
    # Training step
    # -------------------------
    def optimize_parameters(self):
        # forward
        self.fake_B = self.netG(self.real_A)

        # ------- UPDATE D -------
        self.netD.train()
        self.optimizer_D.zero_grad()
        loss_D = self._compute_D_loss(self.fake_B, self.real_B)
        loss_D.backward()
        self.optimizer_D.step()

        # ------- UPDATE G -------
        self.netG.train()
        self.optimizer_G.zero_grad()

        # GAN loss
        self.loss_G_GAN = self._compute_G_gan_loss(self.fake_B) * self.lambda_GAN

        if self.use_simplified:
            # Identity (L1) to encourage reconstruction, optional
            self.loss_idt = self.criterionIdt(self.fake_B, self.real_B) * self.lambda_idt
            self.loss_G_total = self.loss_G_GAN + self.loss_idt
        else:
            # NCE loss
            self.loss_NCE = self.calculate_NCE_loss(self.real_A, self.fake_B) * self.lambda_NCE
            self.loss_G_total = self.loss_G_GAN + self.loss_NCE

        # backprop G
        self.loss_G_total.backward()
        self.optimizer_G.step()

        # For trainer logging, set attributes (scalars)
        self.loss_G_GAN = float(self.loss_G_GAN.detach().cpu())
        if not self.use_simplified:
            self.loss_NCE = float(self.loss_NCE.detach().cpu())
        else:
            self.loss_idt = float(self.loss_idt.detach().cpu())
        self.loss_G_total = float(self.loss_G_total.detach().cpu())

        # D components already set earlier
        self.loss_D_fake = getattr(self, 'loss_D_fake', 0.0)
        self.loss_D_real = getattr(self, 'loss_D_real', 0.0)

    # -------------------------
    # Convenience - compute D/G losses (for compatibility)
    # -------------------------
    def compute_D_loss(self):
        """Compatibility function used by some trainer loops: returns scalar loss for D"""
        # quick forward if not already done
        fake = self.netG(self.real_A).detach()
        return self._compute_D_loss(fake, self.real_B)

    def compute_G_loss(self):
        """Compatibility function used by some trainer loops"""
        fake = self.netG(self.real_A)
        g_gan = self._compute_G_gan_loss(fake) * self.lambda_GAN
        if self.use_simplified:
            idt = self.criterionIdt(fake, self.real_B) * self.lambda_idt
            return g_gan + idt
        else:
            nce = self.calculate_NCE_loss(self.real_A, fake) * self.lambda_NCE
            return g_gan + nce