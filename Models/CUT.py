# CUT.py
# Full implementation of the CUTModel (single-file) that integrates:
# - ResNet generator / PatchGAN discriminator (expected to be provided by Models.Network)
# - PatchNCE loss (expected to be provided by Losses.NCE_losses.PatchNCELoss)
# - A small differentiable DegradationNet that converts HR generator output
#   into a LR / cheap-camera style image (learnable or fixed)
#

import torch
import torch.nn as nn
import torch.nn.functional as F
import random

# Project imports (expected)
from Models.BaseModel import BaseModel
from Models.Network import define_G, define_D
try:
    # Prefer a DegradationNet implemented in Networks module if available
    from Models.Network import DegradationNet
    _HAS_EXTERNAL_DEGRADE = True
except Exception:
    _HAS_EXTERNAL_DEGRADE = False

from Losses.NCE_losses import PatchNCELoss


# -------------------------
# Local fallback DegradationNet (used if Networks doesn't provide one)
# -------------------------
class DegradationNetFallback(nn.Module):
    """
    Simple differentiable degradation module:
      - small blur convs
      - avg-pool downsample by integer factor (downscale)
      - refine convs
      - optional learnable noise scale per-channel
    Returns output in same value domain as generator (tanh expected if generator uses tanh).
    """
    def __init__(self, in_nc=3, mid_nc=32, downscale=4, use_learnable_noise=True):
        super().__init__()
        self.downscale = max(1, int(downscale))
        self.blur = nn.Sequential(
            nn.Conv2d(in_nc, mid_nc, kernel_size=5, padding=2, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_nc, mid_nc, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
        )
        self.refine = nn.Sequential(
            nn.Conv2d(mid_nc, mid_nc, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_nc, in_nc, kernel_size=3, padding=1),
        )
        self.use_learnable_noise = bool(use_learnable_noise)
        if self.use_learnable_noise:
            # initialized near zero (no noise) so the network can learn noise magnitude
            self.noise_scale = nn.Parameter(torch.zeros(in_nc))
        else:
            self.register_buffer('noise_scale', torch.zeros(in_nc))

    def forward(self, x):
        """
        x: B x C x H x W (HR)
        returns: B x C x H/downscale x W/downscale (LR-style)
        """
        b = self.blur(x)
        if self.downscale > 1:
            # average pooling to downsample (differentiable)
            b = F.avg_pool2d(b, kernel_size=self.downscale, stride=self.downscale)
        out = self.refine(b)
        # add gaussian noise scaled per-channel
        if self.use_learnable_noise:
            scale = self.noise_scale.abs().view(1, -1, 1, 1) + 1e-8
            noise = torch.randn_like(out) * scale
            out = out + noise
        else:
            out = out + 0.01 * torch.randn_like(out)
        # match generator output domain (tanh commonly used). If your generator uses other
        # output ranges, adapt accordingly.
        return torch.tanh(out)


# -------------------------
# Patch sampler
# -------------------------
class PatchSampler:
    """
    Sample patches/features from a feature map.
    Input: feat (B x C x H x W)
    Output: sampled (B x C x num_patches)
    """
    def __init__(self, num_patches=256):
        self.num_patches = int(num_patches)

    def sample(self, feat):
        B, C, H, W = feat.shape
        N = H * W
        feat_flat = feat.view(B, C, N)               # B x C x N

        device = feat.device
        if self.num_patches >= N:
            if self.num_patches > N:
                # sample with replacement
                idx = torch.randint(0, N, (B, self.num_patches), device=device)
            else:
                # sample all indices
                idx = torch.arange(N, device=device).unsqueeze(0).expand(B, -1)
        else:
            # sample without replacement per batch element
            perm = torch.stack([torch.randperm(N, device=device)[: self.num_patches] for _ in range(B)])
            idx = perm    # B x num_patches

        # feat_flat: B x C x N -> permute to B x N x C so gather works on dim=1
        feat_t = feat_flat.permute(0, 2, 1)           # B x N x C
        sampled = torch.gather(feat_t, 1, idx.unsqueeze(-1).expand(-1, -1, C))  # B x num_patches x C
        sampled = sampled.permute(0, 2, 1)            # B x C x num_patches
        return sampled


# -------------------------
# CUTModel
# -------------------------
class CUTModel(BaseModel):
    """
    Contrastive Unpaired Translation Model with optional degradation module
    to simulate low-resolution / cheap-camera outputs.

    Expects 'opt' dictionary-like object (or namespace) with keys used below.
    """

    def __init__(self, opt):
        # Initialize BaseModel (sets self.opt, self.device, self.save_dir, etc.)
        super().__init__(opt)

        # Basic network params (with safe defaults)
        in_nc = int(self.opt.get('input_nc', 1))
        out_nc = int(self.opt.get('output_nc', 1))
        ngf = int(self.opt.get('ngf', 64))
        ndf = int(self.opt.get('ndf', 64))
        self.use_simplified = bool(self.opt.get('use_simplified', False))

        # Create Generator and Discriminator
        self.netG = define_G(in_nc, out_nc, ngf)
        self.netD = define_D(out_nc, ndf)

        # Degradation network: try to import from Networks, otherwise fallback
        degrade_conf = self.opt.get('degradation', {})
        downscale = int(degrade_conf.get('downscale', 4))
        learn_noise = bool(degrade_conf.get('learn_noise', True))
        mid_nc = int(degrade_conf.get('mid_nc', 32))

        if _HAS_EXTERNAL_DEGRADE:
            try:
                # prefer DegradationNet from Models.Network if available
                self.netDegrade = DegradationNet(in_nc=out_nc, mid_nc=mid_nc, downscale=downscale, use_learnable_noise=learn_noise)
            except Exception:
                self.netDegrade = DegradationNetFallback(in_nc=out_nc, mid_nc=mid_nc, downscale=downscale, use_learnable_noise=learn_noise)
        else:
            self.netDegrade = DegradationNetFallback(in_nc=out_nc, mid_nc=mid_nc, downscale=downscale, use_learnable_noise=learn_noise)

        # Model name list (used by BaseModel utilities)
        # NOTE: BaseModel.save_networks/load_networks often expect attribute names 'netG', 'netD', etc.
        self.model_names = ['G', 'D', 'Degrade']

        # NCE / Patch sampling options
        cut_opt = self.opt.get('cut', {}) if isinstance(self.opt.get('cut', {}), dict) else {}
        self.num_patches = int(cut_opt.get('num_patches', 256))
        self.sampler = PatchSampler(self.num_patches)

        # nce layers: comma-separated string or list
        nce_layers_raw = cut_opt.get('nce_layers', '0,4,8,12,16')
        if isinstance(nce_layers_raw, str):
            self.nce_layers = [int(x) for x in nce_layers_raw.split(',') if x.strip() != '']
        elif isinstance(nce_layers_raw, (list, tuple)):
            self.nce_layers = [int(x) for x in nce_layers_raw]
        else:
            self.nce_layers = [int(nce_layers_raw)]

        if self.use_simplified:
            self.nce_layers = []

        # NCE temperature and weight
        nce_T = float(cut_opt.get('nce_T', 0.07))
        self.lambda_NCE = float(cut_opt.get('lambda_NCE', 1.0)) if not self.use_simplified else 0.0

        if self.lambda_NCE > 0.0:
            # --- FIX: Removed num_patches from PatchNCELoss constructor for compatibility ---
            # The correct PatchNCELoss uses ALL sampled patches in the batch for the InfoNCE matrix.
            self.criterionNCE = PatchNCELoss(temperature=nce_T).to(self.device)
        else:
            self.criterionNCE = None

        # Losses
        self.criterionGAN = nn.MSELoss().to(self.device)     # LSGAN
        self.criterionIdt = nn.L1Loss().to(self.device)

        # Loss log names
        self.loss_names = []
        self.loss_names.extend(['G_GAN', 'D_real', 'D_fake'])
        if self.lambda_NCE > 0.0:
            self.loss_names.append('NCE')
        if self.use_simplified and float(self.opt.get('lambda_idt', 0.0)) > 0.0:
            self.loss_names.append('G_idt')
        self.loss_names.append('G_total')

        # Other hyperparams
        self.lambda_GAN = float(self.opt.get('lambda_GAN', 1.0))
        self.lambda_idt = float(self.opt.get('lambda_idt', 0.5)) if self.use_simplified else 0.0

        # Optimizers
        train_opt = self.opt.get('training', {}) if isinstance(self.opt.get('training', {}), dict) else {}
        lr = float(train_opt.get('lr', 2e-4))
        beta1 = float(train_opt.get('beta1', 0.5))
        beta2 = float(train_opt.get('beta2', 0.999))

        # Put networks on device
        self.netG.to(self.device)
        self.netD.to(self.device)
        self.netDegrade.to(self.device)

        # Optimizers:
        # include both netG and netDegrade parameters in G optimizer so they learn jointly.
        self.optimizer_G = torch.optim.Adam(list(self.netG.parameters()) + list(self.netDegrade.parameters()), lr=lr, betas=(beta1, beta2))
        self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=lr, betas=(beta1, beta2))

        # register optimizers into BaseModel (expected)
        self.optimizers = []
        self.optimizers.append(self.optimizer_G)
        self.optimizers.append(self.optimizer_D)

        # placeholders for input/output
        self.real_A = None
        self.real_B = None
        self.fake_B = None        # degraded LR output (sent to discriminator)
        self.fake_B_hr = None     # HR generator output (used for NCE)

        # visuals (names used for logging/visualization)
        self.visual_names = ['real_A', 'fake_B_hr', 'fake_B', 'real_B']

    # -------------------------
    # Required data API hooks (dataset loader must provide keys 'A' and 'B')
    # -------------------------
    def set_input(self, input):
        """
        input: dict with keys 'A', 'B' (and optionally 'A_paths', 'B_paths').
        Moves tensors to self.device.
        """
        direction = self.opt.get('direction', 'AtoB')
        AtoB = direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input.get('A_paths', None)

    def forward(self):
        """
        Forward pass produces both HR (fake_B_hr) and LR-degraded (fake_B) outputs.
        """
        # Generator output (HR)
        self.fake_B_hr = self.netG(self.real_A)  # keep same resolution as real_A
        # Degrade to LR-style output (may be lower spatial resolution)
        self.fake_B = self.netDegrade(self.fake_B_hr)
        return self.fake_B

    def test(self):
        """Run forward in eval mode (no gradients)"""
        with torch.no_grad():
            self.netG.eval()
            self.netDegrade.eval()
            self.netD.eval()
            self.forward()
            self.compute_visuals()

    def compute_visuals(self):
        self.visuals = {
            'real_A': self.real_A.detach() if self.real_A is not None else None,
            'fake_B_hr': self.fake_B_hr.detach() if self.fake_B_hr is not None else None,
            'fake_B': self.fake_B.detach() if self.fake_B is not None else None,
            'real_B': self.real_B.detach() if self.real_B is not None else None
        }

    # -------------------------
    # Feature extraction (walk generator's sequential model to pull intermediate features)
    # This assumes self.netG.model is a Sequential-like container with layers indexed by ints.
    # If your generator implementation differs, adapt this function accordingly.
    # -------------------------
    def _extract_generator_features(self, x, layer_ids):
        feats = {}
        current = x
        # attempt to iterate over self.netG.model; support attribute 'model' used earlier
        module_seq = None
        if hasattr(self.netG, 'model'):
            module_seq = self.netG.model
        elif isinstance(self.netG, nn.Sequential):
            module_seq = self.netG
        else:
            # try to find an ordered children list
            module_seq = list(self.netG.children())

        for idx, layer in enumerate(module_seq):
            current = layer(current)
            if idx in layer_ids:
                feats[idx] = current
        return feats

    # -------------------------
    # NCE calculation
    # -------------------------
    def calculate_NCE_loss(self, src, tgt):
        """
        src: real_A (LR input)
        tgt: fake_B_hr (HR output of generator)
        returns: averaged PatchNCE loss over selected layers
        """
        # prepare layer indices that exist in generator
        # find module length
        if hasattr(self.netG, 'model'):
            model_len = len(self.netG.model)
        else:
            try:
                model_len = len(list(self.netG.children()))
            except Exception:
                model_len = 1

        layer_ids = [i for i in self.nce_layers if i < model_len]
        if len(layer_ids) == 0:
            # Fallback to last layer if no specific layers are defined/found
            layer_ids = [model_len - 1] 

        # extract features
        feat_q_dict = self._extract_generator_features(tgt, layer_ids)  # fake / query
        feat_k_dict = self._extract_generator_features(src, layer_ids)  # real / key

        total_loss = 0.0
        for lid in layer_ids:
            fq = feat_q_dict[lid]  # B x C x H x W
            fk = feat_k_dict[lid]
            # sample patches
            q_samples = self.sampler.sample(fq)
            k_samples = self.sampler.sample(fk)
            # PatchNCELoss expects tensors shaped (B, C, num_patches)
            # The InfoNCE loss uses the B*num_patches elements as queries and keys
            loss_layer = self.criterionNCE(q_samples, k_samples)
            total_loss = total_loss + loss_layer

        total_loss = total_loss / float(len(layer_ids))
        return total_loss

    # -------------------------
    # GAN helpers (LSGAN)
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
        # store for logging
        self.loss_D_fake = loss_fake.item() if hasattr(loss_fake, 'item') else float(loss_fake)
        self.loss_D_real = loss_real.item() if hasattr(loss_real, 'item') else float(loss_real)
        return 0.5 * (loss_fake + loss_real)

    # -------------------------
    # Training step (single optimization step)
    # -------------------------
    def optimize_parameters(self):
        # forward -> produces self.fake_B_hr (HR) and self.fake_B (LR-style)
        self.forward()

        # ----- Update Discriminator -----
        self.netD.train()
        self.optimizer_D.zero_grad()
        loss_D = self._compute_D_loss(self.fake_B, self.real_B)
        loss_D.backward()
        self.optimizer_D.step()

        # ----- Update Generator (+ Degrader) -----
        self.netG.train()
        self.netDegrade.train()
        self.optimizer_G.zero_grad()

        # GAN loss computed on degraded output (fake_B)
        loss_G_gan_tensor = self._compute_G_gan_loss(self.fake_B) * self.lambda_GAN

        # Identity loss (optional simplified mode)
        # NOTE: This identity loss is an L1 loss, not the NCE Identity loss.
        if self.use_simplified and self.lambda_idt > 0.0:
            loss_idt_tensor = self.criterionIdt(self.fake_B, self.real_B) * self.lambda_idt
        else:
            loss_idt_tensor = torch.zeros(1, device=self.device)

        # PatchNCE loss computed between real_A (LR input) and fake_B_hr (HR output)
        if self.lambda_NCE > 0.0 and self.criterionNCE is not None:
            loss_NCE_tensor = self.calculate_NCE_loss(self.real_A, self.fake_B_hr) * self.lambda_NCE
        else:
            loss_NCE_tensor = torch.zeros(1, device=self.device)

        # total generator loss: GAN + NCE (+ idt if used)
        loss_G_total = loss_G_gan_tensor + loss_NCE_tensor + loss_idt_tensor

        # backprop and step
        loss_G_total.backward()
        self.optimizer_G.step()

        # Logging: convert tensors to python floats if possible
        self.loss_G_GAN = loss_G_gan_tensor.item() if hasattr(loss_G_gan_tensor, 'item') else float(loss_G_gan_tensor)
        self.loss_NCE = loss_NCE_tensor.item() if hasattr(loss_NCE_tensor, 'item') else float(loss_NCE_tensor)
        self.loss_idt = loss_idt_tensor.item() if hasattr(loss_idt_tensor, 'item') else float(loss_idt_tensor)
        self.loss_G_total = loss_G_total.item() if hasattr(loss_G_total, 'item') else float(loss_G_total)

    # -------------------------
    # Utility: save/load networks (delegated to BaseModel) - if BaseModel provides helpers they will be used
    # -------------------------
# End of CUT.py