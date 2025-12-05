import torch
import torch.nn as nn
import torch.nn.functional as F
from Models.BaseModel import BaseModel
from Models.Network import define_G, define_D
from Losses.NCE_losses import PatchNCELoss
print(">>> CUTTRAIN TOP LEVEL EXECUTED")



# Patch sampler
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
        sampled = torch.gather(
            feat_t,
            1,
            idx.unsqueeze(-1).expand(-1, -1, C)
        )                                            # B x num_patches x C
        sampled = sampled.permute(0, 2, 1)           # B x C x num_patches
        return sampled



# CUTModel
class CUTModel(BaseModel):
    """
    Contrastive Unpaired Translation Model (simplified) for LR -> HR:
      - netG: A (LR) -> fake_B (HR-like)
      - netD: PatchGAN on HR domain
      - NCE loss between features of (real_A, fake_B)

    Expects 'opt' to be a dict-like object with keys used below.
    """

    def __init__(self, opt):
        # Initialize BaseModel (sets self.opt, self.device, etc.)
        super().__init__(opt)

        # ---- Basic network params (safe defaults) ----
        in_nc = int(self.opt.get('input_nc', 1))   # LR input channels (grayscale -> 1)
        out_nc = int(self.opt.get('output_nc', 1)) # HR output channels
        ngf = int(self.opt.get('ngf', 64))
        ndf = int(self.opt.get('ndf', 64))
        self.use_simplified = bool(self.opt.get('use_simplified', False))
        self.lambda_idt = float(self.opt.get('lambda_idt', 0.0)) if self.use_simplified else 0.0

        # ---- Networks ----
        self.netG = define_G(in_nc, out_nc, ngf)
        self.netD = define_D(out_nc, ndf)

        # Model names used by BaseModel's save/load helpers
        self.model_names = ['G', 'D']

        # ---- CUT / NCE options ----
        cut_opt = self.opt.get('cut', {}) if isinstance(self.opt.get('cut', {}), dict) else {}
        self.num_patches = int(cut_opt.get('num_patches', 256))
        self.sampler = PatchSampler(self.num_patches)

        # nce_layers: list of layer indices in generator to use
        nce_layers_raw = cut_opt.get('nce_layers', '0,4,8,12,16')
        if isinstance(nce_layers_raw, str):
            self.nce_layers = [int(x) for x in nce_layers_raw.split(',') if x.strip() != '']
        elif isinstance(nce_layers_raw, (list, tuple)):
            self.nce_layers = [int(x) for x in nce_layers_raw]
        else:
            self.nce_layers = [int(nce_layers_raw)]

        if self.use_simplified:
            # disable NCE if using a simplified pure GAN setup
            self.nce_layers = []

        # NCE temperature & weight
        nce_T = float(cut_opt.get('nce_T', 0.07))
        self.lambda_NCE = float(cut_opt.get('lambda_NCE', 0.3)) if not self.use_simplified else 0.0

        if self.lambda_NCE > 0.0:
            # NOTE: We only pass temperature to PatchNCELoss for compatibility
            self.criterionNCE = PatchNCELoss(temperature=nce_T).to(self.device)
        else:
            self.criterionNCE = None


        self.criterionGAN = nn.MSELoss().to(self.device)   # LSGAN
        self.criterionIdt = nn.L1Loss().to(self.device)    # optional identity L1

        # Loss log names 
        self.loss_names = []
        self.loss_names.extend(['G_GAN', 'D_real', 'D_fake'])
        if self.lambda_NCE > 0.0:
            self.loss_names.append('NCE')
        if self.use_simplified and float(self.opt.get('lambda_idt', 0.0)) > 0.0:
            self.loss_names.append('G_idt')
        self.loss_names.append('G_total')

        # Hyperparameters
        self.lambda_GAN = float(self.opt.get('lambda_GAN', 1.0))
        self.lambda_idt = float(self.opt.get('lambda_idt', 0.0)) if self.use_simplified else 0.0

        # Training options
        train_opt = self.opt.get('training', {}) if isinstance(self.opt.get('training', {}), dict) else {}
        lr_G = float(train_opt.get('lr', 2e-4))
        lr_D = float(train_opt.get('lr_D', lr_G))   # use lr_D if provided
        beta1 = float(train_opt.get('beta1', 0.5))
        beta2 = float(train_opt.get('beta2', 0.999))

        #self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=lr_G, betas=(beta1, beta2))
        #self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=lr_D, betas=(beta1, beta2))


        # Move networks to device
        self.netG.to(self.device)
        self.netD.to(self.device)

        # Optimizers
        self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=lr_G, betas=(beta1, beta2))
        self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=lr_D, betas=(beta1, beta2))

        # Register optimizers for BaseModel
        self.optimizers = [self.optimizer_G, self.optimizer_D]

        # Placeholders for data
        self.real_A = None   # LR input
        self.real_B = None   # HR target
        self.fake_B = None   # generated HR-like image

        # Visuals (used by BaseModel.get_current_visuals)
        self.visual_names = ['real_A', 'fake_B', 'real_B']

    
    # Utility method for trainer to move networks
    def move_networks_to_device(self):
        """Explicitly move networks to self.device (called from trainer)."""
        self.netG.to(self.device)
        self.netD.to(self.device)
        if self.criterionNCE is not None:
            self.criterionNCE.to(self.device)
        self.criterionGAN.to(self.device)
        self.criterionIdt.to(self.device)

    
    # Required data API hooks
    def set_input(self, input):
        """
        input: dict with keys 'A', 'B' (and optionally 'A_paths', 'B_paths').
        A is LR, B is HR when direction='AtoB'.
        """
        direction = self.opt.get('direction', 'AtoB')
        AtoB = direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input.get('A_paths', None)

    def forward(self):
        """
        Forward pass: generate fake HR image from LR input
        """
        self.fake_B = self.netG(self.real_A)   # LR -> HR-like
        return self.fake_B

    def test(self):
        """Run forward in eval mode (no gradients)."""
        with torch.no_grad():
            self.netG.eval()
            self.netD.eval()
            self.forward()
            self.compute_visuals()

    def compute_visuals(self):
        """Prepare visuals dict (used by BaseModel.get_current_visuals)."""
        self.visuals = {
            'real_A': self.real_A.detach() if self.real_A is not None else None,
            'fake_B': self.fake_B.detach() if self.fake_B is not None else None,
            'real_B': self.real_B.detach() if self.real_B is not None else None,
        }

    
    # Feature extraction for NCE
    def _extract_generator_features(self, x, layer_ids):
        feats = {}
        current = x

        # Try common patterns for ResNet/G-style generators
        if hasattr(self.netG, 'model'):
            module_seq = self.netG.model
        elif isinstance(self.netG, nn.Sequential):
            module_seq = self.netG
        else:
            module_seq = list(self.netG.children())

        for idx, layer in enumerate(module_seq):
            current = layer(current)
            if idx in layer_ids:
                feats[idx] = current
        return feats

    def calculate_NCE_loss(self, src, tgt):
        """
        Compute PatchNCE loss between:
          src: real_A (LR input)
          tgt: fake_B (HR-like output)
        """
        # Determine how many layers we have
        if hasattr(self.netG, 'model'):
            model_len = len(self.netG.model)
        else:
            try:
                model_len = len(list(self.netG.children()))
            except Exception:
                model_len = 1

        layer_ids = [i for i in self.nce_layers if i < model_len]
        if len(layer_ids) == 0:
            # fallback: use last layer if nothing valid
            layer_ids = [model_len - 1]

        # Extract features
        feat_q_dict = self._extract_generator_features(tgt, layer_ids)  # fake / query
        feat_k_dict = self._extract_generator_features(src, layer_ids)  # real / key

        total_loss = 0.0
        for lid in layer_ids:
            fq = feat_q_dict[lid]  # B x C x H x W
            fk = feat_k_dict[lid]

            # Sample patches
            q_samples = self.sampler.sample(fq)
            k_samples = self.sampler.sample(fk)

            loss_layer = self.criterionNCE(q_samples, k_samples)
            total_loss = total_loss + loss_layer

        total_loss = total_loss / float(len(layer_ids))
        return total_loss

    
    # GAN helpers (LSGAN)
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

    
    # Training step
    def optimize_parameters(self):
        # Forward: real_A -> fake_B
        self.forward()

        # ----- Update Discriminator -----
        self.netD.train()
        self.optimizer_D.zero_grad()
        loss_D = self._compute_D_loss(self.fake_B, self.real_B)
        loss_D.backward()
        self.optimizer_D.step()

        # ----- Update Generator -----
        self.netG.train()
        self.optimizer_G.zero_grad()

        # GAN loss on fake_B vs real_B
        loss_G_gan_tensor = self._compute_G_gan_loss(self.fake_B) * self.lambda_GAN

        # Optional identity loss (not NCE-IDT, just L1(fake, real))
        if self.use_simplified and self.lambda_idt > 0.0:
            loss_idt_tensor = self.criterionIdt(self.fake_B, self.real_B) * self.lambda_idt
        else:
            loss_idt_tensor = torch.zeros(1, device=self.device)

        # PatchNCE loss between real_A (LR) and fake_B (HR-like)
        if self.lambda_NCE > 0.0 and self.criterionNCE is not None:
            loss_NCE_tensor = self.calculate_NCE_loss(self.real_A, self.fake_B) * self.lambda_NCE
        else:
            loss_NCE_tensor = torch.zeros(1, device=self.device)
        # L1 content loss to help fake_B match real_B pixel-wise (supervised guidance)      
        lambda_L1 = float(self.opt.get("lambda_L1", 0.0))   # will be set in config
        #loss_L1_tensor = torch.zeros(1, device=self.device)

        if lambda_L1 > 0.0:
            loss_L1_tensor = F.l1_loss(self.fake_B, self.real_B) * lambda_L1
        else:
            loss_L1_tensor = torch.zeros(1, device=self.device) 
        # Total generator loss
        loss_G_total = loss_G_gan_tensor + loss_NCE_tensor + loss_idt_tensor + loss_L1_tensor

        loss_G_total.backward()
        self.optimizer_G.step()

        # Logging fields (BaseModel expects loss_<name>)
        self.loss_G_GAN = loss_G_gan_tensor.item() if hasattr(loss_G_gan_tensor, 'item') else float(loss_G_gan_tensor)
        self.loss_NCE = loss_NCE_tensor.item() if hasattr(loss_NCE_tensor, 'item') else float(loss_NCE_tensor)
        self.loss_G_idt = loss_idt_tensor.item() if hasattr(loss_idt_tensor, 'item') else float(loss_idt_tensor)
        self.loss_G_total = loss_G_total.item() if hasattr(loss_G_total, 'item') else float(loss_G_total)
        self.loss_L1 = loss_L1_tensor.item() if hasattr(loss_L1_tensor, 'item') else float(loss_L1_tensor)

