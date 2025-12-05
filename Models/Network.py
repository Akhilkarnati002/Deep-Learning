import torch
import torch.nn as nn
import torch.nn.functional as F

# =======================================
# Residual Block
# =========================================
class ResnetBlock(nn.Module):
    """Residual block with two 3x3 convs and a skip connection."""
    def __init__(self, dim):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=0),
            nn.InstanceNorm2d(dim),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=0),
            nn.InstanceNorm2d(dim)
        )

    def forward(self, x):
        return x + self.conv_block(x)


# ===================================
# ResNet Generator Standard & Stable
# ======================================
class ResnetGenerator(nn.Module):
    """
    Standard ResNet-based generator used in CycleGAN/CUT.
    conv7x7 -> 2x downsample -> N resblocks -> 2x upsample -> conv7x7 -> Tanh
    """
    def __init__(self, input_nc=3, output_nc=3, ngf=64, n_blocks=9):
        """
        input_nc: number input channels (1 for grayscale)
        output_nc: number output channels
        ngf: base number of filters (usually 64)
        n_blocks: number of residual blocks (6 for 128x128, 9 for 256x256)
        """
        assert n_blocks >= 0
        super().__init__()

        # Initial conv (7x7)
        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, ngf, kernel_size=7, stride=1, padding=0),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(inplace=True)
        ]

        # Downsampling layers (2x)
        in_dim = ngf
        for _ in range(2):
            out_dim = in_dim * 2
            model += [
                nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=2, padding=1),
                nn.InstanceNorm2d(out_dim),
                nn.ReLU(inplace=True)
            ]
            in_dim = out_dim

        # Residual blocks
        for _ in range(n_blocks):
            model += [ResnetBlock(in_dim)]

        # Upsampling layers (2x) — use ConvTranspose2d to mirror downsampling
        for _ in range(2):
            out_dim = in_dim // 2
            model += [
                nn.ConvTranspose2d(in_dim, out_dim, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.InstanceNorm2d(out_dim),
                nn.ReLU(inplace=True)
            ]
            in_dim = out_dim

        # Output conv (7x7)
        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_dim, output_nc, kernel_size=7, stride=1, padding=0),
            nn.Tanh()
        ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


# ==========================================
# PatchGAN Discriminator 
# =============================================
class PatchGANDiscriminator(nn.Module):
    """70x70 PatchGAN Discriminator"""
    def __init__(self, input_nc=3, ndf=64):
        super().__init__()
        model = [
            nn.Conv2d(input_nc, ndf, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        ]
        in_dim = ndf
        out_dim = in_dim * 2
        for _ in range(2):
            model += [
                nn.Conv2d(in_dim, out_dim, kernel_size=4, stride=2, padding=1),
                nn.InstanceNorm2d(out_dim),
                nn.LeakyReLU(0.2, inplace=True)
            ]
            in_dim = out_dim
            out_dim = in_dim * 2
        model += [
            nn.Conv2d(in_dim, out_dim, kernel_size=4, stride=1, padding=1),
            nn.InstanceNorm2d(out_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_dim, 1, kernel_size=4, stride=1, padding=1)
        ]
        self.model = nn.Sequential(*model)

    def forward(self, input):
        return self.model(input)


# ================================
# Weights Initialization
# =======================================
def init_weights(net, init_type='normal', init_gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and m.weight is not None:
            if (classname.find('Conv') != -1) or (classname.find('Linear') != -1):
                if init_type == 'normal':
                    nn.init.normal_(m.weight, 0.0, init_gain)
                elif init_type == 'xavier':
                    nn.init.xavier_normal_(m.weight, gain=init_gain)
                elif init_type == 'kaiming':
                    nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    nn.init.orthogonal_(m.weight, gain=init_gain)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
        # InstanceNorm/BatchNorm init (if present)
        if classname.find('InstanceNorm') != -1 or classname.find('BatchNorm') != -1:
            if hasattr(m, 'weight') and m.weight is not None:
                nn.init.normal_(m.weight, 1.0, init_gain)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
    net.apply(init_func)



# Factory functions
def define_G(input_nc=3, output_nc=3, ngf=64, n_blocks=9):
    net = ResnetGenerator(input_nc, output_nc, ngf, n_blocks)
    init_weights(net)
    return net

def define_D(input_nc=3, ndf=64):
    net = PatchGANDiscriminator(input_nc, ndf)
    init_weights(net)
    return net
