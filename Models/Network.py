import torch 
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------
# Helper Block
# ----------------------------
class ResnetBlock(nn.Module):
    """Residual Block with two convolutional layers and skip connection."""
    def __init__(self, dim):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=0),
            nn.InstanceNorm2d(dim),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=0),
            nn.InstanceNorm2d(dim),
        )

    def forward(self, x):
        return x + self.conv_block(x)

# ----------------------------
# Generator
# ----------------------------
class ResnetGenerator(nn.Module):
    """
    ResNet-based Generator for image-to-image translation
    Produces output of fixed size 256x256
    """
    def __init__(self, input_nc=3, output_nc=3, ngf=64, n_blocks=9):
        super().__init__()
        self.skip1 = nn.Identity()
        self.skip2 = nn.Identity()

        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(inplace=True)
        ]

        # Downsampling
        in_dim = ngf
        out_dim = in_dim * 2
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(in_dim, out_dim, 4, stride=2, padding=1),
            nn.InstanceNorm2d(out_dim),
            nn.ReLU(inplace=True)
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(out_dim, out_dim // 2, 4, stride=2, padding=1),
            nn.InstanceNorm2d(out_dim // 2),
            nn.ReLU(inplace=True)
            )

        in_dim = out_dim
        out_dim = in_dim * 2

        # ResNet Blocks
        for _ in range(n_blocks):
            model += [ResnetBlock(in_dim)]

        # Upsampling
        out_dim = in_dim // 2
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(in_dim, out_dim, 4, stride=2, padding=1),
            nn.InstanceNorm2d(out_dim),
            nn.ReLU(inplace=True)
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(out_dim, out_dim // 2, 4, stride=2, padding=1),
            nn.InstanceNorm2d(out_dim // 2),
            nn.ReLU(inplace=True)
        )
        in_dim = out_dim
        out_dim = in_dim // 2

        # Output Layer
        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_dim, output_nc, kernel_size=7, padding=0),
            nn.Tanh()
        ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        e1 = self.model[:5](x)     # before first downsample
        e2 = self.model[5:9](e1)   # before second downsample

        x = self.model[9:](e2)     # resblocks

        x = self.up1(x)
        x = torch.cat([x, e2], dim=1)

        x = self.up2(x)
        x = torch.cat([x, e1], dim=1)

        return self.out_layer(x)

# ----------------------------
# PatchGAN Discriminator
# ----------------------------
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

# ----------------------------
# Weight Initialization
# ----------------------------
def init_weights(net, init_type='normal', init_gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and m.weight is not None:
            if classname.find('Conv') != -1 or classname.find('Linear') != -1:
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
        elif classname.find('BatchNorm2d') != -1:
            if m.weight is not None:
                nn.init.normal_(m.weight, 1.0, init_gain)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
    net.apply(init_func)

# ----------------------------
# Factory functions
# ----------------------------
def define_G(input_nc=3, output_nc=3, ngf=64, n_blocks=9):
    net = ResnetGenerator(input_nc, output_nc, ngf, n_blocks)
    init_weights(net)
    return net

def define_D(input_nc=3, ndf=64):
    net = PatchGANDiscriminator(input_nc, ndf)
    init_weights(net)
    return net
