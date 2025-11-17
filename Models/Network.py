import torch 
import torch.nn as nn
import torch.nn.functional as F



# Helper Block

class ResentBlock(nn.Module):
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
        return x + self.conv_block(x)  # skip connection
    

# Generator Network
#
class ResentGenerator(nn.Module):
    """
    ResNet-based Generator Network for image-to-image translation.
    """
    def __init__(self, input_nc=3, output_nc=3, ngf=64, n_blocks=9):
        super().__init__()

        model =[
                nn.ReflectionPad2d(3),
                nn.Conv2d(input_nc,ngf,kernel_size=7,padding=0),
                nn.InstanceNorm2d(ngf),
                nn.ReLU(inplace=True)
                
                ]    
        

        # Downsampling
        in_dim =ngf
        out_dim = in_dim *2
        for _ in range(2):
            model += [
                nn.Conv2d(in_dim,out_dim,kernel_size=3,stride=2,padding=1),
                nn.InstanceNorm2d(out_dim),
                nn.ReLU(inplace=True)
            ]
            in_dim = out_dim
            out_dim = in_dim *2

        #ResNet Blocks
        for _ in range(n_blocks):
            model += [ResentBlock(in_dim)]

        # Upsampling
        #
        out_dim = in_dim //2
        for _ in range(2):
            model += [
                nn.ConvTranspose2d(in_dim,out_dim,kernel_size=3,stride=2,padding=1,output_padding=1),
                nn.InstanceNorm2d(out_dim),
                nn.ReLU(inplace=True)
            ]
            in_dim = out_dim
            out_dim = in_dim //2   

        #Output Layer
        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_dim,output_nc,kernel_size=7,padding=0),
            nn.Tanh()
        ]
        self.model = nn.Sequential(*model)

    def forward(self,input):
        return self.model(input)
    

    # PATCH GAND Discriminator

class PatchGANDiscriminator(nn.Module):
    """
    70x70 PatchGAN Discriminator Network.
    """
    def __init__(self, input_nc=3, ndf=64):
        super().__init__()


        model = [
            nn.Conv2d(input_nc, ndf, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        ]




        in_dim = ndf
        out_dim = in_dim * 2

        for n in range(3):

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
            nn.LeakyReLU(0.2, inplace=True)
        ]

        model += [
            nn.Conv2d(out_dim, 1, kernel_size=4, stride=1, padding=1)
        ]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        return self.model(input)

# Weight Initialization Function

def init_weights(net, init_type='normal', init_gain=0.02):
    """
    Initialize network weights.
    """
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                nn.init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError(f'initialization method [{init_type}] is not implemented')
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)

        elif classname.find('BatchNorm2d') != -1:
            nn.init.normal_(m.weight.data, 1.0, init_gain)
            nn.init.constant_(m.bias.data, 0.0)

    net.apply(init_func)


 # Factory Functions
 #


def define_G(input_nc=3, output_nc=3, ngf=64, n_blocks=9):
    net = ResentGenerator(input_nc, output_nc, ngf, n_blocks)
    init_weights(net)
    return net

def define_D(input_nc=3, ndf=64):
    net = PatchGANDiscriminator(input_nc, ndf)
    init_weights(net)
    return net