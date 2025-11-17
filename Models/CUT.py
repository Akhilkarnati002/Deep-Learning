import torch 
import torch.nn as nn
import torch.nn.functional as F
from Models.Network import define_G, define_D
from Losses.NCE_losses import PatchNCELoss





class CUTModel(nn.Module):
    """
    CUT Model for image-to-image translation using contrastive learning.
    """
    def  __init__(self, input_nc=1, output_nc=1, ngf=64, nce_layers=('layers3',), lambda_gan=1.0, lambda_nce=1.0):
        super().__init__()

        #Networks
        self.netG = define_G(input_nc,output_nc,ngf)
        self.netD = define_D(output_nc, ndf)

        #CUT Model Needs Feature Enocder = resuse G up to certain layers
        self.nce_layers = nce_layers
        self.nce_loss_fn = PatchNCELoss()

        self.lambda_gan = lambda_gan
        self.lambda_nce = lambda_nce


        #Optimizers will be defined in Trainer class
        self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=0.0002, betas=(0.5, 0.999))


    #Feature Extraction from Generator (CUT STYLE)

    def extract_features(self, x):
        """
        Extract features from specified layers of the generator.
        """
        features = {}
        current = x
        for idx, layer in enumerate(self.netG.model):
            current = layer(current)

            """
                Example Layers to extract features from:

                layer 3 = after some Downsampling
                layer 6 = after some Downsampling 


                B = batch Size
                C = Channels
                H = Height
                W = Width
            """

            name = f'layers{idx+1}'
            if name in self.nce_layers:
                B,C,H,W= current.shape
                features[name]=current.reshape(B,C,H*W)  # [B,C,HW]
        return features        
    
    
    #Forward Pass 
    def forward(self, x):
        return self.netG(x)
    

    # GAN LOSS
    def gan_loss_G(self, fake):
        pred_fake = self.netD (fake)
        return F.mse_loss(pred_fake, torch.ones_like(pred_fake))    

    def  gan_loss_D(self,fake, real):

        pred_real = self.netD (real)
        pred_fake = self.netD (fake.detach())
        loss_real = F.mse_loss(pred_real, torch.ones_like(pred_real))
        loss_fake = F.mse_loss(pred_fake, torch.zeros_like(pred_fake))

        return (loss_real + loss_fake) *0.5



    # NCE LOSS 
    def nce_loss(self,real,fake):
        real_feats= self.extract_features(real)
        fake_feats= self.extract_features(fake)

        nce_total =0.0
        for layer in self.nce_layers:
            q= fake_feats[layer]
            p= real_feats[layer]
            nce_total += self.nce_loss_fn(q,p)
        return nce_total / len(self.nce_layers)
    


    # Generator Update

    def optimize_G(self,real):
        fake = self.netG(real)
        loss_gan= self.gan_loss_G(fake)
        loss_nce= self.nce_loss(real,fake)

        loss_G = self.lambda_gan * loss_gan + self.lambda_nce * loss_nce
        self.optimizer_G.zero_grad()
        loss_G.backward()
        self.optimizer_G.step()

        return{
            "G_GAN" : loss_gan.item(),
            "G_NCE" : loss_nce.item(),
            "G_total": loss_G.item()
        }
    
    # Discriminator Update
    def optimize_D(self, real):
        fake = self.netG(real).detach()
        
        loss_D = self.gan_loss_D(fake, real)
        self.optimizer_D.zero_grad()
        loss_D.backward()
        self.optimizer_D.step()

        return {
            "D_total": loss_D.item()
        }
