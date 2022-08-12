import os
import torch
from torch import Tensor
import numpy as np
import xarray
import torchvision
import pytorch_lightning as pl


class GAN(pl.LightningModule):
    
    def __init__(self, generator, discriminator, device):
        super().__init__
        self.G = generator
        self.D = discriminator
        self.automatic_optimization=False
        self.device = device

    def sample_z(self, n) -> Tensor:
        sample = self._Z.sample((n,))
        return sample

    def sample_G(self, n, real_label) -> Tensor:
        z = self.sample_z(n)
        return self.G(z, real_label)
    
    def configure_optimizers(self):
        g_opt = torch.optim.Adam(self.G.parameters(), lr=1e-5)
        d_opt = torch.optim.Adam(self.D.parameters(), lr=1e-5)
        return g_opt, d_opt

    def training_step(self, batch, batch_idx):
        
        g_opt, d_opt = self.optimizers()
        
        # Format batch
        real_img = batch[0].to(self.device)
        real_label = batch[1].to(self.device)
        b_size = real_img.size(0)


        ### fake_label = 0

        fake_label = torch.full((b_size,), 0, dtype=torch.float, device=self.device)
        # Forward pass real batch through D
        d_output = self.D(real_img, real_label).view(-1)
        # Calculate loss on all-real batch
        errD_real = self.criterion(d_output, real_label)
        # Calculate gradients for D in backward pass        

        ## Train with all-fake batch
        # Generate fake batch from G
        g_X = self.sample_G(b_size, real_label)

        # Classify all fake batch with D
        d_output = self.D(g_X.detach(), real_label).view(-1)
        # Calculate D's loss on the all-fake batch
        errD_fake = self.criterion(d_output, fake_label)#*self.class_imbalance
        # Calculate the gradients for this batch, accumulated (summed) with previous gradients
        
        # Compute error of D as sum over the fake and the real batches
        errD = errD_real + errD_fake
        d_opt.zero_grad()
        self.manual_backward(errD)
        # Update D
        d_opt.step()

        self.log_dict({"g_loss": errG, "d_loss": errD}, prog_bar=True)

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        # Calculate G's loss based on D's output
        errG = self.criterion(self.D(g_X, real_label).view(-1), fake_label)
        # Calculate gradients for G
        g_opt.zero_grad()
        self.manual_backward(errG)
        # Update G
        g_opt.step()



def train_model(in_features):

    generator = torch.nn.Sequential([torch.nn.Linear()])
