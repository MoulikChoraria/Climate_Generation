import os
import torch
from torch import Tensor
import numpy as np
import xarray
import torchvision
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import torchvision.utils as vutils



class GAN(pl.LightningModule):
    
    def __init__(self, generator, discriminator, device):
        super().__init__
        self.G = generator
        self.D = discriminator
        self.automatic_optimization=False
        self.device = device
        self.latent_dim = self.G.input_dim
        self.validation_history = []
    
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("GAN")
        parser.add_argument("--data_path", type=str, default="")
        parser.add_argument("--learning_rate_G", type=float, default=1e-5)
        parser.add_argument("--learning_rate_D", type=float, default=1e-5)
        parser.add_argument("--D_iters", type=int, default=1)
        return parent_parser
    
    def configure_optimizers(self, lr_G = 1e-5, lr_D = 1e-5):
        g_opt = torch.optim.Adam(self.G.parameters(), lr_G)
        d_opt = torch.optim.Adam(self.D.parameters(), lr_D)
        return g_opt, d_opt

    def validation_step(self, batch, batch_idx):
        x, y = batch
        fake = self.G(x, y)
        preds = self.D(fake, y)
        self.validation_history.append((vutils.make_grid(fake, padding=2, normalize=True)[:1, :, :].squeeze(0), preds))
        plt.figure(figsize=(8,8))
        plt.imshow(self.validation_history[-1][0], cmap='gray')
        plt.show()
        plt.show()


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
        errD_real = self.criterion(d_output, real_label)

        # Train with all-fake batch
        noise = torch.randn(b_size, self.latent_dim, 1, 1, device=self.device)
        g_X = self.G(noise, real_label)
        d_output = self.D(g_X.detach(), real_label).view(-1)
        errD_fake = self.criterion(d_output, fake_label)#*self.class_imbalance
        
        # Compute error of D as sum over the fake and the real batches
        errD = errD_real + errD_fake
        d_opt.zero_grad()
        self.manual_backward(errD)
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


