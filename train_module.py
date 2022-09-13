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
    
    def __init__(self, generator, discriminator, lr_G, lr_D, run_name):
        super(GAN, self).__init__()
        self.G = generator
        self.D = discriminator
        self.automatic_optimization=False
        self.G.apply(self.weights_init)
        self.D.apply(self.weights_init)
        #self.device = device
        self.latent_dim = self.G.input_dim
        self.validation_history = []
        self.save_hyperparameters("lr_G", "lr_D", "run_name")
    
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("GAN")
        #parser.add_argument("--data_path", type=str, default="")
        parser.add_argument("--learning_rate_G", type=float, default=1e-5)
        parser.add_argument("--learning_rate_D", type=float, default=1e-5)
        #parser.add_argument("--D_iters", type=int, default=1)
        return parent_parser
    
    def configure_optimizers(self):
        g_opt = torch.optim.Adam(self.G.parameters(), self.hparams.lr_G)
        d_opt = torch.optim.Adam(self.D.parameters(), self.hparams.lr_D)
        return g_opt, d_opt

    def validation_step(self, batch, batch_idx):
        #x, y = batch[0].to(self.device), batch[1].to(self.device)
        x, y = batch[0], batch[1]
        #print(x.size(), y.size())
        fake = self.G(x, y)
        preds = self.D(fake, y)
        self.validation_history.append((vutils.make_grid(fake, padding=2, normalize=True)[:1, :, :].squeeze(0), preds.detach().cpu().numpy()))
        #print(y, preds)
        plt.figure(figsize=(8,8))
        plt.imshow(self.validation_history[-1][0].detach().cpu().numpy())#, cmap='gray')
        plt.savefig('/home/moulikc2/expose/Climate Generation/validation_figs/'\
                        +self.hparams.run_name+'/epoch_'+str(self.current_epoch)+"_batch_"+str(batch_idx)+'.png')
        #plt.show()
        #plt.show()
    
    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
            if m.bias is not None:
                m.bias.data.fill_(0.01)
        elif classname.find('BatchNorm') != -1:
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(m.bias.data, 0)
        elif classname.find('Linear') != -1:
            torch.nn.init.normal_(m.weight.data, 0.0, 0.01)
            if m.bias is not None:
                m.bias.data.fill_(0.01)

    def criterion(self, y_hat, y):
        loss = torch.nn.BCEWithLogitsLoss()
        return loss(y_hat, y.float())

    def training_step(self, batch, batch_idx):
        
        g_opt, d_opt = self.optimizers()
    

        # Format batch
        real_img = batch[0]
        real_label = batch[1]
        b_size = real_img.size(0)

        #print("yo1")
        ### fake_label = 0
        fake_label = torch.full((b_size,), 0, dtype=torch.float)
        fake_label = fake_label.type_as(real_label)

        # Forward pass real batch through D
        d_output = self.D(real_img, real_label).view(-1)
        #print("d_step")

        errD_real = self.criterion(d_output, real_label)
        #print("yo2")
        # Train with all-fake batch
        noise = torch.randn((b_size, self.latent_dim, 1, 1))
        noise = noise.type_as(real_img)
        g_X = self.G(noise, real_label)
        #print("g_step")

        d_output = self.D(g_X.detach(), real_label).view(-1)
        errD_fake = self.criterion(d_output, fake_label)#*self.class_imbalance
        #print("yo3")
        # Compute error of D as sum over the fake and the real batches
        errD = errD_real + errD_fake
        d_opt.zero_grad()
        self.manual_backward(errD)
        d_opt.step()
        #print("yo4")

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        # Calculate G's loss based on D's output
        errG = self.criterion(self.D(g_X, real_label).view(-1), fake_label)
        # Calculate gradients for G
        g_opt.zero_grad()
        self.manual_backward(errG)
        self.log_dict({"g_loss": errG, "d_loss": errD}, prog_bar=True)

        # Update G
        g_opt.step()
        #print("finished one step")


