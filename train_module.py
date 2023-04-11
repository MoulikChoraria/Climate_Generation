import torch
from torch import Tensor
import numpy as np
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import torchvision.utils as vutils
from collections import OrderedDict
import numpy as np
import seaborn as sns
import torch.nn as nn





class GAN(pl.LightningModule):
    
    def __init__(self, generator, discriminator, lr_G, lr_D, run_name, val_epoch, G_iter):
        super(GAN, self).__init__()
        self.G = generator
        self.D = discriminator
        self.automatic_optimization=False
        self.n_classes = self.G.n_classes
        #self.G.apply(self.weights_init)
        #self.D.apply(self.weights_init)
        self.eval_loader = None
        self.test_loader = None

        #self.device = device
        self.latent_dim = self.G.input_dim
        self.validation_history = []
        self.D_training_losses = []
        self.G_training_losses = []
        self.G_validation_losses = []

        self.D_epoch_training_losses = []
        self.G_epoch_training_losses = []
        self.G_epoch_validation_losses = []
        self.save_hyperparameters("lr_G", "lr_D", "run_name", "val_epoch", "G_iter")
    
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("GAN")
        #parser.add_argument("--data_path", type=str, default="")
        parser.add_argument("--learning_rate_G", type=float, default=1e-4)
        parser.add_argument("--learning_rate_D", type=float, default=1e-4)
        #parser.add_argument("--D_iters", type=int, default=1)
        return parent_parser
    
    def configure_optimizers(self):
        #g_opt = torch.optim.RMSprop(self.G.parameters(), self.hparams.lr_G)
        #d_opt = torch.optim.RMSprop(self.D.parameters(), self.hparams.lr_D)
        #g_opt = torch.optim.Adam(self.G.parameters(), self.hparams.lr_G, betas=(0.0, 0.9))
        #d_opt = torch.optim.Adam(self.D.parameters(), self.hparams.lr_D, betas=(0.0, 0.9))
        g_opt = torch.optim.Adam(self.G.parameters(), self.hparams.lr_G, betas=(0.5, 0.99))
        d_opt = torch.optim.Adam(self.D.parameters(), self.hparams.lr_D, betas=(0.5, 0.99))
        return g_opt, d_opt

    def validation_step(self, batch, batch_idx):
        #x, y = batch[0].to(self.device), batch[1].to(self.device)
        x, categs = batch[0], batch[1]
        if(batch_idx == 0):
            eval_batch = next(iter(self.eval_loader))
            fixed_noise, fixed_labels = eval_batch[0], eval_batch[1]
            fixed_noise = fixed_noise.type_as(x)
            fixed_labels = fixed_labels.type(torch.LongTensor).type_as(categs)
            #print(x.size(), y.size())
            fake = self.G(fixed_noise, fixed_labels)
            preds = self.D(fake, fixed_labels)

            self.validation_history.append((vutils.make_grid(fake, padding=2, normalize=True)[:1, :, :].squeeze(0), preds.detach().cpu().numpy()))
            #print(y, preds)
            plt.figure(figsize=(8,8))
            plt.imshow(self.validation_history[-1][0].detach().cpu().numpy())#, cmap='gray')
            plt.savefig('/home/moulikc2/expose/Climate Generation/validation_figs/'\
                            +self.hparams.run_name+'/epoch_'+str(self.current_epoch)+"_batch_"+str(batch_idx)+'.png')
            plt.close()
            
            if(len(self.G_epoch_validation_losses) > 0):
                self.G_validation_losses.append(np.mean(self.G_epoch_validation_losses))
                self.G_epoch_validation_losses = []


            if(self.current_epoch > 10):
                plt.figure(figsize=(8,8))
                plt.plot([i for i in range(len(self.G_validation_losses))], self.G_validation_losses)#, cmap='gray')
                plt.savefig('/home/moulikc2/expose/Climate Generation/loss_curves/'\
                                +self.hparams.run_name+'/G_val_epoch_'+str(self.current_epoch)+"_batch_"+str(batch_idx)+'.png')
                plt.close()

        x, categs = batch[0], batch[1]
        b_size = x.size(0) 
        noise = torch.randn((b_size, self.latent_dim, 1, 1))
        noise = noise.type_as(x)
        g_X = self.G(noise, categs)
        real_label = torch.full((b_size,), 1, dtype=torch.float)
        real_label = real_label.type_as(categs)
        #print("g_step")        

        errG = self.criterion(self.D(g_X, categs).view(-1), real_label)
        self.G_epoch_validation_losses.append(errG.item())

       
            #self.test_loader

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
        #loss = torch.nn.BCELoss()
        return loss(y_hat, y.float())

    def training_step(self, batch, batch_idx):
        
        g_opt, d_opt = self.optimizers()

        if(batch_idx==0):

            if(len(self.D_epoch_training_losses) > 0):
                #self.D_training_losses.append(np.mean(self.D_epoch_training_losses))
                self.D_training_losses += self.D_epoch_training_losses
                self.D_epoch_training_losses = []

                #self.G_training_losses.append(np.mean(self.G_epoch_training_losses))
                self.G_training_losses += self.G_epoch_training_losses
                self.G_epoch_training_losses = []

            if((self.current_epoch+1)%self.hparams.val_epoch == 0):
                plt.figure(figsize=(8,8))
                #plt.plot([i for i in range(self.current_epoch)], self.G_training_losses, label='G_train_loss')#, cmap='gray')
                #plt.plot([i for i in range(self.current_epoch)], self.D_training_losses, label='D_train_loss')
                plt.plot(np.arange(len(self.G_training_losses)), self.G_training_losses, label='G_train_loss')#, cmap='gray')
                plt.plot(np.arange(len(self.D_training_losses)), self.D_training_losses, label='D_train_loss')
                plt.legend()
                plt.savefig('/home/moulikc2/expose/Climate Generation/loss_curves/'\
                                +self.hparams.run_name+'/train_loss_epoch_'+str(self.current_epoch)+"_batch_"+str(batch_idx)+'.png')
                plt.close()
    

        # Format batch
        real_img = batch[0]
        categs = batch[1]
        b_size = real_img.size(0)

        #print("yo1")
        ### fake_label = 0
        if(batch_idx==0 and self.current_epoch == 0):
            #self.k = 0.1
            self.random_tensor = real_img
            self.random_categs = categs
        #elif(batch_idx==0 and self.current_epoch > 20):
            #self.k = 0.02
        k = 0.1
        fake_label = torch.full((b_size,), 0, dtype=torch.float)

        ### add noise on conditional variable
        noise = torch.Tensor(np.random.choice([1, 0], size=b_size, p=[k, 1-k])).type_as(categs)
        noise = torch.where(categs>0, noise, 0)
        categs = categs-noise
        #fake_label+=nums

        fake_label = fake_label.type_as(categs)

        real_label = torch.full((b_size,), 1, dtype=torch.float)
        #nums = torch.Tensor(np.random.choice([0, 1], size=b_size, p=[k, 1-k]))
        #real_label = torch.where(nums > 0, 1, 0)
        #real_label*=nums
        real_label = real_label.type_as(categs)

        # Forward pass real batch through D
        self.D.zero_grad()
        d_output = self.D(real_img, categs).view(-1)
        #print("real:",real_img.size())
        #print("d_step")

        errD_real = self.criterion(d_output, real_label)
        #print("yo2")
        # Train with all-fake batch
        noise = torch.randn((b_size, self.latent_dim, 1, 1))
        noise = noise.type_as(real_img)
        g_X = self.G(noise, categs)
        #print("fake:", g_X.size())
        #print("g_step")

        d_output = self.D(g_X.detach(), categs).view(-1)
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
        #self.G.zero_grad()
        real_label = torch.full((b_size,), 1, dtype=torch.float)
        real_label = real_label.type_as(categs)
        G_losses_temp = []
        g_opt.zero_grad()
        for i in range(self.hparams.G_iter):
          g_X = self.G(noise, categs)
          noise = torch.randn((b_size, self.latent_dim, 1, 1))
          noise = noise.type_as(real_img)
          errG = self.criterion(self.D(g_X, categs).view(-1), real_label)
          self.manual_backward(errG)
          g_opt.step()
          G_losses_temp.append(errG.item())

        self.log_dict({"g_loss": errG, "d_loss": errD}, prog_bar=True)
        self.G_epoch_training_losses.append(np.mean(G_losses_temp))
        self.D_epoch_training_losses.append(errD.item())

    def on_train_epoch_end(self):
    ### testing histogram
        
        if(((self.current_epoch+1)%self.hparams.val_epoch == 0) or self.current_epoch==0):
            rainfall_stats_gan = []
            rainfall_stats_true = []
            
            for i in range(5):
                eval_batch = next(iter(self.test_loader))
                imgs, labels = eval_batch[0], eval_batch[1]
                b_size = imgs.size(0)
                noise = torch.randn((b_size, self.latent_dim, 1, 1))
                noise = noise.type_as(self.random_tensor)
                categs = labels.type(torch.LongTensor).type_as(self.random_categs)
                #print(x.size(), y.size())
                fake = self.G(noise, categs)
                for j in range(imgs.size(0)):
                    rainfall_stats_gan.append(torch.mean(fake[j]).item())
                    rainfall_stats_true.append(torch.mean(imgs[j]).item())

            plt.figure(figsize=(8,8))
            sns.set_style('whitegrid')
            sns.kdeplot(np.array(rainfall_stats_gan), label='GAN samples')
            sns.kdeplot(np.array(rainfall_stats_true), label='True samples')
            plt.xlabel('rainfall')
            plt.ylabel('density')
            plt.legend()
            plt.savefig('/home/moulikc2/expose/Climate Generation/test_figs/'\
                                +self.hparams.run_name+'/test_dist_'+str(self.current_epoch)+'.png')
            plt.tight_layout()
            plt.close()
            sns.reset_orig()

        # errG = self.criterion(self.D(g_X, categs).view(-1), real_label)
        # # Calculate gradients for G
        # g_opt.zero_grad()
        # self.manual_backward(errG)
        # self.log_dict({"g_loss": errG, "d_loss": errD}, prog_bar=True)
        # self.G_epoch_training_losses.append(errG.item())
        # self.D_epoch_training_losses.append(errD.item())

        # # Update G
        # g_opt.step()
        #print("finished one step")



class WGAN(pl.LightningModule):
    
    def __init__(self, generator, discriminator, lr_G, lr_D, run_name, val_epoch, n_critic, lambda_gp):
        super(WGAN, self).__init__()
        self.G = generator
        self.D = discriminator
        self.automatic_optimization=False
        self.G.apply(self.weights_init)
        self.D.apply(self.weights_init)
        self.eval_loader = None
        #self.device = device
        self.latent_dim = self.G.input_dim
        self.validation_history = []
        self.D_training_losses = []
        self.G_training_losses = []
        self.G_validation_losses = []

        self.D_epoch_training_losses = []
        self.G_epoch_training_losses = []
        self.G_epoch_validation_losses = []
        self.save_hyperparameters("lr_G", "lr_D", "run_name", "val_epoch", "n_critic", "lambda_gp")
        self.automatic_optimization = False

    
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("GAN")
        #parser.add_argument("--data_path", type=str, default="")
        parser.add_argument("--learning_rate_G", type=float, default=2e-5)
        parser.add_argument("--learning_rate_D", type=float, default=2e-5)
        #parser.add_argument("--n_critic", type=int, default=5)
        #parser.add_argument("--lambda_gp", type=float, default=10.0)
        #parser.add_argument("--learning_rate_D", type=float, default=1e-4)
        #parser.add_argument("--D_iters", type=int, default=1)
        return parent_parser


    def validation_step(self, batch, batch_idx):
        #x, y = batch[0].to(self.device), batch[1].to(self.device)
        x, categs = batch[0], batch[1]
        if(batch_idx == 0):
            eval_batch = next(iter(self.eval_loader))
            fixed_noise, fixed_labels = eval_batch[0], eval_batch[1]
            fixed_noise = fixed_noise.type_as(x)
            fixed_labels = fixed_labels.type(torch.LongTensor).type_as(categs)
            #print(x.size(), y.size())
            fake = self.G(fixed_noise, fixed_labels)
            #preds = self.D(fake, fixed_labels)

            self.validation_history.append(vutils.make_grid(fake, padding=2, normalize=True)[:1, :, :].squeeze(0))
            #print(y, preds)

            
            if(len(self.G_epoch_validation_losses) > 0):
                self.G_validation_losses.append(np.mean(self.G_epoch_validation_losses))
                self.G_epoch_validation_losses = []


            if((self.current_epoch+1) % self.hparams.val_epoch==0):
                plt.figure(figsize=(8,8))
                plt.imshow(self.validation_history[-1].detach().cpu().numpy())#, cmap='gray')
                plt.savefig('/home/moulikc2/expose/Climate Generation/validation_figs/'\
                                +self.hparams.run_name+'/epoch_'+str(self.current_epoch)+'.png')
                plt.close()

                plt.figure(figsize=(8,8))
                plt.plot([i for i in range(len(self.G_validation_losses))], self.G_validation_losses, marker='o')#, cmap='gray')
                plt.savefig('/home/moulikc2/expose/Climate Generation/loss_curves/'\
                                +self.hparams.run_name+'/G_val_epoch_'+str(self.current_epoch)+'.png')
                plt.close()

        x, categs = batch[0], batch[1]
        b_size = x.size(0) 
        noise = torch.randn((b_size, self.latent_dim, 1, 1))
        noise = noise.type_as(x)
        g_X = self.G(noise, categs)
        #print("g_step")        

        g_loss = -torch.mean(self.D(g_X, categs))
        self.G_epoch_validation_losses.append(g_loss.item())

        #plt.show()
        #plt.show()
    
    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
            if m.bias is not None:
                m.bias.data.fill_(0.01)
        elif classname.find('LayerNorm') != -1:
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(m.bias.data, 0)
        elif classname.find('Linear') != -1:
            torch.nn.init.normal_(m.weight.data, 0.0, 0.01)
            if m.bias is not None:
                m.bias.data.fill_(0.01)

    def compute_gradient_penalty(self, real_samples, fake_samples, categs):
        """Calculates the gradient penalty loss for WGAN GP"""
        # Random weight term for interpolation between real and fake samples
        alpha = torch.Tensor(np.random.random((real_samples.size(0), 1, 1, 1))).type_as(real_samples)
        # Get random interpolation between real and fake samples
        interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
        interpolates = interpolates.type_as(real_samples)
        d_interpolates = self.D(interpolates, categs)
        fake = torch.Tensor(real_samples.shape[0], 1).fill_(1.0).type_as(real_samples)
        # Get gradient w.r.t. interpolates
        gradients = torch.autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=fake,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        gradients = gradients.view(gradients.size(0), -1).to(self.device)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty

    def training_step(self, batch, batch_idx):
        g_opt, d_opt = self.optimizers()

        imgs, categs = batch
        lambda_gp = self.hparams.lambda_gp
        
        d_running_loss = []
        d_opt.zero_grad()
        
        if(self.current_epoch < 5):
            n_critic = 1
        else:
            n_critic = self.hparams.n_critic

        if(batch_idx == 0 and (self.current_epoch+1) % self.hparams.val_epoch==0):

            #print(y, preds)
            grid = vutils.make_grid(imgs, padding=2, normalize=True)[:1, :, :].squeeze(0)
            plt.figure(figsize=(8,8))
            plt.imshow(grid.detach().cpu().numpy())#, cmap='gray')
            plt.savefig('/home/moulikc2/expose/Climate Generation/validation_figs/'\
                            +self.hparams.run_name+'/train_samples_epoch_'+str(self.current_epoch)+'.png')
            plt.close()

        for i in range(n_critic):
            
            z = torch.randn(imgs.shape[0], self.latent_dim, 1, 1)
            z = z.type_as(imgs)
            fake_imgs = self.G(z, categs)

            # Real images
            real_validity = self.D(imgs, categs)
            # Fake images
            fake_validity = self.D(fake_imgs, categs)
            # Gradient penalty
            gradient_penalty = self.compute_gradient_penalty(imgs.data, fake_imgs.data, categs.clone())
            # Adversarial loss
            d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + lambda_gp * gradient_penalty
            self.manual_backward(d_loss)
            d_opt.step()
            
            d_running_loss.append(d_loss)


        # ground truth result (ie: all fake)
        # put on GPU because we created this tensor inside training_loop
        z = torch.randn(imgs.shape[0], self.latent_dim, 1, 1)
        z = z.type_as(imgs)

        g_opt.zero_grad()

        g_loss = -torch.mean(self.D(self.G(z, categs), categs))
        self.manual_backward(g_loss)
        g_opt.step()
       
        self.log_dict({"g_loss": g_loss, "d_loss": d_loss}, prog_bar=True)
        self.G_epoch_training_losses.append(g_loss.item())
        self.D_epoch_training_losses.append(d_loss.item())

            

    def configure_optimizers(self):
        #n_critic = self.hparams.n_critic


        #lr = self.lr
        b1 = 0.5
        b2 = 0.999

        opt_g = torch.optim.Adam(self.G.parameters(), lr=self.hparams.lr_G, betas=(b1, b2))
        opt_d = torch.optim.Adam(self.D.parameters(), lr=self.hparams.lr_D, betas=(b1, b2))
        # return (
        #     {'optimizer': opt_g, 'frequency': 1},
        #     {'optimizer': opt_d, 'frequency': n_critic}
        # )

        return opt_g, opt_d

    def on_train_epoch_end(self):
        
        D_mean_losses = []
        i = 0

        #while(i < len(self.D_epoch_training_losses)):
            #mean_iter_loss = np.mean(self.D_epoch_training_losses[i:i+self.hparams.n_critic])
            #D_mean_losses.append(mean_iter_loss)
            #i += self.hparams.n_critic

        #self.D_training_losses += D_mean_losses
        self.D_training_losses += self.D_epoch_training_losses
        self.D_epoch_training_losses = []

        #self.G_training_losses.append(np.mean(self.G_epoch_training_losses))
        self.G_training_losses += self.G_epoch_training_losses
        self.G_epoch_training_losses = []

        if ((self.current_epoch+1)%self.hparams.val_epoch == 0):
            plt.figure(figsize=(8,8))
            #plt.plot([i for i in range(self.current_epoch)], self.G_training_losses, label='G_train_loss')#, cmap='gray')
            #plt.plot([i for i in range(self.current_epoch)], self.D_training_losses, label='D_train_loss')
            plt.plot([i for i in range(len(self.G_training_losses))], self.G_training_losses, label='G_train_loss')#, cmap='gray')
            plt.plot([i for i in range(len(self.D_training_losses))], self.D_training_losses, label='D_train_loss')
            plt.legend()
            plt.savefig('/home/moulikc2/expose/Climate Generation/loss_curves/'\
                            +self.hparams.run_name+'/train_loss_epoch_'+str(self.current_epoch)+'.png')
            plt.close()
        



class AuxGAN(pl.LightningModule):
    
    def __init__(self, generator, discriminator, lr_G, lr_D, run_name, val_epoch, G_iter):
        super(AuxGAN, self).__init__()
        self.G = generator
        self.D = discriminator
        self.automatic_optimization=False
        self.G.apply(self.weights_init)
        self.D.apply(self.weights_init)
        self.eval_loader = None
        #self.device = device
        self.latent_dim = self.G.input_dim
        self.validation_history = []
        self.D_training_losses = []
        self.G_training_losses = []
        self.G_validation_losses = []

        self.D_epoch_training_losses = []
        self.G_epoch_training_losses = []
        self.G_epoch_validation_losses = []
        self.save_hyperparameters("lr_G", "lr_D", "run_name", "val_epoch", "G_iter")
    
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("AuxGAN")
        #parser.add_argument("--data_path", type=str, default="")
        parser.add_argument("--learning_rate_G", type=float, default=2e-5)
        parser.add_argument("--learning_rate_D", type=float, default=2e-5)
        #parser.add_argument("--D_iters", type=int, default=1)
        return parent_parser
    
    def configure_optimizers(self):
        # g_opt = torch.optim.RMSprop(self.G.parameters(), self.hparams.lr_G)
        # d_opt = torch.optim.RMSprop(self.D.parameters(), self.hparams.lr_D)
        # g_opt = torch.optim.RMSprop(self.G.parameters(), 2e-5)
        # d_opt = torch.optim.RMSprop(self.D.parameters(), 2e-5)
        g_opt = torch.optim.Adam(self.G.parameters(), self.hparams.lr_G, betas=(0.5, 0.999))
        d_opt = torch.optim.Adam(self.D.parameters(), self.hparams.lr_D, betas=(0.5, 0.999))
        scheduler1 = torch.optim.lr_scheduler.MultiStepLR(g_opt,  milestones=[20,50], gamma=0.99)
        scheduler2 = torch.optim.lr_scheduler.MultiStepLR(d_opt, milestones=[20,50], gamma=0.99)
        return [g_opt, d_opt], [scheduler1, scheduler2]

    def validation_step(self, batch, batch_idx):
        #x, y = batch[0].to(self.device), batch[1].to(self.device)
        x, categs = batch[0], batch[1]
        if(batch_idx == 0):
            eval_batch = next(iter(self.eval_loader))
            fixed_noise, fixed_labels = eval_batch[0], eval_batch[1]
            fixed_noise = fixed_noise.type_as(x)
            fixed_labels = fixed_labels.type(torch.LongTensor).type_as(categs)
            #print(x.size(), y.size())
            fake = self.G(fixed_noise, fixed_labels)
            #preds = self.D(fake, fixed_labels)

            self.validation_history.append(vutils.make_grid(fake, padding=2, normalize=True)[:1, :, :].squeeze(0))
            #print(y, preds)
            plt.figure(figsize=(8,8))
            plt.imshow(self.validation_history[-1].detach().cpu().numpy())#, cmap='gray')
            plt.savefig('/home/moulikc2/expose/Climate Generation/validation_figs/'\
                            +self.hparams.run_name+'/epoch_'+str(self.current_epoch)+"_batch_"+str(batch_idx)+'.png')
            plt.close()
            
            if(len(self.G_epoch_validation_losses) > 0):
                self.G_validation_losses.append(np.mean(self.G_epoch_validation_losses))
                self.G_epoch_validation_losses = []


            if(self.current_epoch > 3):
                plt.figure(figsize=(8,8))
                plt.plot([i for i in range(self.current_epoch) if i%self.hparams.val_epoch==0], self.G_validation_losses)#, cmap='gray')
                plt.savefig('/home/moulikc2/expose/Climate Generation/loss_curves/'\
                                +self.hparams.run_name+'/G_val_epoch_'+str(self.current_epoch)+"_batch_"+str(batch_idx)+'.png')
                plt.close()

        x, categs = batch[0], batch[1]
        b_size = x.size(0) 
        noise = torch.randn((b_size, self.latent_dim, 1, 1))
        noise = noise.type_as(x)
        g_X = self.G(noise, categs)
        real_label = torch.full((b_size,), 1, dtype=torch.float)
        real_label = real_label.type_as(categs)
        #print("g_step")        

        d_out, aux_out = self.D(g_X)
        fake_g_id = self.criterion(d_out, real_label)
        fake_g_aux = self.aux_criterion(aux_out, categs)
        errG = fake_g_id + fake_g_aux
        self.G_epoch_validation_losses.append(errG.item())

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
        #loss = torch.nn.BCELoss()
        return loss(y_hat, y.float())
    
    def aux_criterion(self, y_hat, y):
        loss = torch.nn.CrossEntropyLoss(label_smoothing=0.0)
        #loss = torch.nn.BCELoss()
        return loss(y_hat, y)

    def training_step(self, batch, batch_idx):
        
        g_opt, d_opt = self.optimizers()
        sch1, sch2 = self.lr_schedulers()

        if(batch_idx==0):

            if(len(self.D_epoch_training_losses) > 0):
                #self.D_training_losses.append(np.mean(self.D_epoch_training_losses))
                self.D_training_losses += self.D_epoch_training_losses
                self.D_epoch_training_losses = []

                #self.G_training_losses.append(np.mean(self.G_epoch_training_losses))
                self.G_training_losses += self.G_epoch_training_losses
                self.G_epoch_training_losses = []

            if((self.current_epoch+1)%self.hparams.val_epoch == 0):
                plt.figure(figsize=(8,8))
                #plt.plot([i for i in range(self.current_epoch)], self.G_training_losses, label='G_train_loss')#, cmap='gray')
                #plt.plot([i for i in range(self.current_epoch)], self.D_training_losses, label='D_train_loss')
                plt.plot(np.arange(len(self.G_training_losses)), self.G_training_losses, label='G_train_loss')#, cmap='gray')
                plt.plot(np.arange(len(self.D_training_losses)), self.D_training_losses, label='D_train_loss')
                plt.legend()
                plt.savefig('/home/moulikc2/expose/Climate Generation/loss_curves/'\
                                +self.hparams.run_name+'/train_loss_epoch_'+str(self.current_epoch)+"_batch_"+str(batch_idx)+'.png')
                plt.close()
    

        # Format batch
        real_img = batch[0]
        categs = batch[1]
        b_size = real_img.size(0)


        #print("yo1")
        ### fake_label = 0
        fake_label = torch.full((b_size,), 0, dtype=torch.float)
        ### smoothing
        #fake_label += torch.rand((b_size,))*0.2 
        fake_label = fake_label.type_as(categs)

        real_label = torch.full((b_size,), 1, dtype=torch.float)
        ### smoothing
        #real_label -= torch.rand((b_size,))*0.2 
        real_label = real_label.type_as(categs)

        gen_labels = torch.LongTensor(np.random.randint(0, self.G.num_classes, b_size)).type_as(categs)

        # Forward pass real batch through D
        d_opt.zero_grad()
        d_output, d_aux = self.D(real_img)
        #print("d_step")
        real_disc_id = 0.5 * self.criterion(d_output, real_label)
        real_disc_aux = self.aux_criterion(d_aux, categs)
        errD_real = real_disc_id + real_disc_aux
        #print("yo2")
        # Train with all-fake batch
        noise = torch.randn((b_size, self.latent_dim, 1, 1))
        noise = noise.type_as(real_img)
        g_X = self.G(noise, gen_labels)
        #print("g_step")

        d_output, d_aux = self.D(g_X.detach())
        fake_disc_id = 0.5 * self.criterion(d_output, fake_label)
        fake_disc_aux = self.aux_criterion(d_aux, gen_labels)
        errD_fake = fake_disc_id + fake_disc_aux#*self.class_imbalance
        #print("yo3")
        # Compute error of D as sum over the fake and the real batches
        errD = errD_real + errD_fake
        self.manual_backward(errD)
        d_opt.step()
        #print("yo4")

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        # Calculate G's loss based on D's output
        #self.G.zero_grad()
        G_losses_temp = []
        g_opt.zero_grad()
        real_label = torch.full((b_size,), 1, dtype=torch.float)
        ### no-smoothing for generator
        real_label = real_label.type_as(categs)
        for i in range(self.hparams.G_iter):
          g_X = self.G(noise, gen_labels)
          d_out, aux_out = self.D(g_X)
          fake_g_id = 0.5 * self.criterion(d_out, real_label)
          fake_g_aux = self.aux_criterion(aux_out, gen_labels)
          errG = fake_g_id + fake_g_aux
          self.manual_backward(errG)
          g_opt.step()
          G_losses_temp.append(errG.item())

        self.log_dict({"g_id": fake_g_id, "d_id_real": real_disc_id, "d_id_fake": fake_disc_id, "g_aux": fake_g_aux, "d_aux": real_disc_aux}, prog_bar=True)
        self.G_epoch_training_losses.append(np.mean(G_losses_temp))
        self.D_epoch_training_losses.append(errD.item())

        if batch_idx==0:
            print(sch1.get_last_lr(), sch2.get_last_lr())
            sch1.step()
            sch2.step()


class Pix2Pix(pl.LightningModule):
    def __init__(self, generator, discriminator, run_name, learning_rate=0.0003, lambda_recon=20, add_noise=False):

        super().__init__()
        self.save_hyperparameters()
        self.add_noise = add_noise

        #self.gen = Generator(in_channels, out_channels)
        #self.patch_gan = PatchGAN(in_channels + out_channels)
        self.gen = generator
        self.patch_gan = discriminator

        # intializing weights
        self.gen = self.gen.apply(self._weights_init)
        self.patch_gan = self.patch_gan.apply(self._weights_init)

        self.adversarial_criterion = nn.BCEWithLogitsLoss()
        self.recon_criterion = nn.L1Loss()
        self.eval_loader = None

        self.save_hyperparameters("learning_rate", "lambda_recon", "run_name")

    def _weights_init(self, m):
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            torch.nn.init.normal_(m.weight, 0.0, 0.02)
        if isinstance(m, nn.BatchNorm2d):
            torch.nn.init.normal_(m.weight, 0.0, 0.02)
            torch.nn.init.constant_(m.bias, 0)
        
    def _gen_step(self, real_images, conditioned_images):
        # Pix2Pix has adversarial and a reconstruction loss
        # First calculate the adversarial loss
        if self.add_noise:
            noise = torch.randn(conditioned_images.size())
            noise = noise.type_as(conditioned_images)
        else:
            noise = None
            
        fake_images = self.gen(conditioned_images, noise)
        disc_logits = self.patch_gan(fake_images, conditioned_images)
        adversarial_loss = self.adversarial_criterion(disc_logits, torch.ones_like(disc_logits))

        # calculate reconstruction loss
        recon_loss = self.recon_criterion(fake_images, real_images)
        lambda_recon = self.hparams.lambda_recon

        return adversarial_loss + lambda_recon * recon_loss

    def _disc_step(self, real_images, conditioned_images):
        if self.add_noise:
            noise = torch.randn(conditioned_images.size())
            noise = noise.type_as(conditioned_images)
        else:
            noise = None
        
        fake_images = self.gen(conditioned_images, noise).detach()
        fake_logits = self.patch_gan(fake_images, conditioned_images)

        real_logits = self.patch_gan(real_images, conditioned_images)

        fake_loss = self.adversarial_criterion(fake_logits, torch.zeros_like(fake_logits))
        real_loss = self.adversarial_criterion(real_logits, torch.ones_like(real_logits))
        return (real_loss + fake_loss) / 2

    def configure_optimizers(self):
        lr = self.hparams.learning_rate
        gen_opt = torch.optim.Adam(self.gen.parameters(), lr=lr)
        disc_opt = torch.optim.Adam(self.patch_gan.parameters(), lr=lr)
        return disc_opt, gen_opt

    def training_step(self, batch, batch_idx, optimizer_idx):
        real, condition = batch

        loss = None
        if optimizer_idx == 0:
            loss = self._disc_step(real, condition)
            self.log("PatchGAN Loss", loss)
        elif optimizer_idx == 1:
            loss = self._gen_step(real, condition)
            self.log("Generator Loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        #x, y = batch[0].to(self.device), batch[1].to(self.device)
        if self.add_noise:
            x, categs, noise = batch[0], batch[1], batch[2]
        else:
            x, categs = batch[0], batch[1]

        
        if(batch_idx == 0):
            eval_batch = next(iter(self.eval_loader))
            if self.add_noise:
                fixed_images, fixed_maps, noise = eval_batch[0], eval_batch[1], eval_batch[2]
                noise = noise.type_as(categs)
            else:
                fixed_images, fixed_maps = eval_batch[0], eval_batch[1]
                noise = None
            fixed_images = fixed_images.type_as(x)
            fixed_maps = fixed_maps.type_as(categs)
            #print(x.size(), y.size())
            fake_images = self.gen(fixed_maps, noise)
                
            #preds = self.patch_gan(fake, fixed_maps)

            #self.validation_history.append((vutils.make_grid(fake, padding=2, normalize=True)[:1, :, :].squeeze(0), preds.detach().cpu().numpy()))
            #print(y, preds)
            plt.figure(figsize=(8,8))
            plt.imshow(vutils.make_grid(fake_images, padding=2, normalize=True, nrow=5)[:1, :, :].squeeze(0).detach().cpu().numpy())#, cmap='gray')
            plt.savefig('/home/moulikc2/expose/Climate Generation/validation_figs/'\
                            +self.hparams.run_name+'/epoch_'+str(self.current_epoch)+'generated' + '.png')
            plt.close()

            plt.figure(figsize=(8,8))
            plt.imshow(vutils.make_grid(fixed_images, padding=2, normalize=True, nrow=5)[:1, :, :].squeeze(0).detach().cpu().numpy())#, cmap='gray')
            plt.savefig('/home/moulikc2/expose/Climate Generation/validation_figs/'\
                            +self.hparams.run_name+'/epoch_'+str(self.current_epoch)+'actual' + '.png')
            plt.close()

        # x, categs = batch[0], batch[1]
        # b_size = x.size(0) 
        # noise = torch.randn((b_size, self.latent_dim, 1, 1))
        # noise = noise.type_as(x)
        # g_X = self.G(noise, categs)
        # real_label = torch.full((b_size,), 1, dtype=torch.float)
        # real_label = real_label.type_as(categs)
        # #print("g_step")        

        # errG = self.criterion(self.D(g_X, categs).view(-1), real_label)
        # self.G_epoch_validation_losses.append(errG.item())
