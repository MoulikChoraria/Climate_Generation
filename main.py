
from argparse import ArgumentParser
from models.Poly_Discriminator import auxiliary_polydisc, conditional_polydisc
from models.Poly_Generator import conditional_polygen
from models.Resnet_Gen_Proj import ResNetGenerator
from models.Resnet_Disc_Proj import ResNetProjectionDiscriminator
from train_module import *
from data_utils import *
import pytorch_lightning as pl
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from pytorch_lightning import LightningDataModule, LightningModule, Trainer, seed_everything
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from torch.utils.data import TensorDataset, DataLoader
from Climate_Generation.data_utils import *
from models import *



if __name__ == "__main__":
    parser = ArgumentParser()
    #parser = Trainer.add_argparse_args(parser)
    #args = parser.parse_args()
    # add PROGRAM level args
    # parser.add_argument("--conda_env", type=str, default="some_name")
    # parser.add_argument("--notification_email", type=str, default="will@email.com")

    # add training module specific args

    ### add generator specific args
    G_parser = parser.add_argument_group("generator")
    #### input_dim = required_filters - 1 for embedding layer
    G_parser.add_argument("--g_input_dim", type=int, default=128)
    #G_parser.add_argument("--conditional_model", type=bool, default=True)
    G_parser.add_argument("--g_num_classes", type=int, default=5)
    G_parser.add_argument("--g_layers", type=list, default=[8+1, 128, 192, 256, 192, 128, 1])
    G_parser.add_argument("--g_inject_z", type=bool, default=False)
    G_parser.add_argument("--g_remove_hot", type=int, default=2)
    G_parser.add_argument("--g_transform_embedding", type=bool, default=True)
    G_parser.add_argument("--g_transform_z", type=bool, default=False)
    G_parser.add_argument("--g_norm_type", type=str, default="batch")
    G_parser.add_argument("--g_up_mode", type=str, default="conv_transpose")
    G_parser.add_argument("--g_skip_connection", type=bool, default=True)
    G_parser.add_argument("--g_bias", type=bool, default=False)
    G_parser.add_argument("--g_filter_size", type=int, default=3)
    G_parser.add_argument("--g_num_skip", type=int, default=32)
    G_parser.add_argument("--g_skip_size", type=int, default=3)
    G_parser.add_argument("--g_residual", type=bool, default=True)

    ### add discriminator specific args
    D_parser = parser.add_argument_group("discriminator")
    D_parser.add_argument("--d_input_dim", type=tuple, default=(128, 128))
    #D_parser.add_argument("--conditional_model", type=bool, default=True)
    D_parser.add_argument("--d_num_classes", type=int, default=5)
    D_parser.add_argument("--d_layers", type=list, default=[2, 32, 32, 64, 64, 64, 128])
    D_parser.add_argument("--d_inject_z", type=bool, default=False)
    D_parser.add_argument("--d_remove_hot", type=int, default=2)
    D_parser.add_argument("--d_filter_size", type=int, default=3)
    D_parser.add_argument("--d_norm", type=str, default="layer")
    D_parser.add_argument("--d_skip_connection", type=bool, default=True)
    D_parser.add_argument("--d_bias", type=bool, default=False)
    D_parser.add_argument("--d_num_skip", type=int, default=16)
    D_parser.add_argument("--d_skip_size", type=int, default=1)
    D_parser.add_argument("--d_residual", type=bool, default=True)
    D_parser.add_argument("--d_downsample_mode", type=str, default="pooling")
    D_parser.add_argument("--d_pool_type", type=str, default="avg")
    D_parser.add_argument("--d_pool_filter", type=int, default=1)
    #D_parser.add_argument("--transform_rep", type=int, default=1)
    #D_parser.add_argument("--transform_z", type=bool, default=False)

    ### add training specific args
    Trainer_parser = parser.add_argument_group("trainer")
    Trainer_parser.add_argument("--epochs", type=int, default=100)
    Trainer_parser.add_argument("--data_path", type=str)
    #D_parser.add_argument("--conditional_model", type=bool, default=True)
    Trainer_parser.add_argument("--val_epochs", type=int, default=5)
    Trainer_parser.add_argument("--batch_size", type=int, default=64)
    Trainer_parser.add_argument("--seed", type=int, default=566)
    Trainer_parser.add_argument("--train_val_split", type=float, default=1)
    Trainer_parser.add_argument("--run_name", type=str, default="rms_prop_run")
    Trainer_parser.add_argument("--n_critic", type=int, default=4)
    Trainer_parser.add_argument("--lambda_gp", type=float, default=100.0)
    Trainer_parser.add_argument("--G_iter", type=int, default=1)

    #Trainer_parser.add_argument("--lr_G", type=float, default=1e-5)
    #Trainer_parser.add_argument("--lr_D", type=int, default=0)
    #parser.add_argument("--max_epochs", type=int, default=5)



    # add all the available trainer options to argparse
    # ie: now --accelerator --devices --num_nodes ... --fast_dev_run all work in the cli
    parser = pl.Trainer.add_argparse_args(parser)
    parser = GAN.add_model_specific_args(parser)
    #parser = WGAN.add_model_specific_args(parser)
    #parser = AuxGAN.add_model_specific_args(parser)


    args = parser.parse_args()
    ### make run directory

    path_train_figs = '/home/moulikc2/expose/Climate Generation/loss_curves/'+args.run_name

    # Check whether the specified path exists or not
    isExist = os.path.exists(path_train_figs)

    if not isExist:
    # Create a new directory because it does not exist 
        os.makedirs(path_train_figs)

    path_val_figs = '/home/moulikc2/expose/Climate Generation/validation_figs/'+args.run_name

    # Check whether the specified path exists or not
    isExist = os.path.exists(path_val_figs)

    if not isExist:
    # Create a new directory because it does not exist 
        os.makedirs(path_val_figs)
        #print("The new directory is created!")
    
    path_val_figs = '/home/moulikc2/expose/Climate Generation/test_figs/'+args.run_name

    # Check whether the specified path exists or not
    isExist = os.path.exists(path_val_figs)

    if not isExist:
    # Create a new directory because it does not exist 
        os.makedirs(path_val_figs)
        #print("The new directory is created!")


    converted_dict = vars(args)
    disc_keylist = [key for key in list(converted_dict.keys()) if key[:2]=="d_"]
    gen_keylist = [key for key in list(converted_dict.keys()) if key[:2]=="g_"]

    seed_everything(args.seed, workers=True)

    dict_gen = {}
    for key in gen_keylist:
        dict_gen[key[2:]] = converted_dict[key]

    dict_disc = {}
    for key in disc_keylist:
        dict_disc[key[2:]] = converted_dict[key]
    
    #disc_model = conditional_polydisc(**dict_disc)
    disc_model = ResNetProjectionDiscriminator()
    #disc_model = auxiliary_polydisc(**dict_disc)
    #print(disc_model)
    pytorch_total_params = sum(p.numel() for p in disc_model.parameters())
    print("Discriminator Parameters:", pytorch_total_params)    
    #gen_model = conditional_polygen(**dict_gen)
    gen_model = ResNetGenerator()
    pytorch_total_params = sum(p.numel() for p in gen_model.parameters())
    print("Generator Parameters:", pytorch_total_params)    
    


    gan_model = GAN(gen_model, disc_model, args.learning_rate_G, args.learning_rate_D, args.run_name, args.val_epochs, args.G_iter)
    #gan_model = AuxGAN(gen_model, disc_model, args.learning_rate_G, args.learning_rate_D, args.run_name, args.val_epochs, args.G_iter)
    #gan_model = WGAN(gen_model, disc_model, args.learning_rate_G, args.learning_rate_D, args.run_name, args.val_epochs, args.n_critic, args.lambda_gp)
    #gan_model.configure_optimizers(args.lr_G, args.lr_D)

    BATCH_SIZE = args.batch_size
    NUM_WORKERS = 32

    ### dataset
    train_loader, val_loader, test_loader, train_stats = prep_dataloaders(args.data_path, args.g_num_classes, \
                                                        'log_transform', bs = BATCH_SIZE, workers=NUM_WORKERS, sampler=True)

    #print("PHEW")
    # if(val_loader == None):
    print("creating fixed datapoints for eval")
    fixed_noise = torch.randn(40, args.g_input_dim, 1, 1)
    fixed_labels = torch.LongTensor(np.array([i//8 for i in range(40)])[:, np.newaxis])
    #print(fixed_labels)
    #print(gen_model(fixed_noise, fixed_labels))
    chirps_val_dataset = TensorDataset(fixed_noise, fixed_labels)
    eval_dataloader = DataLoader(chirps_val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    gan_model.eval_loader = eval_dataloader
    gan_model.test_loader = test_loader

    device = torch.device("cpu") if not torch.cuda.is_available() else torch.device("cuda:1")
    print(device)
    accelerator_train = "gpu" if device==torch.device("cuda:1") else "cpu"
    print(accelerator_train)
    #trainer = Trainer.from_argparse_args(args)
    trainer = Trainer(accelerator=accelerator_train, max_epochs = args.epochs,\
                                        callbacks=[TQDMProgressBar(refresh_rate=5)], \
                                            check_val_every_n_epoch=1, devices=[0], log_every_n_steps=30)
    #trainer = Trainer(accelerator="auto", devices=1 if torch.cuda.is_available() else None,  # limiting got iPython runs
                        #max_epochs=5,callbacks=[TQDMProgressBar(refresh_rate=20)],)
    trainer.fit(gan_model, train_loader, val_loader)

    #main(args)