
from argparse import ArgumentParser
from train_module import *
import pytorch_lightning as pl
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from dataset import *
from models import *


parser = ArgumentParser()

# add PROGRAM level args
parser.add_argument("--conda_env", type=str, default="some_name")
parser.add_argument("--notification_email", type=str, default="will@email.com")

# add training module specific args
parser = GAN.add_model_specific_args(parser)

### add generator specific args
G_parser = parser.add_argument_group("generator")
G_parser.add_argument("--input_dim", type=int, default=128)
G_parser.add_argument("--conditional_model", type=bool, default=True)
G_parser.add_argument("--num_classes", type=int)
G_parser.add_argument("--g_layers", type=list, default=[8, 64, 128, 256, 256, 128, 1])
G_parser.add_argument("--inject_z", type=bool, default=True)
G_parser.add_argument("--remove_hot", type=int, default=0)
G_parser.add_argument("--transform_rep", type=int, default=1)
G_parser.add_argument("--transform_z", type=bool, default=False)
G_parser.add_argument("--norm_type", type=str, default="batch")
G_parser.add_argument("--up_mode", type=str, default="upsample")
G_parser.add_argument("--skip_connection", type=bool, default=True)
G_parser.add_argument("--bias", type=bool, default=True)
G_parser.add_argument("--filter_size", type=int, default=3)
G_parser.add_argument("--num_skip", type=int, default=32)
G_parser.add_argument("--skip_size", type=int, default=1)
G_parser.add_argument("--residual", type=bool, default=False)

### add discriminator specific args
D_parser = parser.add_argument_group("discriminator")
D_parser.add_argument("--input_dim", type=int, default=128)
D_parser.add_argument("--conditional_model", type=bool, default=True)
D_parser.add_argument("--num_classes", type=int)
D_parser.add_argument("--g_layers", type=list, default=[8, 64, 128, 256, 256, 128, 1])
D_parser.add_argument("--inject_z", type=bool, default=True)
D_parser.add_argument("--remove_hot", type=int, default=0)
D_parser.add_argument("--transform_rep", type=int, default=1)
D_parser.add_argument("--transform_z", type=bool, default=False)
D_parser.add_argument("--norm_type", type=str, default="batch")
D_parser.add_argument("--up_mode", type=str, default="upsample")
D_parser.add_argument("--skip_connection", type=bool, default=True)
D_parser.add_argument("--bias", type=bool, default=True)
D_parser.add_argument("--filter_size", type=int, default=3)
D_parser.add_argument("--num_skip", type=int, default=32)
D_parser.add_argument("--skip_size", type=int, default=1)
D_parser.add_argument("--residual", type=bool, default=False)

# add all the available trainer options to argparse
# ie: now --accelerator --devices --num_nodes ... --fast_dev_run all work in the cli
parser = pl.Trainer.add_argparse_args(parser)

args = parser.parse_args()


PATH_DATASETS = os.environ.get("PATH_DATASETS", ".")
BATCH_SIZE = 256 if torch.cuda.is_available() else 64
NUM_WORKERS = int(os.cpu_count() / 2)

dm = prep_dataloaders()
model = GAN(*dm.size())
trainer = Trainer(
    accelerator="auto",
    devices=1 if torch.cuda.is_available() else None,  # limiting got iPython runs
    max_epochs=5,
    callbacks=[TQDMProgressBar(refresh_rate=20)],
)
trainer.fit(model, dm)