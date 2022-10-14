import os
import time

import torch
from torch import utils
import pytorch_lightning as pl

from multi_read_data import MemoryFriendlyLoader
from pl_model import SCI


#image_path = 'EXP/' + 'Train-{}'.format(time.strftime("%Y%m%d-%H%M%S"))
image_path = "EXP/darkface"
os.makedirs(image_path, exist_ok=True)

model = SCI(img_path=image_path)

data_dir = "/mnt/nas/open_data/Low_Light_Enhancement_RGB/Retinexnet_Dataset(LOL)/LOL/our485/low"
train_data_dir = "/mnt/nas/open_data/Low_Light_Enhancement_RGB/DarkFace_Train_2021/image"

valid_data_dir = "/mnt/nas/sjp/validation"

train_dataset = MemoryFriendlyLoader(img_dir=train_data_dir, task='train')
train_loader = utils.data.DataLoader(train_dataset, batch_size=2, num_workers=0, shuffle=True)

valid_dataset = MemoryFriendlyLoader(img_dir=valid_data_dir, task="valid")
valid_loader = utils.data.DataLoader(valid_dataset, batch_size=2, num_workers=0, shuffle=True)


trainer = pl.Trainer(accelerator="gpu", max_epochs=200, strategy="ddp", check_val_every_n_epoch=50)
trainer.fit(model, train_loader, valid_loader)