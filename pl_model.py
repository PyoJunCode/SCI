import os

import torch
import torch.nn as nn
import pytorch_lightning as pl

from model import EnhanceNetwork, CalibrateNetwork, Network
from utils import save_images

class SCI(pl.LightningModule):
    def __init__(self, stage=3, lr=0.0001, img_path="EXP/"):
        super().__init__()
        self.image_path = img_path
        self.lr = lr
        self.stage = stage
        self.model = Network(stage=3)
        self._weights_init()

        self.losses = []
        
    def _weights_init(self):
        self.model.enhance.in_conv.apply(self.model.weights_init)
        self.model.enhance.conv.apply(self.model.weights_init)
        self.model.enhance.out_conv.apply(self.model.weights_init)
        self.model.calibrate.in_conv.apply(self.model.weights_init)
        self.model.calibrate.convs.apply(self.model.weights_init)
        self.model.calibrate.out_conv.apply(self.model.weights_init)

    def forward(self, x):
        ilist, rlist, inlist, attlist = [], [], [], []
        input_op = x
        for i in range(self.stage):
            inlist.append(input_op)
            i = self.model.enhance(input_op)
            r = x / i
            r = torch.clamp(r, 0, 1)
            att = self.model.calibrate(r)
            input_op = x + att
            ilist.append(i)
            rlist.append(r)
            attlist.append(torch.abs(att))

        return ilist, rlist, inlist, attlist    


    def training_step(self, batch, batch_idx):
        x, _ = batch
        i_list, en_list, in_list, _ = self(x)
        loss = 0
        for i in range(self.stage):
            loss += self.model._criterion(in_list[i], i_list[i])
        
        self.losses.append(loss)
        nn.utils.clip_grad_norm_(self.parameters(), 5)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        print("Start validating")
        x, image_name = batch
        image_name = image_name[0].split('\\')[-1].split('.')[0]
        illu_list, ref_list, input_list, atten= self(x)
        u_name = '%s.png' % (image_name + '_' + str(self.current_epoch))
        u_path = os.path.join(self.image_path, u_name)
        #save_images(ref_list[0], u_path)
        
        return {"img": ref_list[0], "path": u_path}

    def validation_epoch_end(self, outs):
        for out in outs:
            save_images(out["img"], out["path"])

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, betas=(0.9, 0.999), weight_decay=1e-4)
        
        return optimizer