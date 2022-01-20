import pytorch_lightning as pyl
import torch
import torch.nn.functional as F
import json
import time
import sys
import models_seq
import datasets_seq
from common_structures_seq import PointData
import importlib
import os
import sklearn.metrics
import numpy as np


class ModelLightning(pyl.LightningModule):
    def __init__(
            self, config, backbone):
        super().__init__()
        self.config = config
        self.backbone = backbone
        pass

    def setup(self, stage=None):
        pass
        
    def forward(self, batch):
        #if self.global_step == 0:
        #   self.backbone.graph_model.node_features_init = self.backbone.node_features_init.type_as(batch['sources'])
        #   self.backbone.graph_model.edge_initial_feat_dict = self.backbone.edge_initial_feat_dict.type_as(batch['sources'])
        x = self.backbone(
            batch['edge_feat'],
            batch['label_feat'],
            batch['trip_feat'],
            batch['trip_mask'],
            batch['pair_feat'],
            batch['pair_mask'],
            batch['src_feat'],
            batch['src_mask'],
            batch['dst_feat'],
            batch['dst_mask'],
            batch['trip_feat_extra_b'],
            batch['pair_feat_extra_b'],
            batch['src_feat_extra_b'],
            batch['dst_feat_extra_b'],
            batch['user_graphfeat_initial'],
            batch['oppo_grapgfeat_initial'],
            batch['user_1hop_edge_orgfeature'],
            batch['user_2hop_edge_orgfeature'],
            batch['oppo_1hop_edge_orgfeature'],
            batch['oppo_2hop_edge_orgfeature'],
            batch['graph_src_id'],
            batch['graph_dst_id'],
            batch['labels_time']
        )
        return x
        
    def training_step(self, batch, batch_idx):
        logits = self(batch)

        if batch_idx == 0 and self.global_rank == 0:
            # print(batch['eid'])
            # print(torch.sigmoid(logits))
            # print(batch['edge_feat'])
            pass

        loss = F.binary_cross_entropy_with_logits(
            logits, batch['label'], reduction='none')

        loss = torch.mean(loss * batch['label_mask'])
        self.log("loss2", loss, on_step=True, prog_bar=True, logger=False)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        scores = self(batch)

        label_weights = batch['label_weights']
        label = batch['label']

        proba = torch.sigmoid(scores)
        # torch.set_printoptions(profile="full")
        # print(proba)
        proba = 1 - torch.prod(1 - proba * label_weights, dim=-1)
        # proba = (proba * label_weights).max(dim=-1)[0]

        return {'proba': proba, 'label': label}

    def validation_epoch_end(self, outputs):
        pred = torch.cat([output['proba'] for output in outputs])
        label = torch.cat([output['label'] for output in outputs])
        valid_auc = sklearn.metrics.roc_auc_score(label.cpu().numpy().flatten(), pred.cpu().numpy().flatten())
        self.log('valid_auc', valid_auc)
        self.log('learning_rate', self.optimizers(0).param_groups[0]['lr'])
        pass

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.config['learning_rate'])
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=10, gamma=0.7)

        # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        #     optimizer, 10, T_mult=1, eta_min=1e-5)

        # steps_per_epoch = len(self.train_dataloader()) // self.trainer.num_gpus
        # print('steps per epoch: {0}'.format(steps_per_epoch))
        # 
        # self.lr_scheduler = timm.scheduler.CosineLRScheduler(
        #     optimizer, t_initial=config['lr_decay_steps']*steps_per_epoch, lr_min=config['min_lr'],
        #     decay_rate=config['lr_decay_rate'], warmup_t=config['warmup_epochs']*steps_per_epoch,
        #     warmup_lr_init=config['warmup_lr'], warmup_prefix=True,
        #     t_in_epochs=False)
            
        return [optimizer], [scheduler]

    def backward(
                self, loss, *args, **kargs
            ):
        super().backward(loss, *args, **kargs)

        for p in self.parameters():
            if (p.grad is not None and torch.any(torch.isnan(p.grad))) or \
               torch.any(torch.isnan(p)):
                raise RuntimeError('nan happend')
            pass

        pass
    
    def predict_step(self, batch, batch_idx: int , dataloader_idx: int = None):
        scores = self(batch)

        label_weights = batch['label_weights']

        proba = torch.sigmoid(scores)
        proba = 1 - torch.prod(1 - proba * label_weights, dim=-1)
        # proba = (proba * label_weights).max(dim=-1)[0]
        
        return proba.cpu().numpy().flatten()

    pass


if __name__ == '__main__':
    start = time.time()

    config_file = sys.argv[1]
    config = importlib.import_module(config_file).config

    edge_initial_feat_dict = np.load('{}/edge_initial_feat.npy'.format(config['graph_dataset_path']))
    node_initial_feat = np.load('{}/node_initial_feat.npy'.format(config['graph_dataset_path']))

    backbone = models_seq.HierarchicalTransformer(config, edge_initial_feat_dict, node_initial_feat)

    dataset_train = datasets_seq.DygDataset(
        config, 'train', valid_percent=0.01, num=250000)
    # dataset_valid = datasets.DygDataset(
    #     config, 'valid_test', valid_percent=0.01, num=50000)
    dataset_valid = datasets_seq.DygDatasetTest(config, 'val')

    loader_train = torch.utils.data.DataLoader(
        dataset=dataset_train,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_data_workers'],
        collate_fn=datasets_seq.dyg_collate_fn,
        # sampler=datasets_seq.RandomDropSampler(dataset_train, 0.999)
        )
    loader_valid = torch.utils.data.DataLoader(
        dataset=dataset_valid,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_data_workers'],
        collate_fn=datasets_seq.dyg_test_collate_fn,
        )

    model = ModelLightning(
        config, backbone=backbone)

    if len(sys.argv) == 3:
        model.load_state_dict(torch.load(sys.argv[2])['state_dict'])
        pass
    
    checkpoint_callback = pyl.callbacks.ModelCheckpoint(
                monitor='valid_auc',
                mode='max',
                save_last=True,
                save_top_k=5)

    trainer = pyl.Trainer(
        logger=pyl.loggers.CSVLogger('./lightning_logs/logs.csv'),
        gradient_clip_val=0.1,
        replace_sampler_ddp=False,
        max_epochs=1000,
        accelerator=config['accelerator'],
        gpus=config['gpus'],
        callbacks=[checkpoint_callback]
    )

    # torch.autograd.set_detect_anomaly(True)
    trainer.fit(
        model, train_dataloaders=loader_train,
        val_dataloaders=loader_valid
    )
