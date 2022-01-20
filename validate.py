import sys
import importlib
from train import ModelLightning
import models_seq
import torch
import datasets_seq
import pytorch_lightning as pyl
import numpy as np
import os
import sklearn.metrics


if __name__ == '__main__':
    config_file = sys.argv[1]
    ckpt_file = sys.argv[2]
    config = importlib.import_module(config_file).config

    edge_initial_feat_dict = np.load('{}/edge_initial_feat.npy'.format(config['graph_dataset_path']))
    node_initial_feat = np.load('{}/node_initial_feat.npy'.format(config['graph_dataset_path']))

    backbone = models_seq.HierarchicalTransformer(config, edge_initial_feat_dict, node_initial_feat)

    #backbone = models_seq.HierarchicalTransformer(config)
    model = ModelLightning(
        config, backbone=backbone)

    model.load_state_dict(torch.load(ckpt_file)['state_dict'])
    model.eval()

    dataset = datasets_seq.DygDatasetTest(config, 'val')

    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=32,
        shuffle=False,
        num_workers=20,
        collate_fn=datasets_seq.dyg_test_collate_fn
        )

    trainer = pyl.Trainer(
        gpus=1
    )

    with torch.no_grad():
        pred = trainer.predict(
            model, dataloader)
        pass

    pred = np.hstack(pred)
    label = np.load(
        os.path.join(config['dataset_path'], 'val_labels.npy'))

    print(sklearn.metrics.roc_auc_score(label, pred))

    # val_index = np.load(
    #     os.path.join(config['dataset_path'], 'val_index.npy'))
    #
    # pred[val_index[:, 1]==-1] = 0
    # # pred.fill(0)
    #
    # print(sklearn.metrics.roc_auc_score(label, pred))
    # pass
