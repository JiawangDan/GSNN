import constants
import numpy as np

edge_feat_dim = 768

max_label_class = 5000
num_edge_type = 14
num_nodes = 811985

config = {
    'train_file': '../data_set/edges_train_B.csv',
    'val_file': '../data_set/input_B.csv',
    'test_file': '../data_set/input_B_new.csv',
    'dataset_path': '../dataset_B',
    'neg_sample_interval': 2 * 60 * 24 * 60 * 60,
    'neg_sample_num': 64,
    'neg_sample_proba': 0.5,
    'train_start': '20150301',
    'max_train_ts': 1443628799,
    'label_bin_size': 60 * 60,
    'edge_feat_dim': [1],
    'extra_feat_dim': 768,
    'trip_feat_dim': constants.TE_FEAT_DIM.tolist(),
    'pair_feat_dim': np.concatenate((constants.TE_FEAT_DIM, [num_edge_type], [2])).tolist(),
    'label_feat_dim': [max_label_class, constants.NUM_WEEKDAY,
                       constants.MAX_TE_DAYS_DIFF, constants.NUM_HOUR],
    # 'label_feat_dim': [max_label_class],
    'max_label_class': max_label_class,
    'node_history_feat_dim': np.concatenate((constants.TE_FEAT_DIM, [num_nodes, num_edge_type, 2, 2])).tolist(),

    'hidden_size': 128,
    'num_attention_heads': 4,
    'num_hidden_layers_trip': 3,
    'num_hidden_layers_pair': 3,
    'num_hidden_layers2': 3,
    'num_hidden_layers_feat': 1,
    'batch_size': 128,
    'learning_rate': 0.0001,
    'num_data_workers': 15,
    'hidden_dropout_prob': 0,
    'attention_probs_dropout_prob': 0,

    'gpus': 1,
    'accelerator': 'ddp',

##graph other params
    'graph_dataset_path': '../dataset_B_graph',
    'min_timestamp': 1420000000,
    'org_edge_feat_dim': 768,
    'node_initial_dims': 128,
    'edge_initial_dims': 128,
    'neighbors':20,
    'num_edge_type': 14 + 1

}

config['edge_feat_emb_num'] = int(np.sum(config['edge_feat_dim']))
config['trip_feat_emb_num'] = int(np.sum(config['trip_feat_dim']))
config['node_history_feat_emb_num'] = int(np.sum(config['node_history_feat_dim']))
config['pair_feat_emb_num'] = int(np.sum(config['pair_feat_dim']))
config['label_feat_emb_num'] = int(np.sum(config['label_feat_dim']))
