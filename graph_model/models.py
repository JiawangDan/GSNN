import torch.nn
from graph_model.time_encoding import TimeEncode
from graph_model.utils import MergeLayer_output, Node_feat_process, Edge_feat_process, Coattention_process
from graph_model.embedding_module import get_embedding_module


class TGN(torch.nn.Module):
    def __init__(self, config, node_features_init, edge_initial_feat_dict, neighbors =20, n_layers=2,
                 n_heads=2, dropout=0.2, dimension=128):
        super().__init__()
        self.cfg = config

        device = 'cuda:' + str(torch.cuda.current_device())
        # print(device)
        self.node_features_init = torch.from_numpy(node_features_init).float().to(device)
        self.edge_initial_feat_dict = torch.from_numpy(edge_initial_feat_dict).float().to(device)
        #self.node_features_init = torch.from_numpy(node_features_init).float()
        #self.edge_initial_feat_dict = torch.from_numpy(edge_initial_feat_dict).float()
        self.dims = dimension

        self.time_encoder = TimeEncode(dimension=self.dims)

        self.affinity_score = MergeLayer_output(self.dims, self.dims, drop_out=0.2)
        self.node_feat_process_fun = Node_feat_process(self.cfg['node_initial_dims'], self.dims, self.dims)
        self.edge_feat_process_fun = Edge_feat_process(self.cfg['edge_initial_dims'], self.dims, self.dims)
        self.edge_feat_process_concat_fun = Edge_feat_process(self.dims + self.cfg['org_edge_feat_dim'], self.dims,
                                                              self.dims)
        self.neighbors_num = neighbors
        self.n_layers = n_layers
        self.min_timestamp = torch.tensor(config['min_timestamp'], dtype=torch.int64)

        self.embedding_module = get_embedding_module(
                                                     time_encoder=self.time_encoder,
                                                     n_layers=self.n_layers,
                                                     n_node_features=self.dims,
                                                     n_edge_features=self.dims,
                                                     n_time_features=self.dims,
                                                     embedding_dimension=self.dims,
                                                     n_heads=n_heads, dropout=dropout)

        self.co_attention_module = Coattention_process(self.dims)

    def forward(self, sources, destinations, user_graphfeat_initial,
                oppo_grapgfeat_initial, user_1hop_edge_orgfeature,
                user_2hop_edge_orgfeature, oppo_1hop_edge_orgfeature, oppo_2hop_edge_orgfeature):

        sources_max_time = \
            torch.max(user_graphfeat_initial[:, -(self.neighbors_num * self.neighbors_num + self.neighbors_num):],
                      dim=1)[0]
        source_node_embedding_list = self.compute_temporal_embeddings(sources, user_graphfeat_initial, sources_max_time, user_1hop_edge_orgfeature, user_2hop_edge_orgfeature)
        dest_max_time = \
            torch.max(oppo_grapgfeat_initial[:, -(self.neighbors_num * self.neighbors_num + self.neighbors_num):],
                      dim=1)[0]
        dest_node_embedding_list = self.compute_temporal_embeddings(destinations, oppo_grapgfeat_initial, dest_max_time, oppo_1hop_edge_orgfeature, oppo_2hop_edge_orgfeature)

        source_node_embedding_b = source_node_embedding_list[:, 0, :]
        dest_node_embedding = dest_node_embedding_list[:, 0, :]

        coattention_source_node_embedding = self.co_attention_module(source_node_embedding_b, dest_node_embedding_list)
        coattention_dest_node_embedding = self.co_attention_module(dest_node_embedding, source_node_embedding_list)

        source_node_embedding = source_node_embedding_b
        dest_node_embedding = dest_node_embedding
        coattention_source_node_embedding = coattention_source_node_embedding
        coattention_dest_node_embedding = coattention_dest_node_embedding

        return source_node_embedding, dest_node_embedding, coattention_source_node_embedding, coattention_dest_node_embedding, sources_max_time, dest_max_time


    def compute_temporal_embeddings(self, node_id, graphfeature, timestamps, hop1_edge_orgfeature, hop2_edge_orgfeature):
        n_samples = len(node_id)

        timestamps = torch.max(torch.zeros_like(timestamps) ,  timestamps - self.min_timestamp)
        idx = 1
        neighbor_1hop_node = graphfeature[:, idx:idx + self.neighbors_num].long()
        idx += self.neighbors_num
        neighbor_2hop_node = graphfeature[:, idx: self.neighbors_num*self.neighbors_num+idx].long()
        idx += self.neighbors_num * self.neighbors_num
        neighbor_1hop_edgeidx = graphfeature[:, idx:idx + self.neighbors_num].long()
        idx += self.neighbors_num
        neighbor_2hop_edgeidx = graphfeature[:, idx:idx + self.neighbors_num*self.neighbors_num].long()
        idx += self.neighbors_num * self.neighbors_num
        neighbor_1hop_time = graphfeature[:, idx:idx + self.neighbors_num]
        neighbor_1hop_time = torch.max(torch.zeros_like(neighbor_1hop_time), neighbor_1hop_time - self.min_timestamp)

        idx += self.neighbors_num
        neighbor_2hop_time = graphfeature[:, idx:idx + self.neighbors_num*self.neighbors_num]
        neighbor_2hop_time = torch.max(torch.zeros_like(neighbor_2hop_time),
                                           neighbor_2hop_time - self.min_timestamp)

        source_node_feature =  self.node_features_init[node_id]
        source_node_feature = self.node_feat_process_fun(source_node_feature)
        neighbor_1hop_node_feature = self.node_features_init[neighbor_1hop_node]
        neighbor_1hop_node_feature = self.node_feat_process_fun(neighbor_1hop_node_feature)
        neighbor_2hop_node_feature = self.node_features_init[neighbor_2hop_node]
        neighbor_2hop_node_feature = self.node_feat_process_fun(neighbor_2hop_node_feature)

        neighbor_1hop_edge_orgfeature = hop1_edge_orgfeature  # 第一维为type特征， 其它为后续特征边
        neighbor_1hop_edge_feature = self.edge_initial_feat_dict[neighbor_1hop_edge_orgfeature[:, :, 0].long()]
        neighbor_1hop_edge_feature = self.edge_feat_process_fun(neighbor_1hop_edge_feature.float())
        neighbor_1hop_edge_feature = self.edge_feat_process_concat_fun(
            torch.cat((neighbor_1hop_edge_feature, neighbor_1hop_edge_orgfeature[:, :, 1:].float()), dim=-1))

        neighbor_2hop_edge_orgfeature = hop2_edge_orgfeature
        neighbor_2hop_edge_feature = self.edge_initial_feat_dict[neighbor_2hop_edge_orgfeature[:, :, 0].long()]
        neighbor_2hop_edge_feature = self.edge_feat_process_fun(neighbor_2hop_edge_feature.float())
        neighbor_2hop_edge_feature = self.edge_feat_process_concat_fun(
            torch.cat((neighbor_2hop_edge_feature, neighbor_2hop_edge_orgfeature[:, :, 1:].float()), dim=-1))


        node_embedding = self.embedding_module.compute_embedding(
                                                             timestamps=timestamps,
                                                             n_layers=self.n_layers,
                                                             n_neighbors=self.neighbors_num,
                                                             source_node_feature=source_node_feature,
                                                             neighbor_1hop_node_feature=neighbor_1hop_node_feature,
                                                             neighbor_2hop_node_feature=neighbor_2hop_node_feature,
                                                             neighbor_1hop_edge_feature=neighbor_1hop_edge_feature,
                                                             neighbor_2hop_edge_feature=neighbor_2hop_edge_feature,
                                                             neighbor_1hop_time=neighbor_1hop_time,
                                                             neighbor_2hop_time=neighbor_2hop_time,
                                                             neighbor_1hop_node=neighbor_1hop_node,
                                                             neighbor_2hop_node=neighbor_2hop_node
                                    )

        return node_embedding
