import torch
from torch import nn
import numpy as np
import math

from graph_model.temporal_attention import TemporalAttentionLayer


class EmbeddingModule(nn.Module):
  def __init__(self, time_encoder, n_layers,
               n_node_features, n_edge_features, n_time_features, embedding_dimension,
               dropout):
    super(EmbeddingModule, self).__init__()
    # self.memory = memory
    self.time_encoder = time_encoder
    self.n_layers = n_layers
    self.n_node_features = n_node_features
    self.n_edge_features = n_edge_features
    self.n_time_features = n_time_features
    self.dropout = dropout
    self.embedding_dimension = embedding_dimension

  def compute_embedding(self, timestamps, n_layers,n_neighbors,source_node_feature,
                                                             neighbor_1hop_node_feature,
                                                             neighbor_2hop_node_feature,
                                                             neighbor_1hop_edge_feature,
                                                             neighbor_2hop_edge_feature,
                                                             neighbor_1hop_time,
                                                             neighbor_2hop_time,
                        neighbor_1hop_node,neighbor_2hop_node):
    pass


class GraphEmbedding(EmbeddingModule):
  def __init__(self, time_encoder, n_layers,
               n_node_features, n_edge_features, n_time_features, embedding_dimension,
               n_heads=2, dropout=0.1):
    super(GraphEmbedding, self).__init__( time_encoder, n_layers,
                                         n_node_features, n_edge_features, n_time_features,
                                         embedding_dimension, dropout)



  def compute_embedding(self, timestamps, n_layers, n_neighbors,source_node_feature,
                          neighbor_1hop_node_feature, neighbor_2hop_node_feature,
                            neighbor_1hop_edge_feature,neighbor_2hop_edge_feature,
                            neighbor_1hop_time,neighbor_2hop_time, neighbor_1hop_node, neighbor_2hop_node):
    b = source_node_feature.shape[0]
    source_node_features = source_node_feature
    source_nodes_time_embedding = self.time_encoder(torch.zeros_like( torch.unsqueeze(timestamps, dim=1)))

    if n_layers == 0:
        return source_node_features

    #外围聚合
    neighbor_mask = neighbor_2hop_node==0
    neighbor2_edge_deltas = torch.reshape(neighbor_1hop_time, [b*n_neighbors, -1]) - torch.reshape(neighbor_2hop_time, [b*n_neighbors, -1])
    neighbor2_nodes_time_embedding = self.time_encoder(neighbor2_edge_deltas)
    neighbor_source_time_embedding = self.time_encoder(torch.zeros_like(neighbor_1hop_time))
    neighbor_embeddings = self.aggregate(1, torch.reshape(neighbor_1hop_node_feature, [b*n_neighbors, -1]),
                                         torch.reshape(neighbor_source_time_embedding, [b*n_neighbors, 1, -1]),
                                         torch.reshape(neighbor_2hop_node_feature, [b*n_neighbors, n_neighbors, -1]),
                                         torch.reshape(neighbor2_nodes_time_embedding, [b*n_neighbors, n_neighbors, -1]),
                                         torch.reshape(neighbor_2hop_edge_feature,[b*n_neighbors, n_neighbors, -1]),
                                         torch.reshape(neighbor_mask, [b*n_neighbors, n_neighbors])
                                         )
    neighbor_embeddings = torch.reshape(neighbor_embeddings, [b, n_neighbors, -1])

    mask = neighbor_1hop_node == 0
    delta_time = torch.reshape(timestamps, [b, -1]) - torch.reshape(neighbor_1hop_time, [b, -1])
    neignbor_time_embedding = self.time_encoder(delta_time)
    source_node_embeddings = self.aggregate(1, source_node_features,
                                     source_nodes_time_embedding,
                                     neighbor_1hop_node_feature,
                                     neignbor_time_embedding,
                                     neighbor_1hop_edge_feature,
                                     mask
                                     )



    #中心聚合
    mask = neighbor_1hop_node == 0
    #delta_time = torch.reshape(timestamps, [b, -1]) - torch.reshape(neighbor_1hop_time, [b, -1])
    #neignbor_time_embedding = self.time_encoder(delta_time)
    node_embeddings = self.aggregate(0, source_node_embeddings,
                                         source_nodes_time_embedding,
                                         neighbor_embeddings,
                                         neignbor_time_embedding,
                                         neighbor_1hop_edge_feature,
                                         mask
                                         )

    all_node_embeddings = torch.cat([torch.reshape(node_embeddings, [b, 1, -1]), neighbor_embeddings], dim=1)
    return all_node_embeddings

  def aggregate(self, n_layers, source_node_features, source_nodes_time_embedding,
                neighbor_embeddings,
                edge_time_embeddings, edge_features, mask):
    return None


class GraphAttentionEmbedding(GraphEmbedding):
  def __init__(self, time_encoder, n_layers,
               n_node_features, n_edge_features, n_time_features, embedding_dimension,
               n_heads=2, dropout=0.1):
    super(GraphAttentionEmbedding, self).__init__(time_encoder, n_layers,
                                                  n_node_features, n_edge_features,
                                                  n_time_features,
                                                  embedding_dimension, n_heads, dropout)

    self.attention_models = torch.nn.ModuleList([TemporalAttentionLayer(
      n_node_features=n_node_features,
      n_neighbors_features=n_node_features,
      n_edge_features=n_edge_features,
      time_dim=n_time_features,
      n_head=n_heads,
      dropout=dropout,
      output_dimension=n_node_features)
      for _ in range(n_layers)])

  def aggregate(self, n_layer, source_node_features, source_nodes_time_embedding,
                neighbor_embeddings,
                edge_time_embeddings, edge_features, mask):
    attention_model = self.attention_models[n_layer - 1]

    source_embedding, _ = attention_model(source_node_features,
                                          source_nodes_time_embedding,
                                          neighbor_embeddings,
                                          edge_time_embeddings,
                                          edge_features,
                                          mask)

    return source_embedding



def get_embedding_module(time_encoder, n_layers, n_node_features, n_edge_features, n_time_features,
                         embedding_dimension, n_heads=2, dropout=0.1):

    return GraphAttentionEmbedding(time_encoder=time_encoder,
                                    n_layers=n_layers,
                                    n_node_features=n_node_features,
                                    n_edge_features=n_edge_features,
                                    n_time_features=n_time_features,
                                    embedding_dimension=embedding_dimension,
                                    n_heads=n_heads, dropout=dropout)




