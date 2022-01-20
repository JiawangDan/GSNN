
import torch.nn
import transformers
import constants
import pytorch_lightning as pyl
from graph_model.models import TGN
from graph_model.utils import MergeLayer_output, MergeLayer_concat1

class HierarchicalTransformer(torch.nn.Module):

    def __init__(self, config, edge_initial_feat_dict, node_initial_feat):
        super().__init__()

        self.config = config
        bert_config = transformers.BertConfig.from_dict(config)

        self.trip_encode_layers = torch.nn.ModuleList([
            transformers.BertLayer(bert_config)
            for _ in range(config['num_hidden_layers_trip'])])
        self.pair_encode_layers = torch.nn.ModuleList([
            transformers.BertLayer(bert_config)
            for _ in range(config['num_hidden_layers_pair'])])
        self.src_encode_layers = torch.nn.ModuleList([
            transformers.BertLayer(bert_config)
            for _ in range(config['num_hidden_layers_pair'])])
        self.dst_encode_layers = torch.nn.ModuleList([
            transformers.BertLayer(bert_config)
            for _ in range(config['num_hidden_layers_pair'])])

        self.trip_feat_encode_layers = torch.nn.ModuleList([
            transformers.BertLayer(bert_config)
            for _ in range(config['num_hidden_layers_feat'])])
        self.pair_feat_encode_layers = torch.nn.ModuleList([
            transformers.BertLayer(bert_config)
            for _ in range(config['num_hidden_layers_feat'])])
        self.src_feat_encode_layers = torch.nn.ModuleList([
            transformers.BertLayer(bert_config)
            for _ in range(config['num_hidden_layers_feat'])])
        self.dst_feat_encode_layers = torch.nn.ModuleList([
            transformers.BertLayer(bert_config)
            for _ in range(config['num_hidden_layers_feat'])])

        self.higher_encode_layers = torch.nn.ModuleList([
            transformers.BertLayer(bert_config)
            for _ in range(config['num_hidden_layers2'])])
                
        self.trip_feat_emb = torch.nn.Embedding(
            config['trip_feat_emb_num'],
            config['hidden_size'])
        self.pair_feat_emb = torch.nn.Embedding(
            config['pair_feat_emb_num'],
            config['hidden_size'])
        self.node_history_emb = torch.nn.Embedding(
            config['node_history_feat_emb_num'],
            config['hidden_size'])
        
        self.edge_feat_emb = torch.nn.Sequential(
            torch.nn.Embedding(
                config['edge_feat_emb_num'],
                config['hidden_size']),
            torch.nn.LayerNorm(
                config['hidden_size']))
        
        self.label_feat_emb = torch.nn.Embedding(
            config['label_feat_emb_num'],
            config['hidden_size'])

        self.head = torch.nn.Linear(
            config['hidden_size'] * 2, 1)

        self.proj_extra_trip = torch.nn.Sequential(
            torch.nn.Linear(config['extra_feat_dim'], config['hidden_size']),
            torch.nn.LayerNorm(config['hidden_size']))
        self.proj_extra_pair = torch.nn.Sequential(
            torch.nn.Linear(config['extra_feat_dim'], config['hidden_size']),
            torch.nn.LayerNorm(config['hidden_size']))
        self.proj_extra_src = torch.nn.Sequential(
            torch.nn.Linear(config['extra_feat_dim'], config['hidden_size']),
            torch.nn.LayerNorm(config['hidden_size']))
        self.proj_extra_dst = torch.nn.Sequential(
            torch.nn.Linear(config['extra_feat_dim'], config['hidden_size']),
            torch.nn.LayerNorm(config['hidden_size']))

        self.base_code_trip = torch.nn.Parameter(
            torch.zeros((1, 1, config['hidden_size'])))
        self.base_code_pair = torch.nn.Parameter(
            torch.zeros((1, 1, config['hidden_size'])))
        self.base_code_src = torch.nn.Parameter(
            torch.zeros((1, 1, config['hidden_size'])))
        self.base_code_dst = torch.nn.Parameter(
            torch.zeros((1, 1, config['hidden_size'])))


        ####graph_model
        self.graph_model = TGN(config, node_initial_feat, edge_initial_feat_dict)
        self.affinity_score = MergeLayer_output(config['hidden_size'], config['hidden_size'])
        self.merger_layer = MergeLayer_concat1(config['hidden_size'], config['hidden_size'])

        pass

    def encode_feat(self, layers, feat_emb):
        batch_size = feat_emb.shape[0]
        num = feat_emb.shape[1]
        feat_emb = feat_emb.reshape(-1, feat_emb.shape[-2], feat_emb.shape[-1])
        for layer in layers:
            feat_emb = layer(feat_emb)[0]
            pass
        
        feat_emb = feat_emb.reshape(
            batch_size, num, feat_emb.shape[-2], feat_emb.shape[-1])
        feat_emb = feat_emb.sum(dim=-2)
        
        return feat_emb

    def encode_seq(self, layers, emb, mask, base_code):
        # emb: [B, N, C]
        # mask: [B, N]
        B, N, C = emb.shape
        mask = torch.cat(
            (torch.ones((B, 1), device=emb.device),
             mask), dim=-1)
            
        mask_ = (1.0 - mask[:, None, None, :]) * -10000

        emb = torch.cat(
            (base_code.repeat(B, 1, 1), emb), dim=1)
        
        for layer in layers:
            emb = layer(emb, attention_mask=mask_)[0]
            pass
        # emb = emb * mask[:, :, None]
        # emb = emb.sum(dim=-2)
        emb = emb[:, 0, :]

        return emb
    
    def forward(self, edge_feat, label_feat, trip_feat, trip_mask, pair_feat, pair_mask,
                src_feat, src_mask, dst_feat, dst_mask, trip_feat_extra_b, pair_feat_extra_b,
                src_feat_extra_b, dst_feat_extra_b,
                user_graphfeat_initial, oppo_grapgfeat_initial, user_1hop_edge_orgfeature,
               user_2hop_edge_orgfeature, oppo_1hop_edge_orgfeature, oppo_2hop_edge_orgfeature,
               graph_src_id,graph_dst_id, labels_time):
        '''
        edge_feat: [B, EF]
        label_feat: [B, L, LF]
        trip_feat: [B, C, TF]
        trip_feat_extra_b: [B, C, BF]
        '''
        batch_size = label_feat.shape[0]
        trip_emb = self.trip_feat_emb(trip_feat)
        pair_emb = self.pair_feat_emb(pair_feat)
        src_emb = self.node_history_emb(src_feat)
        dst_emb = self.node_history_emb(dst_feat)

        trip_emb_extra = self.proj_extra_trip(trip_feat_extra_b)
        # trip_emb_extra: [B, C, H]
        pair_emb_extra = self.proj_extra_pair(pair_feat_extra_b)
        src_emb_extra = self.proj_extra_src(src_feat_extra_b)
        dst_emb_extra = self.proj_extra_dst(dst_feat_extra_b)
        
        trip_emb = torch.cat((trip_emb, trip_emb_extra[:, :, None, :]), dim=-2)
        pair_emb = torch.cat((pair_emb, pair_emb_extra[:, :, None, :]), dim=-2)
        
        # trip_emb: [B, C, TF, H]
        trip_emb = self.encode_feat(self.trip_feat_encode_layers, trip_emb)
        trip_emb = self.encode_seq(
            self.trip_encode_layers, trip_emb, trip_mask, self.base_code_trip)

        pair_emb = self.encode_feat(self.pair_feat_encode_layers, pair_emb)
        pair_emb = self.encode_seq(
            self.pair_encode_layers, pair_emb, pair_mask, self.base_code_pair)

        src_emb = torch.cat((src_emb, src_emb_extra[:, :, None, :]), dim=-2)
        dst_emb = torch.cat((dst_emb, dst_emb_extra[:, :, None, :]), dim=-2)

        src_emb = self.encode_feat(self.src_feat_encode_layers, src_emb)
        src_emb = self.encode_seq(
            self.src_encode_layers, src_emb, src_mask, self.base_code_src)
        
        dst_emb = self.encode_feat(self.dst_feat_encode_layers, dst_emb)
        dst_emb = self.encode_seq(
            self.dst_encode_layers, dst_emb, dst_mask, self.base_code_dst)

        edge_emb = self.edge_feat_emb(edge_feat).sum(dim=-2)
        # edge_emb: [B, H]
        # edge_emb.fill_(0)

        label_emb = self.label_feat_emb(label_feat).sum(dim=-2)
        # label_emb: [B, L, H]



        #### concat graph embedding
        source_node_embedding, dest_node_embedding, coattention_source_node_embedding, coattention_dest_node_embedding, source_max_time, dest_max_time = \
                    self.graph_model(graph_src_id, graph_dst_id, user_graphfeat_initial, oppo_grapgfeat_initial,
                         user_1hop_edge_orgfeature, user_2hop_edge_orgfeature,
                         oppo_1hop_edge_orgfeature, oppo_2hop_edge_orgfeature)

        labels_time = torch.reshape(labels_time, [batch_size, -1])
        delta_source = labels_time - torch.reshape(source_max_time, [-1, 1]) #[B,L]
        delta_dest = labels_time - torch.reshape(dest_max_time, [-1, 1])

        source_time_encoding = self.graph_model.time_encoder(delta_source) #[B,L, H]
        dest_time_encoding = self.graph_model.time_encoder(delta_dest)


        embs = torch.stack((trip_emb, pair_emb, edge_emb, src_emb, dst_emb,
                            source_node_embedding, dest_node_embedding,
                            coattention_source_node_embedding, coattention_dest_node_embedding), dim=1)
        # embs: [B, N, H]

        for layer in self.higher_encode_layers:
            embs = layer(embs)[0]
            pass

        # embs = embs[:, 0, :]
        #embs = embs.sum(dim=-2)
        # embs: [B, H]
        #print(embs.shape)   #[64, 128]

        #print(label_emb.shape) #[64, 64, 128]

        #embs_new = label_emb * embs[:, None, :].repeat(1, label_emb.shape[1], 1)
        #print(embs_new.shape)   #[64, 64, 128]

        # scores = torch.sum(
        #     embs_new, dim=-1)
        # scores: [B, L]

        embs =  self.merger_layer(embs.reshape(batch_size , -1))
        # [B, H]

        embs_new = torch.unsqueeze(embs, dim=1).repeat(1, label_emb.shape[1], 1)
        # [B, L ,H]

        scores = self.affinity_score(embs_new, label_emb, source_time_encoding, dest_time_encoding)
        scores = torch.squeeze(scores)

        return scores
    pass


class ModelLightningForPredict(pyl.LightningModule):
    def __init__(
            self, config, backbone):
        super().__init__()
        self.config = config
        self.backbone = backbone
        pass

    def setup(self, stage=None):
        pass
        
    def forward(self, batch):
        x = self.backbone(
            batch['edge_feat'],
            batch['label_feat'],
            batch['trip_feat'],
            batch['trip_mask'],
            batch['pair_feat'],
            batch['pair_mask']
        )
        return x

    
    pass
