import torch
import torch.nn as nn
from dgl import DGLGraph
import numpy as np
from MAGNA.MAGNAConv import MAGNALayer

class MAGNA(nn.Module):
    def __init__(self,
                 g: DGLGraph,
                 num_layers: int,
                 input_dim: int,
                 hidden_dim: int,
                 hop_num: int,
                 alpha: float,
                 num_classes: int,
                 heads: list,
                 top_k:int,
                 feat_drop: float,
                 attn_drop: float,
                 negative_slope: float,
                 edge_drop: float,
                 topk_type: str,
                 self_loop_number: int,
                 undirected_graph=True,
                 self_loop=True,
                 layer_norm=True,
                 feed_forward=True,
                 head_tail_shared=True,
                 project_dim=-1):
        super(MAGNA, self).__init__()
        self.g = g
        self.gdt_layers = nn.ModuleList()
        self.self_loop = self_loop
        self.number_self_loops = self_loop_number
        self.undirected_graph = undirected_graph
        self.layer_norm = layer_norm
        self.feed_forward = feed_forward
        if project_dim > 1:
            self.project = nn.Linear(in_features=input_dim, out_features=project_dim)
            self.input_features = project_dim
        else:
            self.register_buffer('project', None)
            self.input_features = input_dim

        self.num_layers = num_layers
        self.edge_drop = edge_drop
        self.gdt_layers.append(MAGNALayer(in_feats=self.input_features, hop_num=hop_num, top_k=top_k, num_heads=heads[0], hidden_dim=hidden_dim,
                                                 topk_type=topk_type, layer_norm=self.layer_norm, feed_forward=self.feed_forward, head_tail_shared=head_tail_shared,
                                                 alpha=alpha, negative_slope=negative_slope, feat_drop=feat_drop, attn_drop=attn_drop))
        for l in range(1, self.num_layers):
            self.gdt_layers.append(MAGNALayer(in_feats=hidden_dim, hop_num=hop_num, hidden_dim=hidden_dim, num_heads=heads[l], top_k=top_k,
                                                     layer_norm=self.layer_norm, feed_forward=self.feed_forward, head_tail_shared=head_tail_shared,
                                                     topk_type=topk_type, alpha=alpha, negative_slope=negative_slope, feat_drop=feat_drop, attn_drop=attn_drop))
        self.classifier = nn.Linear(in_features=hidden_dim, out_features=num_classes)
        self.feat_drop_out = nn.Dropout(p=feat_drop)
        self.reset_parameters()

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        if isinstance(self.classifier, nn.Linear):
            nn.init.xavier_normal_(self.classifier.weight.data)
        if self.project is not None and isinstance(self.project, nn.Linear):
            nn.init.xavier_normal_(self.project.weight.data)

    def forward(self, inputs):
        number_edges = self.g.number_of_edges()
        if self.project is not None:
            h = self.project(self.feat_drop_out(inputs))
        else:
            h = inputs
        for l in range(self.num_layers):
            if self.undirected_graph:
                drop_edge_ids = self.get_drop_edge_pair_ids(number_edges - self.number_self_loops)
            else:
                drop_edge_ids = self.get_drop_edge_ids(number_edges - self.number_self_loops)
            h = self.gdt_layers[l](self.g, h, drop_edge_ids)
        logits = self.classifier(h)
        return logits

    def get_drop_edge_ids(self, number_edges):
        drop_edge_num = int(number_edges * self.edge_drop)
        if self.training:
            if drop_edge_num > 0:
                drop_edges_ids = np.random.choice(number_edges, drop_edge_num, replace=False)
                drop_edges_ids = torch.from_numpy(drop_edges_ids)
            else:
                drop_edges_ids = None
        else:
            drop_edges_ids = None
        return drop_edges_ids

    def get_drop_edge_pair_ids(self, number_edges):
        one_direct_number_edge = number_edges // 2
        drop_edge_num = int(one_direct_number_edge * self.edge_drop * 0.5)
        if self.training:
            if drop_edge_num > 0:
                drop_edges_ids = np.random.choice(one_direct_number_edge, drop_edge_num, replace=False)
                inv_drop_edges_ids = drop_edges_ids + one_direct_number_edge
                drop_edges_ids = np.concatenate([drop_edges_ids, inv_drop_edges_ids])
                drop_edges_ids = torch.from_numpy(drop_edges_ids)
            else:
                drop_edges_ids = None
        else:
            drop_edges_ids = None
        return drop_edges_ids

    def layer_attention_node_features(self, inputs):
        number_edges = self.g.number_of_edges()
        layer_node_features, layer_attentions = [], []
        if self.project is not None:
            h = self.project(self.feat_drop_out(inputs))
        else:
            h = inputs

        for l in range(self.num_layers):
            if self.undirected_graph:
                drop_edge_ids = self.get_drop_edge_pair_ids(number_edges - self.number_self_loops)
            else:
                drop_edge_ids = self.get_drop_edge_ids(number_edges - self.number_self_loops)
            h, attentions = self.gdt_layers[l].forward_for_evaluataion(self.g, h, drop_edge_ids)
            layer_node_features.append(h)
            layer_attentions.append(attentions)
        logits = self.classifier(h)
        return logits, layer_node_features, layer_attentions