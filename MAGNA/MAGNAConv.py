import torch
from torch import nn
from dgl.nn.pytorch.utils import Identity
import torch.nn.functional as F
from dgl.nn.pytorch.softmax import edge_softmax
from dgl import DGLGraph
import dgl.function as fn
# from MAGNA.layernormalization import RMSLayerNorm as LayerNorm
from MAGNA.layernormalization import STDLayerNorm as LayerNorm

class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, model_dim, d_hidden, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(model_dim, d_hidden)
        self.w_2 = nn.Linear(d_hidden, model_dim)
        self.dropout = nn.Dropout(dropout)
        self.init()

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))

    def init(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.w_1.weight, gain=gain)
        nn.init.xavier_normal_(self.w_2.weight, gain=gain)

class MAGNALayer(nn.Module):
    def __init__(self,
                 in_feats: int,
                 hidden_dim: int,
                 num_heads: int,
                 alpha,
                 hop_num,
                 feat_drop,
                 attn_drop,
                 topk_type='local',
                 top_k=-1,
                 layer_norm=True,
                 feed_forward=True,
                 head_tail_shared=True,
                 negative_slope=0.2):
        """
        """
        super(MAGNALayer, self).__init__()
        self.topk_type = topk_type
        self._in_feats = in_feats
        self._out_feats = hidden_dim
        self._num_heads = num_heads
        self.alpha = alpha
        self.hop_num = hop_num
        self.top_k = top_k ## FOR dense graph, edge selection
        self.head_tail_shared = head_tail_shared
        self.layer_norm = layer_norm
        self.feed_forward = feed_forward
        self._att_dim = hidden_dim // num_heads

        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        if self.head_tail_shared:
            self.fc = nn.Linear(in_feats, self._out_feats, bias=False)
        else:
            self.fc_head = nn.Linear(in_feats, self._out_feats, bias=False)
            self.fc_tail = nn.Linear(in_feats, self._out_feats, bias=False)
            self.fc = nn.Linear(in_feats, self._out_feats, bias=False)
        self.fc_out = nn.Linear(self._out_feats, self._out_feats, bias=False)
        if in_feats != self._out_feats:
            self.res_fc = nn.Linear(in_feats, self._out_feats, bias=False)
        else:
            self.res_fc = Identity()

        self.attn_h = nn.Parameter(torch.FloatTensor(size=(1, self._num_heads, self._att_dim)), requires_grad=True)
        self.attn_t = nn.Parameter(torch.FloatTensor(size=(1, self._num_heads, self._att_dim)), requires_grad=True)
        self.graph_norm = LayerNorm(num_features=in_feats)  # entity feature normalization
        self.feed_forward = PositionwiseFeedForward(model_dim=self._out_feats, d_hidden=4 * self._out_feats)  # entity feed forward
        self.ff_norm = LayerNorm(num_features=self._out_feats)  # entity feed forward normalization
        self.reset_parameters()
        self.attention_mask_value = -1e20

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        gain = nn.init.calculate_gain('tanh')
        if isinstance(self.res_fc, nn.Linear):
            nn.init.xavier_normal_(self.res_fc.weight.data, gain=gain)
        nn.init.xavier_normal_(self.fc.weight.data, gain=gain)
        if not self.head_tail_shared:
            nn.init.xavier_normal_(self.fc_head.weight.data, gain=gain)
            nn.init.xavier_normal_(self.fc_tail.weight.data, gain=gain)
            nn.init.xavier_normal_(self.fc_out.weight.data, gain=gain)
        nn.init.xavier_normal_(self.attn_t, gain=gain)
        nn.init.xavier_normal_(self.attn_h, gain=gain)


    def forward(self, graph: DGLGraph, features, drop_edge_ids=None):
        ###Attention computation: pre-normalization structure
        graph = graph.local_var()
        if self.layer_norm:
            h = self.graph_norm(features)
        else:
            h = features
        if self.head_tail_shared:
            feat = self.fc(self.feat_drop(h)).view(-1, self._num_heads, self._att_dim)
            feat_tanh = torch.tanh(feat)
            eh = (feat_tanh * self.attn_h).sum(dim=-1).unsqueeze(-1)
            et = (feat_tanh * self.attn_t).sum(dim=-1).unsqueeze(-1)
            graph.ndata.update({'ft': feat, 'eh': eh, 'et': et})
        else:
            feat_head = torch.tanh(self.fc_head(self.feat_drop(h))).view(-1, self._num_heads, self._att_dim)
            feat_tail = torch.tanh(self.fc_tail(self.feat_drop(h))).view(-1, self._num_heads, self._att_dim)
            feat = self.fc(self.feat_drop(h)).view(-1, self._num_heads, self._att_dim)
            eh = (feat_head * self.attn_h).sum(dim=-1).unsqueeze(-1)
            et = (feat_tail * self.attn_t).sum(dim=-1).unsqueeze(-1)
            graph.ndata.update({'ft': feat, 'eh': eh, 'et': et})
        graph.apply_edges(fn.u_add_v('eh', 'et', 'e'))
        attations = graph.edata.pop('e')
        attations = self.leaky_relu(attations)
        if drop_edge_ids is not None:
            attations[drop_edge_ids] = self.attention_mask_value

        if self.top_k <= 0:
            graph.edata['a'] = edge_softmax(graph, attations)
        else:
            if self.topk_type == 'local':
                graph.edata['e'] = attations
                attations = self.topk_attention(graph)
                graph.edata['a'] = edge_softmax(graph, attations)  ##return attention scores
            else:
                graph.edata['e'] = edge_softmax(graph, attations)
                graph.edata['a'] = self.topk_attention_softmax(graph)

        rst = self.ppr_estimation(graph=graph)
        rst = rst.flatten(1)
        rst = self.fc_out(rst)
        resval = self.res_fc(features)
        rst = resval + self.feat_drop(rst)
        if not self.feed_forward:
            return F.elu(rst)

        if self.layer_norm:
            rst_ff = self.feed_forward(self.ff_norm(rst))
        else:
            rst_ff = self.feed_forward(rst)
        rst = rst + self.feat_drop(rst_ff)
        return rst

    def ppr_estimation(self, graph: DGLGraph):
        graph = graph.local_var()
        feat_0 = graph.ndata.pop('ft')
        feat = feat_0
        attentions = graph.edata.pop('a')
        for _ in range(self.hop_num):
            graph.ndata['h'] = feat
            graph.edata['a_temp'] = self.attn_drop(attentions)
            graph.update_all(fn.u_mul_e('h', 'a_temp', 'm'), fn.sum('m', 'h'))
            feat = graph.ndata.pop('h')
            feat = (1.0 - self.alpha) * feat + self.alpha * feat_0
            feat = self.feat_drop(feat)
        return feat

    def topk_attention(self, graph: DGLGraph):
        graph = graph.local_var()# the graph should be added a self-loop edge
        def send_edge_message(edges):
            return {'m_e': edges.data['e']}
        def topk_attn_reduce_func(nodes):
            topk = self.top_k
            attentions = nodes.mailbox['m_e']
            neighbor_num = attentions.shape[1]
            if topk > neighbor_num:
                topk = neighbor_num
            topk_atts, _ = torch.topk(attentions, k=topk, dim=1)
            kth_attn_value = topk_atts[:, topk-1]
            return {'kth_e': kth_attn_value}

        graph.register_reduce_func(topk_attn_reduce_func)
        graph.register_message_func(send_edge_message)
        graph.update_all(message_func=send_edge_message, reduce_func=topk_attn_reduce_func)
        def edge_score_update(edges):
            scores, kth_score = edges.data['e'], edges.dst['kth_e']
            scores[scores < kth_score] = self.attention_mask_value
            return {'e': scores}
        graph.apply_edges(edge_score_update)
        topk_attentions = graph.edata.pop('e')
        return topk_attentions

    def topk_attention_softmax(self, graph: DGLGraph):
        graph = graph.local_var()
        def send_edge_message(edges):
            return {'m_e': edges.data['e'], 'm_e_id': edges.data['e_id']}
        def topk_attn_reduce_func(nodes):
            topk = self.top_k
            attentions = nodes.mailbox['m_e']
            edge_ids = nodes.mailbox['m_e_id']
            topk_edge_ids = torch.full(size=(edge_ids.shape[0], topk), fill_value=-1, dtype=torch.long)
            if torch.cuda.is_available():
                topk_edge_ids = topk_edge_ids.cuda()
            attentions_sum = attentions.sum(dim=2)
            neighbor_num = attentions_sum.shape[1]
            if topk > neighbor_num:
                topk = neighbor_num
            topk_atts, top_k_neighbor_idx = torch.topk(attentions_sum, k=topk, dim=1)
            top_k_neighbor_idx = top_k_neighbor_idx.squeeze(dim=-1)
            row_idxes = torch.arange(0, top_k_neighbor_idx.shape[0]).view(-1,1)
            top_k_attention = attentions[row_idxes, top_k_neighbor_idx]
            top_k_edge_ids = edge_ids[row_idxes, top_k_neighbor_idx]
            top_k_attention_norm = top_k_attention.sum(dim=1)
            topk_edge_ids[:, torch.arange(0,topk)] = top_k_edge_ids
            return {'topk_eid': topk_edge_ids, 'topk_norm': top_k_attention_norm}
        graph.register_reduce_func(topk_attn_reduce_func)
        graph.register_message_func(send_edge_message)
        graph.update_all(message_func=send_edge_message, reduce_func=topk_attn_reduce_func)
        topk_edge_ids = graph.ndata['topk_eid'].flatten()
        topk_edge_ids = topk_edge_ids[topk_edge_ids >=0]
        mask_edges = torch.zeros((graph.number_of_edges(), 1))
        if torch.cuda.is_available():
            mask_edges = mask_edges.cuda()
        mask_edges[topk_edge_ids] = 1
        attentions = graph.edata['e'].squeeze(dim=-1)
        attentions = attentions * mask_edges
        graph.edata['e'] = attentions.unsqueeze(dim=-1)
        def edge_score_update(edges):
            scores = edges.data['e']/edges.dst['topk_norm']
            return {'e': scores}
        graph.apply_edges(edge_score_update)
        topk_attentions = graph.edata.pop('e')
        return topk_attentions

    def forward_for_evaluataion(self, graph: DGLGraph, features, drop_edge_ids=None):
        ###Attention computation: pre-normalization structure
        graph = graph.local_var()
        h = self.graph_norm(features)
        if self.head_tail_shared:
            feat = self.fc(self.feat_drop(h)).view(-1, self._num_heads, self._att_dim)
            feat_tanh = torch.tanh(feat)
            eh = (feat_tanh * self.attn_h).sum(dim=-1).unsqueeze(-1)
            et = (feat_tanh * self.attn_t).sum(dim=-1).unsqueeze(-1)
            graph.ndata.update({'ft': feat, 'eh': eh, 'et': et})
        else:
            feat_head = torch.tanh(self.fc_head(self.feat_drop(h))).view(-1, self._num_heads, self._att_dim)
            feat_tail = torch.tanh(self.fc_tail(self.feat_drop(h))).view(-1, self._num_heads, self._att_dim)
            feat = self.fc(self.feat_drop(h)).view(-1, self._num_heads, self._att_dim)
            eh = (feat_head * self.attn_h).sum(dim=-1).unsqueeze(-1)
            et = (feat_tail * self.attn_t).sum(dim=-1).unsqueeze(-1)
            graph.ndata.update({'ft': feat, 'eh': eh, 'et': et})
        graph.apply_edges(fn.u_add_v('eh', 'et', 'e'))
        attations = graph.edata.pop('e')
        attations = self.leaky_relu(attations)
        eval_attention = edge_softmax(graph, attations)
        if drop_edge_ids is not None:
            attations[drop_edge_ids] = self.attention_mask_value

        if self.top_k <= 0:
            graph.edata['a'] = edge_softmax(graph, attations)
        else:
            if self.topk_type == 'local':
                graph.edata['e'] = attations
                attations = self.topk_attention(graph)
                graph.edata['a'] = edge_softmax(graph, attations)  ##return attention scores
            else:
                graph.edata['e'] = edge_softmax(graph, attations)
                graph.edata['a'] = self.topk_attention_softmax(graph)

        rst = self.ppr_estimation(graph=graph)
        rst = rst.flatten(1)

        rst = self.fc_out(rst)
        resval = self.res_fc(features)
        rst = resval + self.feat_drop(rst)

        rst_ff = self.feed_forward(self.ff_norm(rst))
        rst = rst + self.feat_drop(rst_ff)
        return rst, eval_attention