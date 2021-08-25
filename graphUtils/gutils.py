import numpy as np
import torch
import random
import dgl
import os
from time import time
from dgl import DGLGraph

def set_seeds(seed):
    "set random seeds"
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # some cudnn methods can be random even after fixing the seed
    # unless you tell it to be deterministic
    torch.backends.cudnn.deterministic = True
    dgl.random.seed(seed)

def deep_dgl_graph_copy(graph: DGLGraph):
    start = time()
    copy_graph = DGLGraph()
    copy_graph.add_nodes(graph.number_of_nodes())
    graph_edges = graph.edges()
    copy_graph.add_edges(graph_edges[0], graph_edges[1])
    for key, value in graph.edata.items():
        copy_graph.edata[key] = value
    for key, value in graph.ndata.items():
        copy_graph.ndata[key] = value
    print('Graph copy take {:.2f} seconds'.format(time() - start))
    return copy_graph

def add_edge_with_same_labels(graph: DGLGraph, train_mask, node_labels, add_ratio=0.025):
    copy_graph = deep_dgl_graph_copy(graph=graph)
    def train_infor_extraction(train_mask, node_labels):
        train_labels = node_labels[train_mask]
        train_idexes = torch.arange(0, node_labels.shape[0])[train_mask]
        return train_labels, train_idexes
    train_labels, train_idxes = train_infor_extraction(train_mask, node_labels)
    pre_edge_number = copy_graph.number_of_edges()
    all_pairs_number = 0
    for i in range(train_labels.shape[0]):
        for j in range(i+1, train_labels.shape[0]):
            idx_i, idx_j = train_idxes[i], train_idxes[j]
            if train_labels[i] == train_labels[j] and ((not copy_graph.has_edge_between(idx_i, idx_j)) and (not copy_graph.has_edge_between(idx_j, idx_i))):
                all_pairs_number = all_pairs_number + 1
                if np.random.rand() < add_ratio:
                    copy_graph.add_edge(idx_i, idx_j)
                    copy_graph.add_edge(idx_j, idx_i)
    post_edge_number = copy_graph.number_of_edges()
    print('Adding {} edges, all same pairs {}'.format(post_edge_number - pre_edge_number, all_pairs_number * 2))
    return copy_graph


def reorginize_self_loop_edges(graph: DGLGraph):
    g_src, g_dest = graph.all_edges()
    s2d_loop = g_src - g_dest
    src, dest = g_src[s2d_loop != 0], g_dest[s2d_loop != 0]
    graph_reorg = DGLGraph()
    graph_reorg.add_nodes(graph.number_of_nodes())
    #+++++++
    upper_diag = dest - src
    half_src, half_dest = src[upper_diag > 0], dest[upper_diag > 0]
    graph_reorg.add_edges(half_src, half_dest)
    graph_reorg.add_edges(half_dest, half_src)
    #+++++++
    self_loop_edge_number = (s2d_loop ==0).sum().item()
    if self_loop_edge_number > 0:
        self_src, self_dest = g_src[s2d_loop == 0], g_dest[s2d_loop == 0]
        graph_reorg.add_edges(self_src, self_dest)
    for key, value in graph.ndata.items():
        graph_reorg.ndata[key] = value
    return graph_reorg, self_loop_edge_number


