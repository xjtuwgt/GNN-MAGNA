from dgl import DGLGraph
import numpy as np
import torch
from time import time
import logging

def comp_deg_norm(g):
    np.seterr(divide='ignore', invalid='ignore')
    g = g.local_var()
    in_deg = g.in_degrees(range(g.number_of_nodes())).float().numpy()
    norm = 1.0 / in_deg
    norm[np.isinf(norm)] = 0
    logging.info('Single nodes = {}'.format((norm==0).sum()))
    return norm

def build_graph_from_triples(num_nodes, num_relations, triples):
    """ Create a DGL graph. The graph is bidirectional because RGCN authors
        use reversed relations.
        This function also generates edge type and normalization factor
        (reciprocal of node incoming degree)
    """
    start = time()
    g = DGLGraph()
    g.add_nodes(num_nodes)
    #+++++++++++++++++++++++++
    src, rel, dst = triples
    inv_rel = rel + num_relations
    src, dst = np.concatenate((src, dst)), np.concatenate((dst, src))
    rel = np.concatenate((rel, inv_rel))
    # +++++++++++++++++++++++++
    node_id = torch.arange(0, num_nodes, dtype=torch.long).view(-1, 1)
    g.ndata.update({'n_id': node_id})
    g.add_edges(src, dst)
    g.edata['e_label'] = torch.from_numpy(rel).view(-1, 1)
    g.add_edges(g.nodes(), g.nodes(), {'e_label': torch.ones(g.number_of_nodes(), 1, dtype=torch.long) * 2 * num_relations})
    n_edges = g.number_of_edges()
    edge_id = torch.arange(0, n_edges, dtype=torch.long)
    g.edata['e_id'] = edge_id
    logging.info('Constructing graph takes {:.2f} seconds'.format(time() - start))
    return g

def build_graph_from_triples_without_loop(num_nodes, num_relations, triples):
    """ Create a DGL graph. The graph is bidirectional because RGCN authors
        use reversed relations.
        This function also generates edge type and normalization factor
        (reciprocal of node incoming degree)
    """
    start = time()
    g = DGLGraph()
    g.add_nodes(num_nodes)
    #+++++++++++++++++++++++++
    src, rel, dst = triples
    inv_rel = rel + num_relations
    src, dst = np.concatenate((src, dst)), np.concatenate((dst, src))
    rel = np.concatenate((rel, inv_rel))
    # +++++++++++++++++++++++++
    node_id = torch.arange(0, num_nodes, dtype=torch.long).view(-1, 1)
    g.ndata.update({'n_id': node_id})
    g.add_edges(src, dst)
    g.edata['e_label'] = torch.from_numpy(rel).view(-1, 1)
    zero_deg_nodes_idx = g.in_degrees(np.arange(0, g.number_of_nodes())) == 0
    zero_deg_node_num = zero_deg_nodes_idx.sum().item()
    zero_deg_nodes = torch.arange(0, num_nodes, dtype=torch.long)[zero_deg_nodes_idx]
    g.add_edges(zero_deg_nodes, zero_deg_nodes, {'e_label': torch.ones(zero_deg_node_num, 1, dtype=torch.long) * 2 * num_relations})
    n_edges = g.number_of_edges()
    edge_id = torch.arange(0, n_edges, dtype=torch.long)
    g.edata['e_id'] = edge_id
    logging.info('Constructing graph takes {:.2f} seconds'.format(time() - start))
    return g

def build_graph_from_triples_directed(num_nodes: int, num_relations: int, triples, multi_graph=True) -> DGLGraph:
    """
    :param num_nodes:
    :param num_relations:
    :param triples: 3 x number of edges
    :return:
    """
    start = time()
    g = DGLGraph(multigraph=multi_graph)
    g.add_nodes(num_nodes)
    src, rel, dst = triples
    node_id = torch.arange(0, num_nodes, dtype=torch.long).view(-1, 1)
    g.ndata.update({'n_id': node_id})
    g.add_edges(src, dst)
    # ===================================================================
    rel = torch.from_numpy(rel).view(-1, 1)
    g.edata['e_label'] = rel
    g.add_edges(g.nodes(), g.nodes(), {'e_label': torch.ones(g.number_of_nodes(), 1, dtype=torch.long) * num_relations})

    n_edges = g.number_of_edges()
    edge_id = torch.arange(0, n_edges, dtype=torch.long)
    g.edata['e_id'] = edge_id
    # ===================================================================
    logging.info('Constructing graph takes {:.2f} seconds'.format(time() - start))
    return g


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