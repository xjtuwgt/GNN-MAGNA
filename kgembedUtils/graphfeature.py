
import sys
import os
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

from time import time
from dgl.contrib.sampling import NeighborSampler
from dgl.nodeflow import NodeFlow
from pandas import DataFrame
import pandas as pd
from dgl import DGLGraph
import numpy as np
from pandas import DataFrame
import logging
import os
from pandarallel import pandarallel
pandarallel.initialize(progress_bar=True)
from kgembedUtils.kgutils import build_graph_from_triples_directed
import torch

import argparse

def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description='Contexturalized node representation',
    )
    parser.add_argument('--data_path', type=str, default='../data/wn18rr')
    parser.add_argument('--hop_num', type=int, default=7)
    return parser.parse_args(args)

def context_extractor(g: DGLGraph, hop_num: int, num_workers=8):
    """
    :param g: whole graph
    :param radius: radius of each ball (bi-directional)
    :return: a set of sub-graphs (each ball is extracted for each node)
    """
    g.readonly(readonly_state=True)
    def NodeFlow2Tree(nf: NodeFlow):
        center = nf.layer_parent_nid(-1)[0].item()
        node_set = nf.layer_parent_nid(hop_num - 1).tolist()
        for i in range(1, hop_num):
            node_set_i = nf.layer_parent_nid(hop_num - i - 1).tolist()
            node_set = node_set + node_set_i
        node_list = list(set(node_set))
        return center, node_list
    expand_factor = g.number_of_nodes()
    out_tree_list = []
    for nf_out in NeighborSampler(g=g, expand_factor=expand_factor, batch_size=1, neighbor_type='out',
                              shuffle=False, num_hops=hop_num, num_workers=num_workers):
        center, out_nodes = NodeFlow2Tree(nf_out)
        out_tree_list.append((center, out_nodes))
        # print('out', center, out_nodes)
    out_tree = pd.DataFrame(out_tree_list, columns=['center', 'out_nodes'])
    in_tree_list = []
    for nf_in in NeighborSampler(g=g, expand_factor=expand_factor, batch_size=1, neighbor_type='in',
                                 shuffle=False, num_hops=hop_num, num_workers=num_workers):
        center, int_nodes = NodeFlow2Tree(nf_in)
        # print('in', center.item(), int_nodes)
        in_tree_list.append((center, int_nodes))
    in_tree = pd.DataFrame(in_tree_list, columns=['center', 'in_nodes'])
    context = pd.merge(left=out_tree, right=in_tree, on='center', how='inner')
    g.readonly(readonly_state=False)
    return context


def feature_extractor(context_df: DataFrame, num_relation, graph: DGLGraph):
    """
    :param tree_df: center, in_nodes, out_nodes
    :return:
    """
    def relation_distribution(row):
        center, in_nodes, out_nodes = row['center'], row['in_nodes'], row['out_nodes']
        sub_graph = graph.subgraph(in_nodes + out_nodes + [center])
        def sub_graph_feature(sub_graph):
            if sub_graph.number_of_edges() == 0:
                feature = np.random.random(num_relation + 1)
                feature[-1] = center
                return feature
            parent_eid = sub_graph.parent_eid
            relations = graph.edata['e_label'][parent_eid].squeeze().numpy()
            value, counts = np.unique(relations, return_counts=True)
            counts = counts / counts.sum()
            feature = np.zeros(num_relation + 1)
            feature[value] = counts
            feature[-1] = center
            return feature
        f_graph = torch.from_numpy(sub_graph_feature(sub_graph))
        return f_graph
    x = context_df.parallel_apply(relation_distribution, axis=1, result_type="expand")
    node_features = x.to_numpy()
    return node_features

def context_feature_extractor(graph, hop_num, num_relation):
    start = time()
    context_df = context_extractor(g=graph, hop_num=hop_num)
    context_df = feature_extractor(context_df=context_df, graph=graph, num_relation=num_relation)
    print('Graph node feature extraction takes {:.2f} seconds'.format(time()-start))
    return context_df

def read_triple(file_path, entity2id, relation2id):
    '''
    Read triples and map them into ids.
    '''
    triples = []
    with open(file_path) as fin:
        for line in fin:
            h, r, t = line.strip().split('\t')
            triples.append((entity2id[h], relation2id[r], entity2id[t]))
    return triples

def graph_construction(triples, num_entities, num_relations):
    graph, _ = build_graph_from_triples_directed(num_nodes=num_entities, num_relations=num_relations,
                                                 triples=np.array(triples, dtype=np.int64).transpose())
    return graph

def graph_node_feature_extractor(args):
    with open(os.path.join(args.data_path, 'entities.dict')) as fin:
        entity2id = dict()
        for line in fin:
            eid, entity = line.strip().split('\t')
            entity2id[entity] = int(eid)

    with open(os.path.join(args.data_path, 'relations.dict')) as fin:
        relation2id = dict()
        for line in fin:
            rid, relation = line.strip().split('\t')
            relation2id[relation] = int(rid)

    nentity = len(entity2id)
    nrelation = len(relation2id)

    args.nentity = nentity
    args.nrelation = nrelation

    print('Data Path: %s' % args.data_path)
    print('#entity: %d' % nentity)
    print('#relation: %d' % nrelation)

    train_triples = read_triple(os.path.join(args.data_path, 'train.txt'), entity2id, relation2id)
    print('#train: %d' % len(train_triples))
    valid_triples = read_triple(os.path.join(args.data_path, 'valid.txt'), entity2id, relation2id)
    print('#valid: %d' % len(valid_triples))
    test_triples = read_triple(os.path.join(args.data_path, 'test.txt'), entity2id, relation2id)
    print('#test: %d' % len(test_triples))

    graph = graph_construction(triples=train_triples, num_relations=nrelation, num_entities=nentity)
    node_features = context_feature_extractor(graph=graph, hop_num=args.hop_num, num_relation=nrelation)
    return node_features

def save_node_features(args, node_features):
    out_file = os.path.join(args.data_path, 'node_feature_' + str(args.hop_num))
    np.save(out_file, node_features)
    return

def main(args):
    node_features = graph_node_feature_extractor(args)
    save_node_features(args, node_features)

if __name__ == '__main__':
    main(parse_args())







