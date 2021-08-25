import dgl
from dgl import DGLGraph
import torch

def build_karate_club_graph()->DGLGraph:
    g = dgl.DGLGraph()
    # add 34 nodes into the graph; nodes are labeled from 0~33
    g.add_nodes(34)
    # all 78 edges as a list of tuples
    edge_list = [(1, 0), (2, 0), (2, 1), (3, 0), (3, 1), (3, 2),
        (4, 0), (5, 0), (6, 0), (6, 4), (6, 5), (7, 0), (7, 1),
        (7, 2), (7, 3), (8, 0), (8, 2), (9, 2), (10, 0), (10, 4),
        (10, 5), (11, 0), (12, 0), (12, 3), (13, 0), (13, 1), (13, 2),
        (13, 3), (16, 5), (16, 6), (17, 0), (17, 1), (19, 0), (19, 1),
        (21, 0), (21, 1), (25, 23), (25, 24), (27, 2), (27, 23),
        (27, 24), (28, 2), (29, 23), (29, 26), (30, 1), (30, 8),
        (31, 0), (31, 24), (31, 25), (31, 28), (32, 2), (32, 8),
        (32, 14), (32, 15), (32, 18), (32, 20), (32, 22), (32, 23),
        (32, 29), (32, 30), (32, 31), (33, 8), (33, 9), (33, 13),
        (33, 14), (33, 15), (33, 18), (33, 19), (33, 20), (33, 22),
        (33, 23), (33, 26), (33, 27), (33, 28), (33, 29), (33, 30),
        (33, 31), (33, 32), (1,0), (1,0)]
    # add edges two lists of nodes: src and dst
    src, dst = tuple(zip(*edge_list))
    node_id = torch.arange(0, 34, dtype=torch.long).view(-1, 1)
    edge_id = torch.arange(0, len(edge_list), dtype=torch.long).view(-1, 1)
    g.ndata.update({'n_id': node_id})

    g.apply_edges(lambda edges: {'node_id_pair': torch.cat((edges.src['n_id'], edges.dst['n_id']), dim=-1)})

    g.add_edges(src, dst)
    g.edata.update({'e_id': edge_id})

    # print(g.edata[0])

    return g

g = build_karate_club_graph()
edge_idx = g.edge_id(1, 0, force_multi=True)

edge_ids = g.edata['e_id'][edge_idx]

print(edge_idx, edge_ids)

# print(g.edge_id(1, 0, force_multi=True))


# graph = build_karate_club_graph()
# print(graph)