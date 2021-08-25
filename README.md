This is an official implementation of Multi-hop Attention Graph Neural Network (IJCAI2021).

## Implementation Environment:

* python >= 3.7
* torch >= 1.4.0
* numpy == 1.17.2
* dgl-cu101 == 0.4.3

## Tasks: Node classification and KG completion
* Node classification
*      datasets: Cora, Citeseer, Pubmed (from DGL dataset)
*      codes --> node classification (train.py)

* Knowledge Graph Completion
*      datasets: Freebase237-15k and WordNet18RR (from https://github.com/villmow/datasets_knowledge_embedding)
*      codes_kge --> KG emebdding (all the pakages with 'kge' are used for Knowledge graph embedding) (runkge.py)
* parametertuning --> random search based hyper-parameter tuning


## Hyper-parameter demonstration for MAGNA

*  in_feats: dimension of input node features (for KG embedding: in_ent_feats, in_rel_feats represent the dimension of
*  entity embedding and relation embedding, respectively)
*  hidden_dim: dimension of hidden dimension
*  num_heads: head number of multi-head attention,
*  alpha: transition probability in personal page-rank (0.05 ~ 0.25, data dependent)
*  hop_num: number of iterations to approximate (3 ~ 10, data dependent)
*  feat_drop: dropout ratio over the feature (input node features)
*  attn_drop: dropout ratio over attention matrix
*  topk_type='local': if the degree as nodes is large. ('local': select top-k for each head, otherwise select top-k
*  shared by all heads)
*  top_k=-1: top-k neighbor selection, -1 means no 'top-k' selection is performed

## References
@inproceedings{wang2020multi,
  title={Multi-hop Attention Graph Neural Network},
  author={Wang, Guangtao and Ying, Zhitao and Huang, Jing and Leskovec, Jure},
  booktitle={International Joint Conference on Artificial Intelligence},
  year={2021}
}

## Related resources

* GAT in DGL: https://docs.dgl.ai/en/0.4.x/tutorials/models/1_gnn/9_gat.html
* APPNP: https://github.com/klicperajo/ppnp, https://github.com/dmlc/dgl/tree/master/examples/pytorch/appnp (DGL implementation)
* RotatE: https://github.com/DeepGraphLearning/KnowledgeGraphEmbedding