#!/usr/bin/python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import torch
import torch.nn as nn
import numpy as np
import math
from dgl.nn.pytorch.utils import Identity
import torch.nn.functional as F
from MAGNA_KGE.MAGNA_KGEncoder import MAGNAKGEncoder
from kge_losses.lossfunction import CESmoothLossKvsAll, CESmoothLossOnevsAll, KGESmoothCELoss
from torch.utils.data import DataLoader
from kge_dataloader.graphdataloader import TestDataset

class ConvE(nn.Module):
    def __init__(self, num_entities, args):
        super(ConvE, self).__init__()
        self.inp_drop = torch.nn.Dropout(args.input_drop)
        self.hidden_drop = torch.nn.Dropout(args.fea_drop)
        self.emb_dim1 = args.conv_embed_shape1 ##
        self.filter_size = args.conv_filter_size
        self.channels = args.conv_channels
        self.padding = 0
        self.stride = 1

        self.project_on = args.project_on == 1
        self.graph_on = args.graph_on == 1

        if self.graph_on:
            self.emb_dim2 = args.hidden_dim // self.emb_dim1
        else:
            if self.project_on:
                self.emb_dim2 = args.embed_dim // self.emb_dim1
            else:
                self.emb_dim2 = args.ent_embed_dim // self.emb_dim1

        self.conv1 = torch.nn.Conv2d(in_channels=1,
                                     out_channels=self.channels,
                                     kernel_size=(self.filter_size, self.filter_size),
                                     stride=self.stride,
                                     padding=self.padding,
                                     bias=args.conv_bias)
        self.bn0 = torch.nn.BatchNorm2d(1)
        self.bn1 = torch.nn.BatchNorm2d(self.channels)

        if self.graph_on:
            self.bn2 = torch.nn.BatchNorm1d(args.hidden_dim)
        else:
            if self.project_on:
                self.bn2 = torch.nn.BatchNorm1d(args.embed_dim)
            else:
                self.bn2 = torch.nn.BatchNorm1d(args.ent_embed_dim)
        self.register_parameter('b', nn.Parameter(torch.zeros(num_entities), requires_grad=True))

        conv_output_1 = int(((self.emb_dim1 * 2) - self.filter_size + (2 * self.padding)) / self.stride) + 1
        conv_output_2 = int((self.emb_dim2 - self.filter_size + (2 * self.padding)) / self.stride) + 1
        assert self.filter_size < self.emb_dim2 and self.filter_size < self.emb_dim1
        self.conv_hid_size = self.channels * conv_output_1 * conv_output_2 # as 3x3 filter is used
        if self.graph_on:
            self.fc = torch.nn.Linear(self.conv_hid_size, args.hidden_dim)
        else:
            if self.project_on:
                self.fc = torch.nn.Linear(self.conv_hid_size, args.embed_dim)
            else:
                self.fc = torch.nn.Linear(self.conv_hid_size, args.ent_embed_dim)

        self.initial_parameters()

    def initial_parameters(self):
        nn.init.kaiming_normal_(tensor=self.conv1.weight.data)
        nn.init.xavier_normal_(tensor=self.fc.weight.data)

    def score_computation(self, e1_emb, rel_emb, all_ent_emb):
        e1_embedded = e1_emb.view(-1, 1, self.emb_dim1, self.emb_dim2)
        rel_embedded = rel_emb.view(-1, 1, self.emb_dim1, self.emb_dim2)
        stacked_inputs = torch.cat([e1_embedded, rel_embedded], 2)

        stacked_inputs = self.bn0(stacked_inputs)
        x = self.inp_drop(stacked_inputs)
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.inp_drop(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        x = self.hidden_drop(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = torch.mm(x, all_ent_emb.transpose(1, 0))
        x += self.b.expand_as(x)
        scores = x
        return scores

    def forward(self, e1_emb, rel_emb, all_ent_emb):
        scores = self.score_computation(e1_emb=e1_emb, rel_emb=rel_emb, all_ent_emb=all_ent_emb)
        return scores

class DistMult(torch.nn.Module):
    def __init__(self, args):
        super(DistMult, self).__init__()
        self.inp_drop = torch.nn.Dropout(args.input_drop)

    def forward(self, e1_emb, rel_emb, all_ent_emb, inverse_rel_emb=None):
        e1_embedded = self.inp_drop(e1_emb)
        rel_embedded = self.inp_drop(rel_emb)
        if inverse_rel_emb is not None:
            inv_rel_embedded = self.inp_drop(inverse_rel_emb)
            comb_rel_embedded = (rel_embedded + inv_rel_embedded) * 0.5
            pred = torch.mm(e1_embedded*comb_rel_embedded, all_ent_emb.transpose(1,0))
        else:
            pred = torch.mm(e1_embedded*rel_embedded, all_ent_emb.transpose(1,0))
        return pred

class BiDistMult(torch.nn.Module):
    def __init__(self, args):
        super(BiDistMult, self).__init__()
        self.inp_drop = torch.nn.Dropout(args.input_drop)

    def forward(self, e1_emb, rel_emb, inverse_rel_emb, all_ent_emb):
        e1_embedded = self.inp_drop(e1_emb)
        rel_embedded = self.inp_drop(rel_emb)
        inv_rel_embedded = self.inp_drop(inverse_rel_emb)
        comb_rel_embedded = (rel_embedded + inv_rel_embedded) * 0.5
        pred = torch.mm(e1_embedded * comb_rel_embedded, all_ent_emb.transpose(1, 0))
        return pred

class TuckER(torch.nn.Module):
    def __init__(self, args):
        super(TuckER, self).__init__()
        self.input_dropout = torch.nn.Dropout(args.input_drop)
        self.hidden_dropout = torch.nn.Dropout(args.fea_drop)

        self.project_on = args.project_on == 1
        self.graph_on = args.graph_on == 1
        if self.graph_on:
            self.ent_emb_dim = args.hidden_dim
            self.rel_emb_dim = args.hidden_dim
        else:
            if self.project_on:
                self.ent_emb_dim = args.embed_dim
                self.rel_emb_dim = args.embed_dim
            else:
                self.ent_emb_dim = args.ent_embed_dim
                self.rel_emb_dim = args.rel_embed_dim
        if args.cuda:
            self.W = torch.nn.Parameter(
                torch.tensor(np.random.uniform(-1, 1, (self.rel_emb_dim, self.ent_emb_dim, self.ent_emb_dim)),
                             dtype=torch.float, device='cuda', requires_grad=True), requires_grad=True)
        else:
            self.W = torch.nn.Parameter(
                torch.tensor(np.random.uniform(-1, 1, (self.rel_emb_dim, self.ent_emb_dim, self.ent_emb_dim)),
                             dtype=torch.float, requires_grad=True), requires_grad=True)

        self.bn0 = torch.nn.BatchNorm1d(self.ent_emb_dim)
        self.bn1 = torch.nn.BatchNorm1d(self.ent_emb_dim)

    def forward(self, e1_emb, rel_emb, all_ent_emb):
        e1 = e1_emb
        x = self.bn0(e1_emb)
        x = self.input_dropout(x)
        x = x.view(-1, 1, e1.size(1))

        r = rel_emb
        W_mat = torch.mm(r, self.W.view(r.size(1), -1))
        W_mat = W_mat.view(-1, e1.size(1), e1.size(1))
        W_mat = self.hidden_dropout(W_mat)

        x = torch.bmm(x, W_mat)
        x = x.view(-1, e1.size(1))
        x = self.bn1(x)
        x = self.hidden_dropout(x)
        x = torch.mm(x, all_ent_emb.transpose(1, 0))
        return x


class KGEModel(nn.Module):
    def __init__(self, nentity, nrelation, ntriples, args):
        """
        Support ConvE, distMult and Cross Entropy and Binary Cross Entropy Loss
        :param nentity:
        :param nrelation:
        :param args:
        :param smooth_factor:
        :param graph_on:
        :param mask_on:
        :param bce_loss:
        """
        super(KGEModel, self).__init__()
        self.model_name = args.model
        self._nentity = nentity
        self._nrelation = nrelation
        self._ntriples = ntriples
        self._ent_emb_size = args.ent_embed_dim
        self._rel_emb_size = args.rel_embed_dim
        self.embed_dim = args.embed_dim
        self.hidden_dim = args.hidden_dim
        self.entity_embedding = nn.Parameter(torch.zeros(nentity, self._ent_emb_size), requires_grad=True)
        self.relation_embedding = nn.Parameter(torch.zeros(nrelation * 2 + 1, self._rel_emb_size), requires_grad=True) #inverse realtion + self-loop
        self.inp_drop = nn.Dropout(p=args.input_drop)
        self.feature_drop = nn.Dropout(p=args.fea_drop)
        self.mask_on = args.mask_on == 1
        self.graph_on = args.graph_on == 1
        self.project_on = args.project_on == 1

        if (not self.graph_on) and (self._ent_emb_size != self._rel_emb_size):
            self.project_on = True

        if self.project_on:
            self.ent_map = nn.Linear(self._ent_emb_size, self.embed_dim, bias=False)
            self.rel_map = nn.Linear(self._rel_emb_size, self.embed_dim, bias=False)
            graph_ent_in_dim = self.embed_dim
            graph_rel_in_dim = self.embed_dim
        else:
            self.ent_map, self.rel_map = Identity(), Identity()
            graph_ent_in_dim = self._ent_emb_size
            graph_rel_in_dim = self._rel_emb_size

        self.dag_entity_encoder = MAGNAKGEncoder(
            in_ent_dim=graph_ent_in_dim,
            in_rel_dim=graph_rel_in_dim,
            hidden_dim=args.hidden_dim,
            num_layers=args.layers,
            input_drop=args.input_drop,
            num_heads=args.num_heads,
            hop_num=args.hops,
            attn_drop=args.att_drop,
            feat_drop=args.fea_drop,
            negative_slope=args.slope,
            edge_drop=args.edge_drop,
            topk_type=args.topk_type,
            alpha=args.alpha,
            topk=args.top_k,
            ntriples=ntriples)

        if self.model_name == 'DistMult':
            self.score_function = DistMult(args=args)
        elif self.model_name == 'ConvE':
            self.score_function = ConvE(num_entities=nentity, args=args)
        elif self.model_name == 'TuckER':
            self.score_function = TuckER(args=args)
        elif self.model_name == 'BiDistMult':
            self.score_function = BiDistMult(args=args)
        else:
            raise ValueError('model %s not supported' % self.model_name)

        self.loss_type = args.loss_type
        if self.loss_type == 0:
            self.loss_function_onevsall = CESmoothLossOnevsAll(smoothing=args.gamma)
        else:
            self.loss_function_onevsall = None

        if self.loss_type == 1:
            self.loss_function_kvsall = CESmoothLossKvsAll(smoothing=args.gamma)
        else:
            self.loss_function_kvsall = None

        self.regularization = args.regularization
        self.regularization_type = args.reg_type
        if self.regularization > 0:
            self.kge_loss_function = KGESmoothCELoss()
        else:
            self.kge_loss_function = None
        self.init()

    def init(self):
        sqrt_size = 6.0 / math.sqrt(self._ent_emb_size)
        nn.init.uniform_(tensor=self.entity_embedding, a=-sqrt_size, b=sqrt_size)
        sqrt_size = 6.0 / math.sqrt(self._rel_emb_size)
        nn.init.uniform_(tensor=self.entity_embedding, a=-sqrt_size, b=sqrt_size)
        if self.project_on and isinstance(self.rel_map, nn.Linear):
            nn.init.xavier_normal_(tensor=self.rel_map.weight.data, gain=1.414)
        if self.project_on and isinstance(self.ent_map, nn.Linear):
            nn.init.xavier_normal_(tensor=self.ent_map.weight.data, gain=1.414)

    def kg_encoder(self, graph, edge_ids=None):
        graph = graph.local_var()
        if not self.mask_on:
            edge_ids = None
        entity_embedder, relation_embedder = self.ent_map(self.entity_embedding), self.rel_map(self.entity_embedding)
        if self.graph_on:
            entity_embedder, relation_embedder = self.dag_entity_encoder(graph, entity_embedder,
                                                                         relation_embedder, edge_ids)
        return entity_embedder, relation_embedder

    def kge_loss_computation(self, sample, entity_embed=None, relation_embed=None):
        head_part, rel_part, tail_part = sample[:, 0], sample[:, 1], sample[:, 2]
        if entity_embed is None:
            entity_embedding = self.entity_embedding
        else:
            entity_embedding = entity_embed
        if relation_embed is None:
            relation_embedding = self.relation_embedding
        else:
            relation_embedding = relation_embed

        head_embed = torch.index_select(entity_embedding, dim=0, index=head_part)
        tail_embed = torch.index_select(entity_embedding, dim=0, index=tail_part)
        regularization = self.regularization * self.kge_loss_function(head_embed, tail_embed, relation_embedding, rel_part)
        return regularization

    def forward(self, sample, entity_embed, relation_embed, true_labels=None):
        head_part, rel_part, tail_part = sample[:, 0], sample[:, 1], sample[:, 2]
        relation = torch.index_select(relation_embed, dim=0, index=rel_part)
        ent_embed = torch.index_select(entity_embed, dim=0, index=head_part)
        labels = tail_part
        # +++++++++++
        if self.model_name != 'BiDistMult':
            scores = self.score_function(ent_embed, relation, entity_embed)
        else:
            first_rel = rel_part[0].item()
            if first_rel > self._nrelation:
                inv_relation = torch.index_select(relation_embed, dim=0, index=rel_part - self._nrelation)
            else:
                inv_relation = torch.index_select(relation_embed, dim=0, index=rel_part + self._nrelation)
            scores = self.score_function(ent_embed, relation, inv_relation, entity_embed)
        # +++++++++++
        if self.training:
            if self.loss_type == 0:
                loss = self.loss_function_onevsall(scores, labels)
            elif self.loss_type == 1:
                loss = self.loss_function_kvsall(scores, true_labels)
            else:
                ValueError('loss %s not supported' % self.loss_type)

            if self.regularization != 0.0:
                if self.regularization_type == 0:
                    regularization = self.kge_loss_computation(sample=sample)
                else:
                    regularization = self.kge_loss_computation(sample=sample,  entity_embed=entity_embed, relation_embed=relation_embed)
            else:
                regularization = torch.Tensor([0.0]).to(loss.device)
            loss = loss + regularization
            return loss + regularization, regularization
        else:
            return scores

    @staticmethod
    def train_step(model, graph, optimizer, train_iterator, args):
        model.train()
        optimizer.zero_grad()
        samples, true_labels, edge_ids, _ = next(train_iterator)
        if args.cuda:
            samples = samples.cuda()
            edge_ids = edge_ids.cuda()
            true_labels = true_labels.cuda()

        entity_embedder, relation_embedder = model.kg_encoder(graph=graph, edge_ids=edge_ids)
        loss, regularization = model(samples, entity_embedder, relation_embedder, true_labels=true_labels)
        loss.backward()
        optimizer.step()
        log = {
            'loss': loss.item(),
            'reg': regularization.item()
        }
        return log

    @staticmethod
    def test_step(model, graph, test_triples, all_true_triples, args, load_mode=None):
        '''
                Evaluate the model on test or valid datasets
        '''
        model.eval()
        # Otherwise use standard (filtered) MRR, MR, HITS@1, HITS@3, and HITS@10 metrics
        # Prepare dataloader for evaluation
        test_dataloader_head = DataLoader(
            TestDataset(
                test_triples,
                all_true_triples,
                args.nentity,
                args.nrelation,
                'head-batch'
            ),
            batch_size=args.test_batch_size,
            num_workers=1,
            collate_fn=TestDataset.collate_fn
        )

        test_dataloader_tail = DataLoader(
            TestDataset(
                test_triples,
                all_true_triples,
                args.nentity,
                args.nrelation,
                'tail-batch'
            ),
            batch_size=args.test_batch_size,
            num_workers=1,
            collate_fn=TestDataset.collate_fn
        )


        if load_mode is not None:
            if load_mode == 'head-batch':
                test_dataset_list = [test_dataloader_head]
            else:
                test_dataset_list = [test_dataloader_tail]
        else:
            test_dataset_list = [test_dataloader_head, test_dataloader_tail]

        logs = []

        step = 0
        # total_steps = sum([len(dataset) for dataset in test_dataset_list])

        with torch.no_grad():
            entity_embedder, relation_embedder = model.kg_encoder(graph=graph)
            for test_dataset in test_dataset_list:
                # for positive_sample, _, filter_bias, mode in test_dataset:
                for positive_sample, filter_bias, _ in test_dataset:
                    if args.cuda:
                        positive_sample = positive_sample.cuda()
                        filter_bias = filter_bias.cuda()

                    batch_size = positive_sample.size(0)
                    score = model(positive_sample, entity_embedder, relation_embedder)
                    score = torch.sigmoid(score)
                    score += filter_bias

                    # Explicitly sort all the entities to ensure that there is no test exposure bias
                    argsort = torch.argsort(score, dim=1, descending=True)
                    positive_arg = positive_sample[:, 2]
                    for i in range(batch_size):
                        # Notice that argsort is not ranking
                        ranking = (argsort[i, :] == positive_arg[i]).nonzero()
                        assert ranking.size(0) == 1

                        # ranking + 1 is the true ranking used in evaluation metrics
                        ranking = 1 + ranking.item()
                        logs.append({
                            'MRR': 1.0 / ranking,
                            'MR': float(ranking),
                            'HITS@1': 1.0 if ranking <= 1 else 0.0,
                            'HITS@3': 1.0 if ranking <= 3 else 0.0,
                            'HITS@10': 1.0 if ranking <= 10 else 0.0,
                        })

                    if step % args.test_log_steps == 0:
                        # logging.info('Evaluating the model... (%d/%d)' % (step, total_steps))
                        logging.info('Evaluating the model... (%d)' % (step))

                    step += 1
                    torch.cuda.empty_cache()

        metrics = {}
        for metric in logs[0].keys():
            metrics[metric] = sum([log[metric] for log in logs]) / len(logs)

        return metrics