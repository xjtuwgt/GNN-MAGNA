import sys
import os


PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

import numpy as np
import networkx as nx
import time
import torch
from dgl import DGLGraph
from dgl.data import load_data
from codes.MAGNA import MAGNA
from codes.utils import EarlyStopping
from graphUtils.gutils import set_seeds, reorginize_self_loop_edges
from codes.utils import save_config, save_model, remove_models
import argparse
import logging

def parse_args(args=None):
    parser = argparse.ArgumentParser(description='GAT')
    parser.add_argument('--cuda', default=False, action='store_true', help='use GPU')
    parser.add_argument('--do_train', default=True, action='store_true')
    parser.add_argument("--dataset", type=str, default='cora')
    parser.add_argument("--epochs", type=int, default=600,
                        help="number of training epochs")
    parser.add_argument("--num_heads", type=int, default=8,
                        help="number of hidden attention heads")
    parser.add_argument("--num_layers", type=int, default=2,
                        help="number of hidden layers")
    parser.add_argument("--top_k", type=int, default=5,
                        help="top k selection")
    parser.add_argument("--project_dim", type=int, default=-1,
                        help="projection dimension")
    parser.add_argument("--num_hidden", type=int, default=512,
                        help="number of hidden units")
    parser.add_argument("--residual", action="store_true", default=True,
                        help="use residual connection")
    parser.add_argument("--in_drop", type=float, default=.25,
                        help="input feature dropout")
    parser.add_argument("--attn_drop", type=float, default=0.5,
                        help="attention dropout")
    parser.add_argument("--edge_drop", type=float, default=.1,
                        help="edge dropout")
    parser.add_argument("--clip", type=float, default=1.0, help="grad_clip")
    parser.add_argument("--alpha", type=float, default=.15,
                        help="alpha")
    parser.add_argument("--hop_num", type=int, default=4,
                        help="hop number")
    parser.add_argument("--p_norm", type=int, default=0.0,
                        help="p_norm")
    parser.add_argument("--layer_norm", type=bool, default=True)
    parser.add_argument("--feed_forward", type=bool, default=True)
    parser.add_argument("--topk_type", type=str, default='local',
                        help="topk type")
    parser.add_argument("--patience", type=int, default=300, help="patience")
    parser.add_argument('-save', '--save_path', default='../models/', type=str)
    parser.add_argument("--lr", type=float, default=0.0002,
                        help="learning rate")
    parser.add_argument("--lr_reduce_factor", type=float, default=0.5, help="Please give a value for lr_reduce_factor")
    parser.add_argument("--lr_schedule_patience", type=float, default=25, help="Please give a value for lr_reduce_patience")
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help="weight decay")
    parser.add_argument('--negative_slope', type=float, default=0.2,
                        help="the negative slope of leaky relu")
    parser.add_argument('--self_loop', default=1, type=int, help='whether self-loop')
    parser.add_argument('--early-stop', action='store_true', default=False,
                        help="indicates whether to use early stop or not")
    parser.add_argument('--fastmode', action="store_true", default=False,
                        help="skip re-evaluate the validation set")
    parser.add_argument('--head_tail_shared', type=int, default=1,
                        help="random seed")
    parser.add_argument('--seed', type=int, default=2020,
                        help="random seed")
    args = parser.parse_args(args)
    return args

def accuracy(logits, labels):
    _, indices = torch.max(logits, dim=1)
    correct = torch.sum(indices == labels)
    return correct.item() * 1.0 / len(labels)


def evaluate(model, features, labels, mask):
    model.eval()
    with torch.no_grad():
        full_logits = model(features)
        logits = full_logits[mask]
        labels = labels[mask]
        return accuracy(logits, labels), full_logits

def set_logger(args):
    '''
    Write logs to checkpoint and console
    '''

    if args.do_train:
        log_file = os.path.join(args.save_path, 'train.log')
    else:
        log_file = os.path.join(args.save_path, 'test.log')

    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S',
        filename=log_file,
        filemode='w'
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

def preprocess(args):
    random_seed = args.seed
    set_seeds(random_seed)
    if args.save_path and not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    set_logger(args)
    logging.info("Model information...")
    for key, value in vars(args).items():
        logging.info('\t{} = {}'.format(key, value))

    model_folder_name = args2foldername(args)
    model_save_path = os.path.join(args.save_path, model_folder_name)
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)
    save_config(args, model_save_path)
    logging.info('Model saving path: {}'.format(model_save_path))
    return model_save_path

def args2foldername(args):
    folder_name = args.dataset + 'lr_' + str(round(args.lr, 5)) + \
                 "lyer_" + str(args.num_layers) + 'hs_' + str(args.num_heads) + \
                 'ho_' + str(args.hop_num) + 'hi_' + str(args.num_hidden) + 'tk_' + str(args.top_k) + \
                 'pd_' + str(args.project_dim) + 'ind_' + str(round(args.in_drop, 4)) + \
                 'att_' + str(round(args.attn_drop, 4)) + 'ed_' + str(round(args.edge_drop, 4)) + 'alpha_' + \
                 str(round(args.alpha, 3)) + 'decay_' + str(round(args.weight_decay, 6))
    return folder_name


def main(args):
    # load and preprocess dataset
    #+++++
    model_save_path = preprocess(args)
    #+++++
    data = load_data(args)
    features = torch.FloatTensor(data.features)
    labels = torch.LongTensor(data.labels)
    if hasattr(torch, 'BoolTensor'):
        train_mask = torch.BoolTensor(data.train_mask)
        val_mask = torch.BoolTensor(data.val_mask)
        test_mask = torch.BoolTensor(data.test_mask)
    else:
        train_mask = torch.ByteTensor(data.train_mask)
        val_mask = torch.ByteTensor(data.val_mask)
        test_mask = torch.ByteTensor(data.test_mask)
    all_zero_indexes = features.sum(dim=-1) == 0
    num_zero = all_zero_indexes.sum()
    if num_zero > 0:
        features = features + 1e-15
    num_feats = features.shape[1]
    n_classes = data.num_labels
    n_edges = data.graph.number_of_edges()
    logging.info("""----Data statistics------'
      #Edges %d
      #Classes %d 
      #Train samples %d
      #Val samples %d
      #Test samples %d""" %
          (n_edges, n_classes,
           train_mask.int().sum().item(),
           val_mask.int().sum().item(),
           test_mask.int().sum().item()))

    cuda = args.cuda
    if cuda:
        features = features.cuda()
        labels = labels.cuda()
        train_mask = train_mask.cuda()
        val_mask = val_mask.cuda()
        test_mask = test_mask.cuda()

    g = data.graph
    # add self loop
    if args.self_loop == 1:
        g.remove_edges_from(nx.selfloop_edges(g))
        g = DGLGraph(g)
        g.add_edges(g.nodes(), g.nodes())
    else:
        g = DGLGraph(g)
        zero_degree_idxes = g.in_degrees(np.arange(0, g.number_of_nodes())) == 0
        num_zero_degree = zero_degree_idxes.sum()
        if num_zero_degree > 0:
            zero_degree_nodes = torch.arange(0, g.number_of_nodes(), dtype=torch.long)[zero_degree_idxes]
            g.add_edges(zero_degree_nodes, zero_degree_nodes)

    # print(g.number_of_edges())
    # # print(g.in_degrees(np.arange(0, g.number_of_nodes())).float().median())
    #
    # # g = add_edge_with_same_labels(graph=g, train_mask=train_mask, node_labels=labels)
    g, self_loop_number = reorginize_self_loop_edges(graph=g)

    # print(g.number_of_edges(), self_loop_number, g.number_of_nodes())

    n_edges = g.number_of_edges()
    # add edge ids
    edge_id = torch.arange(0, n_edges, dtype=torch.long)
    g.edata.update({'e_id': edge_id})
    if cuda:
        for key, value in g.ndata.items():
            g.ndata[key] = value.cuda()
        for key, value in g.edata.items():
            g.edata[key] = value.cuda()
    # create model
    heads = [args.num_heads] * args.num_layers
    model = MAGNA(g=g,
                num_layers=args.num_layers,
                input_dim=num_feats,
                project_dim=args.project_dim,
                hidden_dim=args.num_hidden,
                num_classes=n_classes,
                heads=heads,
                feat_drop=args.in_drop,
                attn_drop=args.attn_drop,
                alpha=args.alpha,
                hop_num=args.hop_num,
                top_k=args.top_k,
                topk_type=args.topk_type,
                edge_drop=args.edge_drop,
                layer_norm=args.layer_norm,
                feed_forward=args.feed_forward,
                self_loop_number=self_loop_number,
                self_loop=(args.self_loop==1),
                head_tail_shared=(args.head_tail_shared == 1),
                negative_slope=args.negative_slope)

    if cuda:
        model = model.cuda()
    logging.info(model)
    if args.early_stop:
        stopper = EarlyStopping(patience=100)
    if cuda:
        model.cuda()
    loss_fcn = torch.nn.CrossEntropyLoss()
    ##++++++++++++++++++++++++++++++++++++++++++++
    weight_decay = args.weight_decay
    ##++++++++++++++++++++++++++++++++++++++++++++
    # use optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=args.epochs, eta_min=1e-8)
    dur = []
    best_valid_acc = 0.0
    test_acc = 0.0
    patience_count = 0
    best_model_name = None
    for epoch in range(args.epochs):
        model.train()
        if epoch >= 3:
            t0 = time.time()
        # forward
        logits = model(features)
        loss = loss_fcn(logits[train_mask], labels[train_mask])
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()
        scheduler.step()

        if epoch >= 3:
            dur.append(time.time() - t0)

        train_acc = accuracy(logits[train_mask], labels[train_mask])

        if args.fastmode:
            val_acc = accuracy(logits[val_mask], labels[val_mask])
        else:
            val_acc, logits = evaluate(model, features, labels, val_mask)
            if args.early_stop:
                if stopper.step(val_acc, model):
                    break

        if val_acc >= best_valid_acc:
            best_valid_acc = val_acc
            acc = accuracy(logits[test_mask], labels[test_mask])
            # ++++++++++++++++++++++++++++++++++++++++
            model_name = str(epoch) + '_vacc_' + str(best_valid_acc) + '_tacc_' + str(acc) + '.pt'
            if not cuda:
                model_path_name = os.path.join(model_save_path, model_name)
                save_model(model, model_save_path=model_path_name, step=epoch)
            best_model_name = model_name
            # ++++++++++++++++++++++++++++++++++++++++
            test_acc = acc
            patience_count = 0
        else:
            patience_count = patience_count + 1

        logging.info("Epoch {:05d} | Time(s) {:.4f} | Loss {:.4f} | TrainAcc {:.4f} |"
              " ValAcc {:.4f} | ETputs(KTEPS) {:.2f}".
              format(epoch, np.mean(dur), loss.item(), train_acc,
                     val_acc, n_edges / np.mean(dur) / 1000))

        if patience_count >= args.patience:
            break

    logging.info('\n')
    if args.early_stop:
        model.load_state_dict(torch.load('es_checkpoint.pt'))

    final_test_acc, _ = evaluate(model, features, labels, test_mask)
    logging.info('Best validation acc: {}\nBest test acc: {} \nFinal test acc: {}'.format(best_valid_acc, test_acc, final_test_acc))
    logging.info('Best model name: {}'.format(best_model_name))
    remove_models(model_save_path, best_model_name=best_model_name)

if __name__ == '__main__':
    main(parse_args())