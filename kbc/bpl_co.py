#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, sys
import pickle
import os.path as osp
import json

from tqdm import tqdm
import torch

from kbc.utils import QuerDAG
from kbc.utils import preload_env
from kbc.bpl_metrics import evaluation


def score_queries(args):
    mode = args.mode

    dataset = osp.basename(args.path)

    data_hard_path = osp.join(args.path, f'{dataset}_{mode}_hard.pkl')
    data_complete_path = osp.join(args.path, f'{dataset}_{mode}_complete.pkl')

    data_hard = pickle.load(open(data_hard_path, 'rb'))
    data_complete = pickle.load(open(data_complete_path, 'rb'))
    valid_heads_path = osp.join(args.path, 'kbc_data','valid_heads.pickle')
    valid_heads = pickle.load(open(valid_heads_path, 'rb'))
    valid_tails_path = osp.join(args.path, 'kbc_data','valid_tails.pickle')
    valid_tails = pickle.load(open(valid_tails_path, 'rb'))
    ent_id_path = osp.join(args.path, 'ind2ent.pkl')
    rel_id_path = osp.join(args.path, 'ind2rel.pkl')
    ent_id = pickle.load(open(ent_id_path, 'rb'))
    rel_id = pickle.load(open(rel_id_path, 'rb'))

    # Instantiate singleton KBC object
    preload_env(args.model_path, data_hard, args.chain_type, mode='hard', valid_heads=valid_heads, valid_tails=valid_tails)
    env = preload_env(args.model_path, data_complete, args.chain_type,
                      mode='complete', valid_heads=valid_heads, valid_tails=valid_tails)

    if 'SimplE' in args.model_path:
        model_type = 'SimplE'
    elif 'DistMult' in args.model_path:
        model_type = 'DistMult'
    # list of hard queries (queries[0]: 2865_5_-1_-1_63_-1234)
    queries = env.keys_hard
    # a dictionary of the form: {2865_5_-1_-1_63_-1234: [4822, 4398]]}
    test_ans_hard = env.target_ids_hard
    test_ans = env.target_ids_complete
    chains = env.chains
    kbc = env.kbc
    possible_heads_emb = env.possible_heads_emb
    possible_tails_emb = env.possible_tails_emb
    all_nodes_indices = torch.arange(kbc.model.sizes[0])
    # make a tensor of range (0, 14541)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    all_nodes_embs = kbc.model.entity_embeddings(all_nodes_indices.to(device))

    if args.reg is not None:
        env.kbc.regularizer.weight = args.reg

    disjunctive = args.chain_type in (QuerDAG.TYPE2_2_disj.value,
                                      QuerDAG.TYPE4_3_disj.value)

    if args.chain_type == QuerDAG.TYPE1_1.value:
        # scores = kbc.model.link_prediction(chains)
        # embedding of the lhs nodes
        s_emb = chains[0][0]
        #embedding of the relations
        p_emb = chains[0][1]

        scores_lst = []
        #no of queries
        nb_queries = s_emb.shape[0]
        for i in tqdm(range(nb_queries)):
            # embedding of the lhs node for the ith query
            batch_s_emb = s_emb[i, :].view(1, -1)
            # embedding of the relation for the ith query
            batch_p_emb = p_emb[i, :].view(1, -1)
            batch_chains = [(batch_s_emb, batch_p_emb, None)]
            # batch scores is a tensor containing the score of each node for being the target
            batch_scores = kbc.model.link_prediction(batch_chains)
            scores_lst += [batch_scores]
        #scores is a tensor of shape (no of queries, no of nodes) containing the score of each node for being the target for each query
        scores = torch.cat(scores_lst, 0)
# 2p and 3p
    elif args.chain_type in (QuerDAG.TYPE1_2.value, QuerDAG.TYPE1_3.value):
        scores = kbc.model.optimize_chains_bpl(chains, kbc.regularizer
                                           ,cov_anchor=args.cov_anchor,
                                            cov_var=args.cov_var,
                                             cov_target=args.cov_target, possible_heads_emb=possible_heads_emb
                                            , possible_tails_emb=possible_tails_emb,
                                            all_nodes_embs=all_nodes_embs
                                           ,model_type = model_type)
# 2i, 2u, 3i 
    elif args.chain_type in (QuerDAG.TYPE2_2.value, QuerDAG.TYPE2_2_disj.value,
                             QuerDAG.TYPE2_3.value):
        scores = kbc.model.optimize_intersections_bpl(chains, kbc.regularizer,
                                            disjunctive=disjunctive, cov_anchor=args.cov_anchor,
                                            cov_var=args.cov_var, cov_target=args.cov_target, possible_heads_emb=possible_heads_emb
                                            , possible_tails_emb=possible_tails_emb,
                                            all_nodes_embs=all_nodes_embs
                                            ,model_type = model_type)
# pi
    elif args.chain_type == QuerDAG.TYPE3_3.value:
        scores = kbc.model.optimize_3_3_bpl(chains, kbc.regularizer,
                                                  max_steps=args.max_steps,
                                                  lr=args.lr,
                                                  optimizer=args.optimizer,
                                                  norm_type=args.t_norm,
                                                  cov_anchor=args.cov_anchor,
                                            cov_var=args.cov_var, cov_target=args.cov_target, possible_heads_emb=possible_heads_emb
                                            , possible_tails_emb=possible_tails_emb,
                                            all_nodes_embs=all_nodes_embs
                                            ,model_type = model_type)
# ip and up
    elif args.chain_type in (QuerDAG.TYPE4_3.value,
                             QuerDAG.TYPE4_3_disj.value):
        scores = kbc.model.optimize_4_3_bpl(chains, kbc.regularizer,
                                        max_steps=args.max_steps,
                                        lr=args.lr,
                                        optimizer=args.optimizer,
                                        norm_type=args.t_norm,
                                        disjunctive=disjunctive, cov_anchor=args.cov_anchor,
                                            cov_var=args.cov_var, cov_target=args.cov_target, possible_heads_emb=possible_heads_emb
                                            , possible_tails_emb=possible_tails_emb,
                                            all_nodes_embs=all_nodes_embs
                                            ,model_type = model_type)
    else:
        raise ValueError(f'Uknown query type {args.chain_type}')


    return scores, queries, test_ans, test_ans_hard

def main(args):
    print("BPL OPTIMIZATION")
    print("dataset:", args.dataset)
    print("chain type:", args.chain_type)
    print("mode:", args.mode)
    print("cov_anchor:", args.cov_anchor)
    print("cov_var:", args.cov_var)
    print("cov_target:", args.cov_target)
    scores, queries, test_ans, test_ans_hard = score_queries(args)

    ent_id_path = osp.join(args.path, 'ind2ent.pkl')
    rel_id_path = osp.join(args.path, 'ind2rel.pkl')
    ent_id = pickle.load(open(ent_id_path, 'rb'))
    rel_id = pickle.load(open(rel_id_path, 'rb'))
    metrics = evaluation(scores, queries, test_ans, test_ans_hard, rel_id, ent_id, args.explain)
    
    print(metrics)

    model_name = osp.splitext(osp.basename(args.model_path))[0]
    reg_str = f'{args.reg}' if args.reg is not None else 'None'
    
    with open(f'cont_n={model_name}_t={args.chain_type}_r={reg_str}_m={args.mode}_lr={args.lr}_opt={args.optimizer}_ms={args.max_steps}.json', 'w') as f:
        json.dump(metrics, f)


if __name__ == "__main__":

    datasets = ['FB15k', 'FB15k-237', 'NELL']
    modes = ['valid', 'test', 'train']
    explains = ['no', 'yes']
    chain_types = [t.value for t in QuerDAG]

    t_norms = ['min', 'prod']

    parser = argparse.ArgumentParser(description="Complex Query Decomposition - Continuous Optimisation")
    parser.add_argument('path', help='Path to directory containing queries')
    parser.add_argument('--model_path', help="The path to the KBC model. Can be both relative and full")
    parser.add_argument('--dataset', choices=datasets, help="Dataset in {}".format(datasets))
    parser.add_argument('--mode', choices=modes, default='test',
                        help="Dataset validation mode in {}".format(modes))
    parser.add_argument('--explain', choices=explains, default='no',
                        help="Whether to store other top candidates")
    parser.add_argument('--chain_type', choices=chain_types, default=QuerDAG.TYPE1_1.value,
                        help="Chain type experimenting for ".format(chain_types))

    parser.add_argument('--t_norm', choices=t_norms, default='prod', help="T-norms available are ".format(t_norms))
    parser.add_argument('--reg', type=float, help='Regularization coefficient', default=None)
    parser.add_argument('--lr', type=float, default=0.1, help='Learning rate')
    parser.add_argument('--optimizer', type=str, default='adam',
                        choices=['adam', 'adagrad', 'sgd'])
    parser.add_argument('--max-steps', type=int, default=1000)
    parser.add_argument('--cov_anchor', type=float, default=0.1, help='Covariance of the anchor node')
    parser.add_argument('--cov_var', type=float, default=0.1, help='Covariance of the variable node')
    parser.add_argument('--cov_target', type=float, default=0.1, help='Covariance of the target node')

    main(parser.parse_args())
