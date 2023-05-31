#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import numpy as np
import os.path as osp
import argparse
import pickle
import json
import sys
from kbc.chain_dataset_bpl import ChaineDataset
from kbc.chain_dataset_bpl import Chain

from kbc.utils import QuerDAG
from kbc.utils import preload_env
from kbc.utils import QuerDAG
from kbc.utils import DynKBCSingleton

from kbc.metrics import evaluation
from kbc.evaluate_bpl import evaluate_existential


def run(kbc_path, dataset_hard, dataset_name, t_norm='min', candidates=3,
 scores_normalize=0, kg_path=None, explain=False, user_likes =None, ent_id =None,
  quantifier=None, valid_heads=None, valid_tails=None, non_items_array=None, 
  cov_anchor=None, cov_var=None, cov_target=None):
    experiments = [t.value for t in QuerDAG]
    experiments.remove(QuerDAG.TYPE1_1.value)
    #experiments.remove(QuerDAG.TYPE1_2.value)
    experiments.remove(QuerDAG.TYPE2_2.value)
    experiments.remove(QuerDAG.TYPE2_2_disj.value)
    experiments.remove(QuerDAG.TYPE1_3.value)
    experiments.remove(QuerDAG.TYPE2_3.value)
    experiments.remove(QuerDAG.TYPE3_3.value)
    experiments.remove(QuerDAG.TYPE4_3.value)
    experiments.remove(QuerDAG.TYPE4_3_disj.value)
    experiments.remove(QuerDAG.TYPE1_3_joint.value)

    print(kbc_path, dataset_name, t_norm, candidates)


    path_entries = kbc_path.split('-')
    rank = path_entries[path_entries.index('rank') + 1] if 'rank' in path_entries else 'None'

    for exp in experiments:
        metrics = answer(kbc_path, dataset_hard, t_norm, exp, candidates, scores_normalize, kg_path, explain, user_likes, ent_id, quantifier=quantifier, valid_heads=valid_heads,
         valid_tails=valid_tails, non_items_array=non_items_array, cov_anchor=cov_anchor, cov_var=cov_var, cov_target=cov_target)

        with open(f'topk_d={dataset_name}_t={t_norm}_e={exp}_rank={rank}_k={candidates}_sn={scores_normalize}.json', 'w') as fp:
            json.dump(metrics, fp)
    return


def answer(kbc_path, dataset_hard, t_norm='min', query_type=QuerDAG.TYPE1_2, candidates=3, scores_normalize = 0, kg_path=None, 
explain=False, user_likes=None, ent_id=None, quantifier=None, valid_heads=None, valid_tails=None, non_items_array=None, cov_anchor=None, cov_var=None, cov_target=None):
    # takes each query chain, creates instruction on what type it is, and replaces each entity with its embedding
    env = preload_env(kbc_path, dataset_hard, query_type, mode='hard', kg_path=kg_path, explain=explain, valid_heads=valid_heads, valid_tails=valid_tails, ent_id=ent_id)
    #env = preload_env(kbc_path, dataset_complete, query_type, mode='complete', explain=explain, valid_heads=valid_heads, valid_tails=valid_tails)

    # tells us how many parts there are in each query
    if '1' in env.chain_instructions[-1][-1]:
        part1, part2 = env.parts
    elif '2' in env.chain_instructions[-1][-1]:
        part1, part2, part3 = env.parts

    kbc = env.kbc
    chains = env.chains

##########

    queries = env.keys_hard
    if quantifier == 'existential':
        print("existential: cov_anchor:", cov_anchor)
        print("existential: cov_var:", cov_var)
        print("existential: cov_target:", cov_target)
        scores = kbc.model.query_answering_BF_Exist(env, candidates, t_norm=t_norm , batch_size=1, scores_normalize = scores_normalize, explain=explain)

    elif quantifier == 'marginal_ui':
        print("Marginal UI: cov_anchor:", cov_anchor)
        print("Marginal UI: cov_var:", cov_var)
        print("Marginal UI: cov_target:", cov_target)
        scores = kbc.model.query_answering_BF_Marginal_UI(env, candidates, t_norm=t_norm ,
         batch_size=1, scores_normalize = scores_normalize, explain=explain, cov_anchor=cov_anchor, cov_var=cov_var, cov_target=cov_target)
 
    elif quantifier == 'marginal_i':
        print("instantiated: cov_anchor:", cov_anchor)
        print("instantiated: cov_var:", cov_var)
        print("instantiated: cov_target:", cov_target)
        scores = kbc.model.query_answering_BF_Instantiated(env, candidates, t_norm=t_norm ,
         batch_size=1, scores_normalize = scores_normalize, explain=explain, cov_anchor=cov_anchor, cov_var=cov_var, cov_target=cov_target)
    
    test_ans_hard = env.target_ids_hard
    test_ans = 	env.target_ids_complete
    metrics = evaluate_existential(env, scores, user_likes, non_items_array)
  
    #metrics = evaluation(scores, queries, test_ans, test_ans_hard)
    print(f'{quantifier}', metrics)
    sys.exit()

    return metrics


if __name__ == "__main__":

    big_datasets = ['Bio','FB15K', 'WN', 'WN18RR', 'FB237', 'YAGO3-10']
    datasets = big_datasets
    dataset_modes = ['valid', 'test', 'train']

    chain_types = [QuerDAG.TYPE1_1.value, QuerDAG.TYPE1_2.value, QuerDAG.TYPE2_2.value, QuerDAG.TYPE1_3.value,
                   QuerDAG.TYPE1_3_joint.value, QuerDAG.TYPE2_3.value, QuerDAG.TYPE3_3.value, QuerDAG.TYPE4_3.value,
                   'All', 'e']

    t_norms = ['min', 'product']
    normalize_choices = ['0', '1']
    quantifiers = ['existential', 'marginal_i', 'marginal_ui']

    parser = argparse.ArgumentParser(
    description="Complex Query Decomposition - Beam"
    )

    parser.add_argument('path', help='Path to directory containing queries')

    parser.add_argument(
    '--model_path',
    help="The path to the KBC model. Can be both relative and full"
    )

    parser.add_argument(
    '--dataset',
    help="The pickled Dataset name containing the chains"
    )

    parser.add_argument(
    '--mode', choices=dataset_modes, default='test',
    help="Dataset validation mode in {}".format(dataset_modes)
    )

    parser.add_argument(
    '--scores_normalize', choices=normalize_choices, default='0',
    help="A normalization flag for atomic scores".format(chain_types)
    )

    parser.add_argument(
    '--t_norm', choices=t_norms, default='min',
    help="T-norms available are ".format(t_norms)
    )

    parser.add_argument(
    '--candidates', default=5,
    help="Candidate amount for beam search"
    )

    parser.add_argument(
        '--quantifier', choices=quantifiers , default='existential',
        help="existential ,marginal_i, or marginal_ui"
    )

    parser.add_argument('--explain', default=False,
                        action='store_true',
                        help='Generate log file with explanations for 2p queries')
    parser.add_argument('--cov_anchor', type=float, default=0.1, help='Covariance of the anchor node')
    parser.add_argument('--cov_var', type=float, default=0.1, help='Covariance of the variable node')
    parser.add_argument('--cov_target', type=float, default=0.1, help='Covariance of the target node') 
    args = parser.parse_args()

    dataset = osp.basename(args.path)
    mode = args.mode

    data_hard_path = osp.join(args.path, f'{dataset}_{mode}_hard.pkl')
    #data_complete_path = osp.join(args.path, f'{dataset}_{mode}_complete.pkl')
    #print(data_complete_path)
    #sys.exit()

    data_hard = pickle.load(open(data_hard_path, 'rb'))
    #data_complete = pickle.load(open(data_complete_path, 'rb'))
    valid_heads_path = osp.join(args.path, 'kbc_data','valid_heads.pickle')
    valid_heads = pickle.load(open(valid_heads_path, 'rb'))
    valid_tails_path = osp.join(args.path, 'kbc_data','valid_tails.pickle')
    valid_tails = pickle.load(open(valid_tails_path, 'rb'))
    user_likes_path = osp.join(args.path, 'user_likes.pickle')
    user_likes = pickle.load(open(user_likes_path, 'rb'))
    ent_id = pickle.load(open(osp.join(args.path, 'ent_id.pickle'), 'rb'))
    rel_id = pickle.load(open(osp.join(args.path, 'rel_id.pickle'), 'rb'))
    non_items_array = np.load(osp.join(args.path, 'non_items_array.npy'))

    print("Beam:")


    candidates = int(args.candidates)
    run(args.model_path, data_hard,
        dataset, t_norm=args.t_norm, candidates=candidates,
        scores_normalize=int(args.scores_normalize),
        kg_path=args.path, explain=args.explain, user_likes=user_likes, ent_id=ent_id,
        quantifier=args.quantifier, valid_heads=valid_heads, valid_tails=valid_tails
        , non_items_array=non_items_array, cov_anchor=args.cov_anchor, cov_var=args.cov_var,
        cov_target=args.cov_target)
