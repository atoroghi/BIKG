#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os.path as osp
import argparse
import pickle
import json
import sys

from kbc.utils import QuerDAG
from kbc.utils import preload_env

from kbc.metrics import evaluation


def run(kbc_path, dataset_hard, dataset_complete, dataset_name, t_norm='min', candidates=3, scores_normalize=0, kg_path=None, explain=False, chain_type='1_2', reasoning_mode=None, seq = 'No',
    valid_heads = None, valid_tails=None, cov_anchor=None, cov_var=None,cov_target=None, ):

    chain_type_experiments = {'1_1': QuerDAG.TYPE1_1.value, '1_2': QuerDAG.TYPE1_2.value, '1_3': QuerDAG.TYPE1_3.value,
    '2_2':QuerDAG.TYPE2_2.value, '2_2_disj': QuerDAG.TYPE2_2_disj.value, '1_4': QuerDAG.TYPE1_4.value, '2_3': QuerDAG.TYPE2_3.value
    , '3_3': QuerDAG.TYPE3_3.value, '4_3': QuerDAG.TYPE4_3.value, '4_3_disj':QuerDAG.TYPE4_3_disj.value, '1_3_joint': QuerDAG.TYPE1_3_joint.value,
    '1_2_seq': QuerDAG.TYPE1_2_seq.value, '1_3_seq': QuerDAG.TYPE1_3_seq.value, '2_2_seq': QuerDAG.TYPE2_2_seq.value, '2_3_seq': QuerDAG.TYPE2_3_seq.value,
    '3_3_seq': QuerDAG.TYPE3_3_seq.value, '4_3_seq': QuerDAG.TYPE4_3_seq.value, '1_1_seq': QuerDAG.TYPE1_1_seq.value, '2_2_disj_seq': QuerDAG.TYPE2_2_disj_seq.value,'4_3_disj_seq': QuerDAG.TYPE4_3_disj_seq.value}
    experiments = [t.value for t in QuerDAG]

    for key in chain_type_experiments.keys():
        if key != chain_type:
            experiments.remove(chain_type_experiments[key])

    
    # experiments = [t.value for t in QuerDAG]
    # experiments.remove(QuerDAG.TYPE1_1.value)
    # experiments.remove(QuerDAG.TYPE1_3_joint.value)
    # #experiments.remove(QuerDAG.TYPE1_2.value)
    # experiments.remove(QuerDAG.TYPE2_2.value)
    # experiments.remove(QuerDAG.TYPE2_2_disj.value)
    # experiments.remove(QuerDAG.TYPE2_3.value)

    print(kbc_path, dataset_name, t_norm, candidates)



    path_entries = kbc_path.split('-')
    rank = path_entries[path_entries.index('rank') + 1] if 'rank' in path_entries else 'None'

    for exp in experiments:
        metrics = answer(kbc_path, dataset_hard, dataset_complete, t_norm, exp, candidates, scores_normalize, kg_path, explain, reasoning_mode, seq, valid_heads, valid_tails, cov_anchor, cov_var, cov_target)

        with open(f'topk_d={dataset_name}_t={t_norm}_e={exp}_rank={rank}_k={candidates}_sn={scores_normalize}.json', 'w') as fp:
            json.dump(metrics, fp)
    return


def answer(kbc_path, dataset_hard, dataset_complete, t_norm='min', query_type=QuerDAG.TYPE1_2, candidates=3, scores_normalize = 0, kg_path=None, explain=False, reasoning_mode='cqd', seq = 'No',
           valid_heads=None, valid_tails=None,cov_anchor=None, cov_var=None, cov_target=None):
    # takes each query chain, creates instruction on what type it is, and replaces each entity with its embedding
    env = preload_env(kbc_path, dataset_hard, query_type, mode='hard', kg_path=kg_path, explain=explain, valid_heads=valid_heads, valid_tails=valid_tails)
    env = preload_env(kbc_path, dataset_complete, query_type, mode='complete', kg_path=kg_path,explain=explain, valid_heads=valid_heads, valid_tails=valid_tails)
    if len(env.parts) == 2:
        part1, part2 = env.parts
    elif len(env.parts) == 3:
        part1, part2, part3 = env.parts
    elif len(env.parts) == 4:
        part1, part2, part3, part4 = env.parts
    elif len(env.parts) == 5:
        part1, part2, part3, part4, part5 = env.parts
    elif len(env.parts) == 6:
        part1, part2, part3, part4, part5, part6 = env.parts
    test_ans_hard = env.target_ids_hard

    # tells us how many parts there are in each query
    # if '1' in env.chain_instructions[-1][-1]:
    #     part1, part2 = env.parts
    # elif '2' in env.chain_instructions[-1][-1]:
    #     part1, part2, part3 = env.parts
    kbc = env.kbc
    if reasoning_mode == 'cqd':
        scores = kbc.model.query_answering_BF(env, candidates, t_norm=t_norm , batch_size=1, scores_normalize = scores_normalize, explain=explain)
    elif reasoning_mode == 'bayesian1':
        scores = kbc.model.query_answering_Bayesian1(env, candidates, t_norm=t_norm , batch_size=1, scores_normalize = scores_normalize, explain=explain,
        cov_anchor=cov_anchor, cov_var=cov_var, cov_target=cov_target)
    elif reasoning_mode == 'bayesian2':
        scores = kbc.model.query_answering_Bayesian2(env, candidates, t_norm=t_norm , batch_size=1, scores_normalize = scores_normalize, explain=explain,
        cov_anchor=cov_anchor, cov_var=cov_var, cov_target=cov_target)
    elif reasoning_mode == 'bayesian3':
        scores = kbc.model.query_answering_Bayesian3(env, candidates, t_norm=t_norm , batch_size=1, scores_normalize = scores_normalize, explain=explain,
        cov_anchor=cov_anchor, cov_var=cov_var, cov_target=cov_target)
    elif reasoning_mode == 'bayesian4':
        scores = kbc.model.query_answering_Bayesian4(env, candidates, t_norm=t_norm , batch_size=1, scores_normalize = scores_normalize, explain=explain,
        cov_anchor=cov_anchor, cov_var=cov_var, cov_target=cov_target)

    queries = env.keys_hard
    test_ans_hard = env.target_ids_hard
    test_ans = 	env.target_ids_complete
    # scores = torch.randint(1,1000, (len(queries),kbc.model.sizes[0]),dtype = torch.float).cuda()
    #
    metrics = evaluation(scores, queries, test_ans, test_ans_hard, seq)
    print(metrics)
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

    parser = argparse.ArgumentParser(
    description="Complex Query Decomposition - Beam"
    )

    parser.add_argument('path', help='Path to directory containing queries')

    parser.add_argument(
    '--model_path',
    help="The path to the KBC model. Can be both relative and full"
    )
    parser.add_argument(
        '--reasoning_mode', choices = ['cqd', 'bayesian1', 'bayesian2', 'bayesian3', 'bayesian4'],default='cqd'
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

    parser.add_argument('--explain', default=False,
                        action='store_true',
                        help='Generate log file with explanations for 2p queries')
    parser.add_argument('--seq', default='No',
                        help='Sequential or single shot')
    parser.add_argument('--chain_type', type=str, default='1_2', help='Chain type of experiment') 
    parser.add_argument('--cov_anchor', type=float, default=0.1, help='Covariance of the anchor node')
    parser.add_argument('--cov_var', type=float, default=0.1, help='Covariance of the variable node')
    parser.add_argument('--cov_target', type=float, default=0.1, help='Covariance of the target node') 

    args = parser.parse_args()

    dataset = osp.basename(args.path)
    mode = args.mode

    data_hard_path = osp.join(args.path, f'{dataset}_{mode}_hard.pkl')
    data_complete_path = osp.join(args.path, f'{dataset}_{mode}_complete.pkl')

    assert args.seq == 'yes' and 'seq' in args.chain_type or args.seq == 'no' and 'seq' not in args.chain_type, 'seq and chain_type must match'

    if args.seq == 'yes':
        data_hard_path = osp.join(args.path, f'{dataset}_{mode}_hard_seq.pkl')
        data_complete_path = osp.join(args.path, f'{dataset}_{mode}_complete_seq.pkl')
    data_hard = pickle.load(open(data_hard_path, 'rb'))
    data_complete = pickle.load(open(data_complete_path, 'rb'))
    valid_heads_path = osp.join(args.path, 'kbc_data','valid_heads.pickle')
    valid_heads = pickle.load(open(valid_heads_path, 'rb'))
    valid_tails_path = osp.join(args.path, 'kbc_data','valid_tails.pickle')
    valid_tails = pickle.load(open(valid_tails_path, 'rb'))

    candidates = int(args.candidates)
    run(args.model_path, data_hard, data_complete,
        dataset, t_norm=args.t_norm, candidates=candidates,
        scores_normalize=int(args.scores_normalize),
        kg_path=args.path, explain=args.explain, chain_type=args.chain_type, reasoning_mode=args.reasoning_mode, seq=args.seq,
        valid_heads=valid_heads, valid_tails=valid_tails, cov_anchor=args.cov_anchor, cov_var=args.cov_var, cov_target=args.cov_target)
