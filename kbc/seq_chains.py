import torch
import numpy as np
import os.path as osp
import argparse
import pickle
import json
import tqdm
import sys
from kbc.chain_dataset_bpl import ChaineDataset
from kbc.chain_dataset_bpl import Chain

from kbc.utils import QuerDAG
from kbc.utils import preload_env
from kbc.utils import QuerDAG
from kbc.utils import DynKBCSingleton

from kbc.metrics import evaluation
from kbc.evaluate_bpl import evaluate_existential
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
    description="Complex Query Decomposition - Beam"
    )
    parser.add_argument('path', help='Path to directory containing queries')
    parser.add_argument(
    '--dataset',
    help="The pickled Dataset name containing the chains"
    )
    parser.add_argument(
    '--mode', default='test'
    )
    args = parser.parse_args()
    dataset = osp.basename(args.path)
    mode = args.mode

    data_hard_path = osp.join(args.path, f'{dataset}_{mode}_hard.pkl')
    data_hard = pickle.load(open(data_hard_path, 'rb'))

    train = pickle.load(open(osp.join(args.path, f'kbc_data/train.txt.pickle'), 'rb'))
    test = pickle.load(open(osp.join(args.path, f'kbc_data/test.txt.pickle'), 'rb'))
    valid = pickle.load(open(osp.join(args.path, f'kbc_data/valid.txt.pickle'), 'rb'))
    to_skip = pickle.load(open(osp.join(args.path, f'kbc_data/to_skip.pickle'), 'rb'))
    all_data = np.concatenate((train, test, valid), axis=0)
    chain_type = '1_2'

    if chain_type == '1_2':
        not_enough = 0
        enough = 0
        for chain in tqdm.tqdm(data_hard.type1_2chain):
            raw_chain = chain.data['raw_chain']

            chain1, chain2 = raw_chain[0], raw_chain[1]

            targets = chain2[2]
            rel2 = chain2[1]

            rel1 = chain1[1]
            anchor = chain1[0]
            modified_anchors = []
            for target in targets:
                possible_vars_r = test[np.where((test[:, 2]==target) & (test[:,1]==rel2))[0]][:,0]
                possible_vars_l = to_skip['rhs'][(anchor, rel1)]
                possible_vars = np.intersect1d(possible_vars_r, possible_vars_l).astype(int)

                if len(possible_vars) == 0:
                    continue
                for possible_var in possible_vars:
                    if (possible_var, rel1-1) not in to_skip['lhs']:
                        continue
                    all_anchors = to_skip['lhs'][(possible_var, rel1-1)]
                    if anchor not in all_anchors:
                        continue
                    other_anchors = np.setdiff1d(all_anchors, anchor)
                    if other_anchors.shape[0]<2:
                        continue
 
                    new_anchors = other_anchors[:2]
                    modified_anchors = list(new_anchors) + [anchor]
 
                    break

                if len(modified_anchors) == 3:
                    break
            if len(modified_anchors) != 3:
                not_enough += 1
            else:
                enough += 1
                         

        print(f"{enough} chains had enough anchors:")
        print(f"{not_enough} chains did not have enough anchors:")
        sys.exit()


    first_target = data_hard.type1_2chain[0].data['targets'][0]
    first_rel = data_hard.type1_2chain[0].data['raw_chain'][1][1]
    print(first_target, first_rel)
    print((to_skip['rhs'][(2865, 5)]))
    print((to_skip['lhs'][(517, 62)]))
    print(test[np.where((test[:, 2]==first_target) & (test[:,1]==first_rel))[0]])


    sys.exit()