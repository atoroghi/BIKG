#%%
#%%
import torch
import numpy as np
import os.path as osp
import argparse
import pickle
import json
import tqdm
import sys
from kbc.chain_dataset import ChaineDataset
from kbc.chain_dataset import save_chain_data
from kbc.chain_dataset import Chain
from kbc.datasets import Dataset
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
    parser.add_argument(
    '--hardness', default='hard'
    )
    args = parser.parse_args()
    dataset = osp.basename(args.path)
    mode = args.mode
    hardness = args.hardness
    data_hard_path = osp.join(args.path, f'{dataset}_{mode}_hard.pkl')
    data_hard = pickle.load(open(data_hard_path, 'rb'))

    train = pickle.load(open(osp.join(args.path, f'kbc_data/train.txt.pickle'), 'rb'))
    test = pickle.load(open(osp.join(args.path, f'kbc_data/test.txt.pickle'), 'rb'))
    valid = pickle.load(open(osp.join(args.path, f'kbc_data/valid.txt.pickle'), 'rb'))
    to_skip = pickle.load(open(osp.join(args.path, f'kbc_data/to_skip.pickle'), 'rb'))
    all_data = np.concatenate((train, test, valid), axis=0)
    extended_chain = ChaineDataset(Dataset(osp.join('data',args.dataset,'kbc_data')),1000)
    #extended_chain..set_attr(type1_2chain=chain1_2)
    chain_types = ['1_2', '1_3']
    #chain_types = ['1_2']
    #data_name = str(args.dataset) + '_'+ mode +'_' +'hard_seq'
    data_name = str(args.dataset) + '_'+ mode +'_' +f'{hardness}_seq'

    for chain_type in chain_types:
        if chain_type == '2_2':
            enough = 0; not_enough = 0
            for j, chain in tqdm.tqdm(enumerate(data_hard.type2_2chain)):

                raw_chain = chain.data['raw_chain']

                # split to two parts
                chain1, chain2 = raw_chain[0], raw_chain[1]
                anchor1 , rel1 , targets = chain1
                anchor2, rel2, targets = chain2
                if rel1 % 2 == 0:
                    rel1_inv = rel1 + 1
                else: 
                    rel1_inv = rel1 - 1
                if rel2 % 2 == 0:
                    rel2_inv = rel2 + 1
                else:
                    rel2_inv = rel2 - 1
                modified_anchors = []
                for target in targets:
                    if (target, rel1_inv) not in to_skip['lhs'] or (target, rel2_inv) not in to_skip['lhs']:
                        continue
                    all_anchors_1 = to_skip['lhs'][(target, rel1_inv)]
                    all_anchors_2 = to_skip['lhs'][(target, rel2_inv)]
                    if anchor1 not in all_anchors_1 or len(all_anchors_1)<2:
                        continue
                    if anchor2 not in all_anchors_2 or len(all_anchors_2)<2:
                        continue


                    all_anchors1 = [ x for x in all_anchors_1 if (x, rel1) in to_skip['rhs']]
                    all_anchors2 = [ x for x in all_anchors_2 if (x, rel2) in to_skip['rhs']]
                    other_anchors1 = np.setdiff1d(all_anchors1, anchor1); other_anchors2 = np.setdiff1d(all_anchors2, anchor2)
                    new_anchors1, new_anchors2 = other_anchors1[:2] , other_anchors2[:2]
                    modified_anchors1, modified_anchors2 = list(new_anchors1) + [anchor1] , list(new_anchors2) + [anchor2]
                    modified_anchors = [modified_anchors1, modified_anchors2]

                    # found enough anchors
                    break                    
                if len(modified_anchors)<2:
                    not_enough += 1
                    continue
                targets1, targets2, targets3= set(), set(), set()
                targets_this1, targets_this2 = set(), set()
                for i, anch in enumerate(modified_anchors1):
                    if hardness == 'complete':
                        possible_targets = set(to_skip['rhs'][(anch, rel1)])
                    elif hardness == 'hard':
                        possible_targets = set(test[np.where((test[:,0]==anch) & (test[:,1]==rel1))][:,2])
                    if i == 0:
                        targets_this1 = possible_targets
                    else:
                        targets_this1 = targets_this1.intersection(possible_targets)
                for i, anch in enumerate(modified_anchors2):
                    possible_targets = set(to_skip['rhs'][(anch, rel2)])
                    if i == 0:
                        targets_this2 = possible_targets
                    else:
                        targets_this2 = targets_this2.intersection(possible_targets)
                targets_new = list(targets_this1.intersection(targets_this2))
                # not enough targets available with these anchors
                if len(targets) < 1:
                    not_enough += 1
                    continue
                else:
                    enough += 1
                new_raw_chain = [[modified_anchors1, rel1, targets_new], [modified_anchors2, rel2, targets_new]]
                new_chain = Chain()
                new_chain.data['type'] = '2chain2'
                new_chain.data['raw_chain'] = new_raw_chain
                new_chain.data['anchors'].append(new_raw_chain[0][0])
                new_chain.data['anchors'].append(new_raw_chain[1][0])
                new_chain.data['optimisable'].append(new_raw_chain[0][2])
                extended_chain.type2_2chain.append(new_chain) 

        if chain_type == '1_2':
            not_enough = 0
            enough = 0
            for chain in tqdm.tqdm(data_hard.type1_2chain):
                # get the raw chain first
                raw_chain = chain.data['raw_chain']
                # split to two parts

                chain1, chain2 = raw_chain[0], raw_chain[1]
                _, rel2, targets = chain2 
                # in data_complete, each of the chains can come from all_data (targets are all possible entities)
                # in data_hard, chain2 comes from the test set only and doesn't cover every target in test set
                anchor, rel1, _ = chain1
                if rel1 % 2 == 0:
                    rel1_inv = rel1 + 1
                else: 
                    rel1_inv = rel1 - 1
                if rel2 % 2 == 0:
                    rel2_inv = rel2 + 1
                else:
                    rel2_inv = rel2 - 1
                modified_anchors = []

                for target in targets:
                    # finding what variables are possible to reach from each target to the anchor
                    possible_vars_r = test[np.where((test[:, 2]==target) & (test[:,1]==rel2))[0]][:,0]
                    possible_vars_l = to_skip['rhs'][(anchor, rel1)]
                    possible_vars = np.intersect1d(possible_vars_r, possible_vars_l).astype(int)

                    if len(possible_vars) == 0:
                        continue
                    for possible_var in possible_vars:
                        # there might be a possible var that has no anchors, then look for a new var
                        if (possible_var, rel1_inv) not in to_skip['lhs']:
                            continue
                        all_anchors = to_skip['lhs'][(possible_var, rel1_inv)]
                        # the current anchor must be in all_anchors list. if it isn't, then look for a new var
                        if anchor not in all_anchors:
                            continue
                        other_anchors = np.setdiff1d(all_anchors, anchor)
                        # we must have enough anchors to choose from
                        if other_anchors.shape[0]<2:
                            continue
    
                        new_anchors = other_anchors[:2]
                        modified_anchors = list(new_anchors) + [anchor]

    
                        break
                    # we have enough anchors, no need to explore other targets
                    if len(modified_anchors) == 3:
                        break
                # for this chain, we couldn't find enough anchors, thus we skip it
                if len(modified_anchors) != 3:
                    not_enough += 1
                    continue
                else:
                    enough += 1 
                    # targets are the nodes that are reachable from all the anchors and also in the test set
                    
                    targets1, targets2, targets3 = set(), set(), set()

                    for i, anchor in enumerate(modified_anchors):
                        targets_this = set()
                        possible_vars_backward = to_skip['rhs'][(anchor, rel1)]
                        for possible_var_backward in possible_vars_backward:
                            if (possible_var_backward, rel2) not in to_skip['rhs']:
                                continue
                            if hardness == 'complete':
                                targets_this = targets_this.union(set(to_skip['rhs'][(possible_var_backward, rel2)]))
                            elif hardness == 'hard':
                                targets_this = targets_this.union(set(test[np.where((test[:,0]==possible_var_backward)&(test[:,1]==rel2))][:,2]))
                        if i==0:
                            targets1 = targets_this
                        elif i==1:
                            targets2 = targets_this
                        else:
                            targets3 = targets_this
                    #print("old targets", targets)
                    
                    targets_new = targets1.intersection(targets2).intersection(targets3)
                    targets_old = set(targets)
                    acceptable_targets = list(targets_old.intersection(targets_new))
                    #print("acceptable:", acceptable_targets)
                    #print(raw_chain)
                    # new_raw_chain = [[anch0, rel1, -1], [anch1, rel1, -1], [anch2, rel1, -1] , [-1, rel2, [targets]] ]
                    new_raw_chain = [[modified_anchors[0], raw_chain[0][1], raw_chain[0][2]], [modified_anchors[1], raw_chain[0][1], raw_chain[0][2]], [modified_anchors[2], raw_chain[0][1], raw_chain[0][2]] , [raw_chain[1][0], raw_chain[1][1], acceptable_targets] ]
                    new_chain = Chain()
                    new_chain.data['type'] = '1chain2'
                    new_chain.data['raw_chain'] = new_raw_chain
                    new_chain.data['anchors'].append(new_raw_chain[0][0])
                    new_chain.data['anchors'].append(new_raw_chain[1][0])
                    new_chain.data['anchors'].append(new_raw_chain[2][0])
                    new_chain.data['optimisable'].append(new_raw_chain[0][2])
                    new_chain.data['optimisable'].append(new_raw_chain[3][2])
                    extended_chain.type1_2chain.append(new_chain)
                    #print(extended_chain.type1_2chain[0].data['raw_chain'])
                    #sys.exit()


                    

                            

        # print(f"{enough} chains had enough anchors:")
        # print(f"{not_enough} chains did not have enough anchors:")
        # sys.exit()

        elif chain_type == '1_3':
            not_enough = 0
            enough = 0
            for chain in tqdm.tqdm(data_hard.type1_3chain):
                raw_chain = chain.data['raw_chain']
                chain1, chain2, chain3 = raw_chain[0], raw_chain[1], raw_chain[2]
                _, rel3, targets = chain3
                _, rel2, _ = chain2
                anchor, rel1, _ = chain1
                if rel1 % 2 == 0:
                    rel1_inv = rel1 + 1
                else: 
                    rel1_inv = rel1 - 1
                if rel2 % 2 == 0:
                    rel2_inv = rel2 + 1
                else:
                    rel2_inv = rel2 - 1
                if rel3 % 2 == 0:
                    rel3_inv = rel3 + 1
                else:
                    rel3_inv = rel3 - 1
                modified_anchors = []

                for target in targets:
                    # finding what variables are possible to reach from each target to the anchor
                    possible_vars_2 = test[np.where((test[:, 2]==target) & (test[:,1]==rel3))[0]][:,0]

                    if len(possible_vars_2) == 0:
                        continue
                    for possible_var_2 in possible_vars_2:
                        if (possible_var_2, rel2_inv) not in to_skip['lhs']:
                            continue
                    
                        possible_vars_1_r = to_skip['lhs'][(possible_var_2, rel2_inv)]
                        possible_vars_1_l = to_skip['rhs'][(anchor, rel1)]
                        possible_vars_1 = np.intersect1d(possible_vars_1_r, possible_vars_1_l).astype(int)
                        # the selected variable was not compatible with the anchor
                        if len(possible_vars_1) < 1:
                            continue
                        
                        for possible_var in possible_vars_1:
                            #print(raw_chain)
                            # there might be a possible var that has no anchors, then look for a new var
                            if (possible_var, rel1_inv) not in to_skip['lhs']:
                                continue
                            all_anchors = to_skip['lhs'][(possible_var, rel1_inv)]
                            # the current anchor must be in all_anchors list. if it isn't, then look for a new var
                            if anchor not in all_anchors:
                                continue
                            other_anchors = np.setdiff1d(all_anchors, anchor)
                            # we must have enough anchors to choose from
                            if other_anchors.shape[0]<2:
                                continue
                            new_anchors = other_anchors[:2]
                            modified_anchors = list(new_anchors) + [anchor]
                            break
                        # we have enough anchors, no need to explore other vars
                        if len(modified_anchors) == 3:
                            break
                    # we have enough anchors, no need to explore other targets

                    if len(modified_anchors) == 3:
                        break
                if len(modified_anchors) != 3:
                    not_enough += 1
                    continue
                else:
                    enough += 1
                    # targets are the nodes that are reachable from all the anchors and also in the test set
                    targets1, targets2, targets3 = set(), set(), set()
                    for i, anchor in enumerate(modified_anchors):
                        targets_this = set()
                        possible_vars_backward_1 = to_skip['rhs'][(anchor, rel1)]
                        for possible_var_backward_1 in possible_vars_backward_1:
                            if (possible_var_backward_1, rel2) not in to_skip['lhs']:
                                continue
                            possible_vars_backward_2 = to_skip['rhs'][(possible_var_backward_1, rel2)]
                            for possible_var_backward_2 in possible_vars_backward_2:
                                if (possible_var_backward_2, rel3) not in to_skip['rhs']:
                                    continue
                                if hardness == 'complete':
                                    targets_this = targets_this.union(set(to_skip['rhs'][(possible_var_backward_2, rel3)]))
                                elif hardness == 'hard':
                                    targets_this = targets_this.union(set(test[np.where((test[:,0]==possible_var_backward_2)&(test[:,1]==rel3))][:,2]))
                        if i==0:
                            targets1 = targets_this
                        elif i==1:
                            targets2 = targets_this
                        else:
                            targets3 = targets_this
                    #print("old targets", targets)
                    targets_new = targets1.intersection(targets2).intersection(targets3)
                    targets_old = set(targets)
                    acceptable_targets = list(targets_old.intersection(targets_new))

                    #print(raw_chain)
                    #print(acceptable_targets)
                    #print(modified_anchors)
                    # new_raw_chain = [[anch0, rel1, -1], [anch1, rel1, -1], [anch2, rel1, -1] , [-1, rel2, -1], [-1, rel3, [targets]] ]
                    new_raw_chain = [[modified_anchors[0], raw_chain[0][1], raw_chain[0][2]], [modified_anchors[1], raw_chain[0][1], raw_chain[0][2]], [modified_anchors[2], raw_chain[0][1], raw_chain[0][2]] , [raw_chain[1][0], raw_chain[1][1], raw_chain[1][2]], [raw_chain[2][0], raw_chain[2][1], acceptable_targets] ]
                    new_chain = Chain()
                    new_chain.data['type'] = '1chain3'
                    new_chain.data['raw_chain'] = new_raw_chain
                    new_chain.data['anchors'].append(new_raw_chain[0][0])
                    new_chain.data['optimisable'].append(new_raw_chain[0][2])
                    new_chain.data['optimisable'].append(new_raw_chain[1][2])
                    new_chain.data['optimisable'].append(new_raw_chain[2][2])
                    extended_chain.type1_3chain.append(new_chain)
                    #print(extended_chain.type1_3chain[0].data['raw_chain'])
                    #sys.exit()

                    #print("acceptable:", acceptable_targets)
                    #sys.exit()


                #sys.exit()
        print(f"{enough} chains had enough anchors:")
        print(f"{not_enough} chains did not have enough anchors:")

    save_chain_data(args.path,data_name,extended_chain)
    #sys.exit()