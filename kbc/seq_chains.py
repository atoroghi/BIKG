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
import random
from kbc.metrics import evaluation
from kbc.evaluate_bpl import evaluate_existential
from itertools import combinations_with_replacement, combinations

def get_invrel(rel):
    if rel % 2 == 0:
        return int(rel + 1)
    elif rel % 2 == 1:
        return int(rel - 1)


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
    extended_chain = ChaineDataset(Dataset(osp.join('data',args.dataset,'kbc_data')),5000)
    #extended_chain..set_attr(type1_2chain=chain1_2)
    chain_types = ['1_1' ,'1_2', '1_3', '2_2', '2_3', '3_3', '4_3']
    #chain_types = ['1_3']
    #data_name = str(args.dataset) + '_'+ mode +'_' +'hard_seq'
    data_name = str(args.dataset) + '_'+ mode +'_' +f'{hardness}_seq'

    for chain_type in chain_types:
        print("chain_type: ", chain_type)

        if chain_type == '1_2':
            used_targets = []
            used_anchors = []
            if mode == 'test':
                considered_dataset = test
            elif mode == 'valid':
                considered_dataset = valid
            enough = 0
            for j, triple in tqdm.tqdm(enumerate(considered_dataset)):
                if len(extended_chain.type1_2chain) > 3000:
                    for z in range(5):
                        print(extended_chain.type1_2chain[z].data['raw_chain'])
                    print("number of extracted chains for 1_2:", len(extended_chain.type1_2chain))
                    break

                anchor, rel1, var = triple
                if anchor in used_anchors:
                    continue
                
                rel1_inv = get_invrel(rel1)
                all_vars = set(to_skip['rhs'][(anchor, rel1)])
                if var not in all_vars:
                    continue
                all_anchors = set()
                for v in all_vars:
                    for anch in set(to_skip['lhs'][(v, rel1_inv)]):
                        all_anchors.add(anch)
                if len(all_anchors)<3:
                    continue
                all_anchors = list(all_anchors)[:3]

                neighbour_rels = {}


                for v in all_vars:

                    neighbour_rels[v] = list(set(considered_dataset[np.where((considered_dataset[:,0] == v))][:,1]))
                
                rel2 = neighbour_rels[var][0]

                all_targets_hard = set(considered_dataset[np.where((considered_dataset[:,0]==var)& (considered_dataset[:,1]==rel2))][:,2])
                all_targets_complete = set(to_skip['rhs'][(var, rel2)])

                for v in all_vars:
                    all_targets_hard = all_targets_hard.union(set(considered_dataset[np.where((considered_dataset[:,0]==v)& (considered_dataset[:,1]==rel2))][:,2]))
                    if (v, rel2) in to_skip['rhs']:
                        all_targets_complete = all_targets_complete.union(set(to_skip['rhs'][(v, rel2)]))

                if len(all_targets_hard) < 1 or len(all_targets_complete) < 1:
                    continue
                all_targets_hard = list(all_targets_hard)
                all_targets_complete = list(all_targets_complete)
                if hardness == 'hard':
                    # new_raw_chain = [[anch0, rel1, -1], [anch1, rel1, -1], [anch2, rel1, -1] , [-1, rel2, [targets]] ]
                    new_raw_chain = [[all_anchors[0], rel1, -1], [all_anchors[1], rel1, -1], [all_anchors[2], rel1, -1] ,  [-1, rel2, all_targets_hard] ]
                elif hardness == 'complete':
                    new_raw_chain = [[all_anchors[0], rel1, -1], [all_anchors[1], rel1, -1], [all_anchors[2], rel1, -1] ,  [-1, rel2, all_targets_complete] ]
                
                used_anchors.append(anchor)
                new_chain = Chain()
                new_chain.data['type'] = '1chain2'
                new_chain.data['raw_chain'] = new_raw_chain
                new_chain.data['anchors'].append(new_raw_chain[0][0])
                new_chain.data['anchors'].append(new_raw_chain[1][0])
                new_chain.data['anchors'].append(new_raw_chain[2][0])
                new_chain.data['optimisable'].append(new_raw_chain[0][2])
                new_chain.data['optimisable'].append(new_raw_chain[3][2])
                extended_chain.type1_2chain.append(new_chain)


        if chain_type == '1_3':
            used_targets = []
            used_anchors = []
            if mode == 'test':
                considered_dataset = test
            elif mode == 'valid':
                considered_dataset = valid
            enough = 0
            for j, triple in tqdm.tqdm(enumerate(considered_dataset)):
                if len(extended_chain.type1_3chain) > 3000:
                    for z in range(5):
                        print(extended_chain.type1_3chain[z].data['raw_chain'])
                    print("number of extracted chains for 1_3:", len(extended_chain.type1_3chain))
                    break

                anchor, rel1, var = triple
                if anchor in used_anchors:
                    continue
                rel1_inv = get_invrel(rel1)
                all_vars = set(to_skip['rhs'][(anchor, rel1)])
                if var not in all_vars:
                    continue
                all_anchors = set()
                for v in all_vars:
                    for anch in set(to_skip['lhs'][(v, rel1_inv)]):
                        all_anchors.add(anch)
                if len(all_anchors)<3:
                    continue
                all_anchors = list(all_anchors)[:3]
                neighbour_rels = {}
                for v in all_vars:
                    neighbour_rels[v] = list(set(considered_dataset[np.where((considered_dataset[:,0] == v))][:,1]))
                
                rel2 = neighbour_rels[var][0]

                all_vars2 = to_skip['rhs'][(var, rel2)]

                if len(all_vars2) < 1:
                    continue
                
                var2 = all_vars2[0]
                neighbour_rels2 = {}
                for v in all_vars2:
                    neighbour_rels2[v] = list(set(considered_dataset[np.where((considered_dataset[:,0] == v))][:,1]))
                if len(neighbour_rels2[var2]) < 1:
                    continue
                rel3 = neighbour_rels2[var2][0]
                all_targets_hard = set(considered_dataset[np.where((considered_dataset[:,0]==var2)& (considered_dataset[:,1]==rel3))][:,2])
                all_targets_complete = set(to_skip['rhs'][(var2, rel3)])

                for v2 in all_vars2:
                    all_targets_hard = all_targets_hard.union(set(considered_dataset[np.where((considered_dataset[:,0]==v2)& (considered_dataset[:,1]==rel3))][:,2]))
                    if (v2, rel3) in to_skip['rhs']:
                        all_targets_complete = all_targets_complete.union(set(to_skip['rhs'][(v2, rel3)]))

                if len(all_targets_hard) < 1 or len(all_targets_complete) < 1:
                    continue
                all_targets_hard = list(all_targets_hard)
                all_targets_complete = list(all_targets_complete)
                if hardness == 'hard':
                    # new_raw_chain = [[anch0, rel1, -1], [anch1, rel1, -1], [anch2, rel1, -1] , [-1, rel2, -1], [-1, rel3, [targets]] ]
                    new_raw_chain = [[all_anchors[0], rel1, -1], [all_anchors[1], rel1, -1], [all_anchors[2], rel1, -1] , [-1, rel2, -1], [-1, rel3, all_targets_hard] ]
                elif hardness == 'complete':
                    new_raw_chain = [[all_anchors[0], rel1, -1], [all_anchors[1], rel1, -1], [all_anchors[2], rel1, -1] , [-1, rel2, -1], [-1, rel3, all_targets_complete] ]
                used_anchors.append(anchor)
                new_chain = Chain()
                new_chain.data['type'] = '1chain3'
                new_chain.data['raw_chain'] = new_raw_chain
                new_chain.data['anchors'].append(new_raw_chain[0][0])
                new_chain.data['anchors'].append(new_raw_chain[1][0])
                new_chain.data['anchors'].append(new_raw_chain[2][0])
                new_chain.data['optimisable'].append(new_raw_chain[0][2])
                new_chain.data['optimisable'].append(new_raw_chain[3][2])
                new_chain.data['optimisable'].append(new_raw_chain[4][2])
                extended_chain.type1_3chain.append(new_chain)


        if chain_type == '1_1':
            used_targets = []
            if mode == 'test':
                considered_dataset = test
            elif mode == 'valid':
                considered_dataset = valid
            enough = 0
            for j, triple in tqdm.tqdm(enumerate(considered_dataset)):
                if len(extended_chain.type1_1chain) > 1500:
                    # for z in range(5):
                    #     print(extended_chain.type1_1chain[z].data['raw_chain'])
                    print("number of extracted chains for 1_1:", len(extended_chain.type1_1chain)) 
                    break
                anch1, rel1, target = triple
                if target in used_targets:
                    continue
                rel1_inv = get_invrel(rel1)
                # in fact other_anch1 also includes anch1
                other_anchs1 = set(considered_dataset[np.where((considered_dataset[:, 2] == target) & (considered_dataset[:,1]==rel1))][:,0])
                removed_anchs1 = []
                for other_anch1 in other_anchs1:
                    if (other_anch1, rel1) not in to_skip['rhs']:
                        removed_anchs1.append(other_anch1)
                other_anchs1 = other_anchs1.difference(set(removed_anchs1))
            
                anchs1 = [anch1] + list(other_anchs1)
                if len(other_anchs1) < 3:
                    continue
                other_anchs1 = list(other_anchs1)[:3]
                all_targets_hard = list(set(considered_dataset[np.where((considered_dataset[:, 0] == anch1) & (considered_dataset[:,1]==rel1))][:,2]))
                all_targets_complete = list(set(to_skip['rhs'][(anch1,rel1)]))
                anchors1 = other_anchs1
                if hardness == 'hard':
                    new_raw_chain = [[anchors1[0], rel1, all_targets_hard], [anchors1[1], rel1, all_targets_hard], [anchors1[2], rel1, all_targets_hard]]
                elif hardness == 'complete':
                    new_raw_chain = [[anchors1[0], rel1, all_targets_complete], [anchors1[1], rel1, all_targets_complete], [anchors1[2], rel1, all_targets_complete]]
                used_targets.append(target)
                new_chain = Chain()
                new_chain.data['type'] = '1chain1'
                new_chain.data['raw_chain'] = new_raw_chain
                new_chain.data['anchors'].append(new_raw_chain[0][0])
                new_chain.data['anchors'].append(new_raw_chain[1][0])
                new_chain.data['anchors'].append(new_raw_chain[2][0])
                new_chain.data['optimisable'].append(new_raw_chain[0][2])
                extended_chain.type1_1chain.append(new_chain)


        if chain_type == '2_3':
            used_targets = []
            if mode == 'test':
                considered_dataset = test
            elif mode == 'valid':
                considered_dataset = valid
            enough = 0
            for j, triple in tqdm.tqdm(enumerate(considered_dataset)):
                if len(extended_chain.type2_3chain) > 3000:
                    for z in range(5):
                        print(extended_chain.type2_3chain[z].data['raw_chain'])
                    print("number of extracted chains for 2_3:", len(extended_chain.type2_3chain)) 
                    break
                anch1, rel1, target = triple
                if target in used_targets:
                    continue
                rel1_inv = get_invrel(rel1)
                # in fact other_anch1 also includes anch1
                other_anchs1 = set(considered_dataset[np.where((considered_dataset[:, 2] == target) & (considered_dataset[:,1]==rel1))][:,0])
                removed_anchs1 = []
                for other_anch1 in other_anchs1:
                    if (other_anch1, rel1) not in to_skip['rhs']:
                        removed_anchs1.append(other_anch1)
                other_anchs1 = other_anchs1.difference(set(removed_anchs1))
                anchs1 = [anch1] + list(other_anchs1)
                if len(other_anchs1) < 3:
                    continue
                other_anchs1 = list(other_anchs1)[:3]
                all_targets_hard = set(considered_dataset[np.where((considered_dataset[:, 0] == anch1) & (considered_dataset[:,1]==rel1))][:,2])
                all_targets_complete = set(to_skip['rhs'][(anch1,rel1)])
                for other_anch in other_anchs1:
                    all_targets_hard = all_targets_hard.intersection(set(considered_dataset[np.where((considered_dataset[:, 0] == other_anch) & (considered_dataset[:,1]==rel1))][:,2]))
                    all_targets_complete = all_targets_complete.intersection(set(to_skip['rhs'][(other_anch,rel1)]))
                if len(all_targets_hard) < 1 or len(all_targets_complete) < 1:
                    continue
                all_targets_hard = list(all_targets_hard)
                all_targets_complete = list(all_targets_complete)
                for target in all_targets_hard:
                    neighbour_rels = list(set(all_data[np.where((all_data[:,2]==target))][:,1]))
                    if len(neighbour_rels) < 2:
                        continue
                    neighbour_rel_combinations = list(combinations(neighbour_rels, 2))
                    for rel2, rel3 in neighbour_rel_combinations:
                        anchors2 = list(set(all_data[np.where((all_data[:,2]==target) & (all_data[:,1]==rel2))][:,0]))
                        anchors3 = list(set(all_data[np.where((all_data[:,2]==target) & (all_data[:,1]==rel3))][:,0]))
                        if len(anchors2) < 3 or len(anchors3) < 3:
                            continue
                        anchors1 = other_anchs1
                        if hardness == 'hard':
                            new_raw_chain = [[anchors1[0], rel1, all_targets_hard], [anchors2[0], rel2, all_targets_hard], [anchors3[0], rel3, all_targets_hard]
                            , [anchors1[1], rel1, all_targets_hard], [anchors2[1], rel2, all_targets_hard], [anchors3[1], rel3, all_targets_hard],
                            [anchors1[2], rel1, all_targets_hard], [anchors2[2], rel2, all_targets_hard], [anchors3[2], rel3, all_targets_hard]]
                        elif hardness == 'complete':
                            new_raw_chain = [[anchors1[0], rel1, all_targets_complete], [anchors2[0], rel2, all_targets_complete], [anchors3[0], rel3, all_targets_complete]
                            , [anchors1[1], rel1, all_targets_complete], [anchors2[1], rel2, all_targets_complete], [anchors3[1], rel3, all_targets_complete],
                            [anchors1[2], rel1, all_targets_complete], [anchors2[2], rel2, all_targets_complete], [anchors3[2], rel3, all_targets_complete]]
                        used_targets.append(target)
                        new_chain = Chain()
                        new_chain.data['type'] = '2chain3'
                        new_chain.data['raw_chain'] = new_raw_chain
                        new_chain.data['anchors'].append(new_raw_chain[0][0])
                        new_chain.data['anchors'].append(new_raw_chain[1][0])
                        new_chain.data['anchors'].append(new_raw_chain[2][0])
                        new_chain.data['anchors'].append(new_raw_chain[3][0])
                        new_chain.data['anchors'].append(new_raw_chain[4][0])
                        new_chain.data['anchors'].append(new_raw_chain[5][0])
                        new_chain.data['anchors'].append(new_raw_chain[6][0])
                        new_chain.data['anchors'].append(new_raw_chain[7][0])
                        new_chain.data['anchors'].append(new_raw_chain[8][0])
                        new_chain.data['optimisable'].append(new_raw_chain[0][2])
                        extended_chain.type2_3chain.append(new_chain)






        if chain_type == '2_2':
            #print(data_hard.type1_2chain[0].data['raw_chain'])
            # raw chain for original 2_2: [2865, 5, -1], [-1, 63, [517, 5126]]
            used_targets = []
            if mode == 'test':
                considered_dataset = test
            elif mode == 'valid':
                considered_dataset = valid
            enough = 0
            for j, triple in tqdm.tqdm(enumerate(considered_dataset)):
                if len(extended_chain.type2_2chain) > 3000:
                    for z in range(5):
                        print(extended_chain.type2_2chain[z].data['raw_chain'])
                    print("number of extracted chains for 2_2:", len(extended_chain.type2_2chain)) 
                    break
                anch1, rel1, target = triple
                if target in used_targets:
                    continue
                rel1_inv = get_invrel(rel1)
                # in fact other_anch1 also includes anch1
                other_anchs1 = set(considered_dataset[np.where((considered_dataset[:, 2] == target) & (considered_dataset[:,1]==rel1))][:,0])
                removed_anchs1 = []
                for other_anch1 in other_anchs1:
                    if (other_anch1, rel1) not in to_skip['rhs']:
                        removed_anchs1.append(other_anch1)
                other_anchs1 = other_anchs1.difference(set(removed_anchs1))
                anchs1 = [anch1] + list(other_anchs1)
                if len(other_anchs1) < 3:
                    continue
                other_anchs1 = list(other_anchs1)[:3]
                all_targets_hard = set(considered_dataset[np.where((considered_dataset[:, 0] == anch1) & (considered_dataset[:,1]==rel1))][:,2])
                all_targets_complete = set(to_skip['rhs'][(anch1, rel1)])
                for other_anch in other_anchs1:
                    all_targets_hard = all_targets_hard.intersection(set(considered_dataset[np.where((considered_dataset[:, 0] == other_anch) & (considered_dataset[:,1]==rel1))][:,2]))
                    all_targets_complete = all_targets_complete.intersection(set(to_skip['rhs'][(other_anch, rel1)]))
                if len(all_targets_hard) < 1 or len(all_targets_complete) < 1:
                    continue
                all_targets_hard = list(all_targets_hard)
                all_targets_complete = list(all_targets_complete)

                for target in all_targets_hard:
                    neighbour_rels = list(set(all_data[np.where((all_data[:,2]==target))][:,1]))
                    for neighbour_rel in neighbour_rels:
                        rel2 = neighbour_rel
                        anchors2 = list(set(all_data[np.where((all_data[:,2]==target) & (all_data[:,1]==neighbour_rel))][:,0]))
                        if len(anchors2) < 3:
                            continue
                        
                        anchors1 = other_anchs1
                        if hardness == 'hard':
                            new_raw_chain = [ [anchors1[0], rel1, all_targets_hard], [anchors2[0], rel2, all_targets_hard],
                             [anchors1[1], rel1, all_targets_hard], [anchors2[1], rel2, all_targets_hard],
                             [anchors1[2], rel1, all_targets_hard], [anchors2[2], rel2, all_targets_hard]]
                        elif hardness == 'complete':
                            new_raw_chain = [ [anchors1[0], rel1, all_targets_complete], [anchors2[0], rel2, all_targets_complete],
                             [anchors1[1], rel1, all_targets_complete], [anchors2[1], rel2, all_targets_complete],
                             [anchors1[2], rel1, all_targets_complete], [anchors2[2], rel2, all_targets_complete]]
                        used_targets.append(target)
                        new_chain = Chain()
                        new_chain.data['type'] = '2chain2'
                        new_chain.data['raw_chain'] = new_raw_chain
                        new_chain.data['anchors'].append(new_raw_chain[0][0])
                        new_chain.data['anchors'].append(new_raw_chain[1][0])
                        new_chain.data['anchors'].append(new_raw_chain[2][0])
                        new_chain.data['anchors'].append(new_raw_chain[3][0])
                        new_chain.data['anchors'].append(new_raw_chain[4][0])
                        new_chain.data['anchors'].append(new_raw_chain[5][0])
                        new_chain.data['optimisable'].append(new_raw_chain[0][2])
                        extended_chain.type2_2chain.append(new_chain)

                            

        if chain_type == '4_3':
            # raw chain for original 4_3: [[62, 51, -1], [2388, 30, -1], [-1, 381, [9009, 13493, 11703]]]

            used_targets = []
            if mode == 'test':
                considered_dataset = test
            elif mode == 'valid':
                considered_dataset = valid
            enough = 0
            for j, triple in tqdm.tqdm(enumerate(considered_dataset)):
                if len(extended_chain.type4_3chain) > 3000:
                    for z in range(5):
                        print(extended_chain.type4_3chain[z].data['raw_chain'])
                    print("number of extracted chains for 4_3:", len(extended_chain.type4_3chain)) 
                    break
                var3, rel3, target = triple
                if target in used_targets:
                    continue
                rel3_inv = get_invrel(rel3)
                # in fact other_vars3 also includes var3
                other_vars3 = set(considered_dataset[np.where((considered_dataset[:, 2] == target) & (considered_dataset[:,1]==rel3))][:,0])
                removed_vars3 = []
                for other_var3 in other_vars3:
                    if (other_var3, rel3) not in to_skip['rhs']:
                        removed_vars3.append(other_var3)
                other_vars3 = other_vars3.difference(set(removed_vars3))
                vars3 = [var3] + list(other_vars3)
                all_targets_hard = set(considered_dataset[np.where((considered_dataset[:, 0] == var3) & (considered_dataset[:,1]==rel3))][:,2])
                all_targets_complete = set(to_skip['rhs'][(var3, rel3)])
                for other_var3 in other_vars3:
                    all_targets_hard = all_targets_hard.intersection(set(considered_dataset[np.where((considered_dataset[:, 0] == other_var3) & (considered_dataset[:,1]==rel3))][:,2]))
                    all_targets_complete = all_targets_complete.intersection(set(to_skip['rhs'][(other_var3, rel3)]))
                if len(all_targets_hard) < 1 or len(all_targets_complete) < 1:
                    continue
                all_targets_hard = list(all_targets_hard)
                all_targets_complete = list(all_targets_complete)

                # going from each variable to extract two anchors
                for variable in vars3[:1]:
                    neighbour_rels = list(set(all_data[np.where((all_data[:,2]==variable))][:,1]))
                    if len(neighbour_rels) < 2:
                        continue
                    if len(neighbour_rels) > 4:
                        neighbour_rels = random.sample(neighbour_rels, 3)
                    #rel_pairs = list(combinations_with_replacement(neighbour_rels, 2))
                    rel_pairs = list(combinations(neighbour_rels, 2))
                    for rel_pair in rel_pairs:
                        rel1, rel2 = rel_pair
                        anchors1 = list(set(all_data[np.where((all_data[:,2]==variable) & (all_data[:,1]==rel1))][:,0]))[:3]
                        anchors2 = list(set(all_data[np.where((all_data[:,2]==variable) & (all_data[:,1]==rel2))][:,0]))[:3]
                        if len(anchors1)<3 or len(anchors2)<3:
                            continue

                        if hardness == 'hard':
                            new_raw_chain = [ [anchors1[0], rel1, -1 ], [anchors2[0], rel2, -1], [-1, rel3,  all_targets_hard], 
                                            [anchors1[1], rel1, -1 ], [anchors2[1], rel2, -1], [-1, rel3,  all_targets_hard],
                                            [anchors1[2], rel1, -1 ], [anchors2[2], rel2, -1], [-1, rel3,  all_targets_hard]]
                        elif hardness == 'complete':
                            new_raw_chain = [ [anchors1[0], rel1, -1 ], [anchors2[0], rel2, -1], [-1, rel3,  all_targets_complete], 
                                            [anchors1[1], rel1, -1 ], [anchors2[1], rel2, -1], [-1, rel3,  all_targets_complete],
                                            [anchors1[2], rel1, -1 ], [anchors2[2], rel2, -1], [-1, rel3,  all_targets_complete]]
                        used_targets.append(target)
                        new_chain = Chain()
                        new_chain.data['type'] = '4chain3'
                        new_chain.data['raw_chain'] = new_raw_chain
                        new_chain.data['anchors'].append(new_raw_chain[0][0])
                        new_chain.data['anchors'].append(new_raw_chain[1][0])
                        new_chain.data['anchors'].append(new_raw_chain[3][0])
                        new_chain.data['anchors'].append(new_raw_chain[4][0])
                        new_chain.data['anchors'].append(new_raw_chain[6][0])
                        new_chain.data['anchors'].append(new_raw_chain[7][0])
                        new_chain.data['optimisable'].append(new_raw_chain[0][2])
                        new_chain.data['optimisable'].append(new_raw_chain[2][2])
                        new_chain.data['optimisable'].append(new_raw_chain[3][2])
                        new_chain.data['optimisable'].append(new_raw_chain[5][2])
                        extended_chain.type4_3chain.append(new_chain)


        if chain_type == '3_3':
            used_targets = []
            if mode == 'test':
                considered_dataset = test
            elif mode == 'valid':
                considered_dataset = valid

            enough = 0
            for j, triple in tqdm.tqdm(enumerate(considered_dataset)):
                if len(extended_chain.type3_3chain) > 3000:
                    for z in range(5):
                        print(extended_chain.type3_3chain[z].data['raw_chain'])

                    print("number of extracted chains for 3_3:", len(extended_chain.type3_3chain)) 
                    break
            
                anchor3, rel3, target = triple
                if target in used_targets:
                    continue
                rel3_inv = get_invrel(rel3)
                other_anchors3 = set(considered_dataset[np.where((considered_dataset[:, 2] == target) & (considered_dataset[:,1]==rel3))][:,0])
                if len(other_anchors3) < 2:
                    continue
                removed_anchors3 = []
                for other_anchor3 in other_anchors3:
                    if (other_anchor3, rel3) not in to_skip['rhs']:
                        removed_anchors3.append(other_anchor3)
                other_anchors3 = other_anchors3.difference(set(removed_anchors3))
                if len(other_anchors3) < 3:
                    continue

                anchors3 = [anchor3] + list(other_anchors3)

                all_targets_hard = set(considered_dataset[np.where((considered_dataset[:, 0] == anchor3) & (considered_dataset[:,1]==rel3))][:,2])
                all_targets_complete = set(to_skip['rhs'][(anchor3, rel3)])
                for other_anchor3 in other_anchors3:
                    all_targets_hard = all_targets_hard.intersection(set(considered_dataset[np.where((considered_dataset[:, 0] == other_anchor3) & (considered_dataset[:,1]==rel3))][:,2]))
                    all_targets_complete = all_targets_complete.intersection(set(to_skip['rhs'][(other_anchor3, rel3)]))
                if len(all_targets_hard) < 1 or len(all_targets_complete) < 1:
                    continue
                all_targets_hard = list(all_targets_hard)
                all_targets_complete = list(all_targets_complete)
                
                # going from each new target toward anchor1
                for new_target in all_targets_hard:
                    if new_target in used_targets:
                        continue

                    neighbour_rels = set(all_data[np.where((all_data[:,2]==new_target))][:,1])

                    for neighbour_rel in list(neighbour_rels)[:1]:
                        rel2_inv = get_invrel(neighbour_rel)
                        if (new_target, rel2_inv) not in to_skip['lhs']:
                            continue
                        possible_vars = to_skip['lhs'][(new_target, rel2_inv)]

                        for possible_var in possible_vars[:1]:
                            neighbour_rel_vars = set(all_data[np.where((all_data[:,2]==possible_var))][:,1])
                            for neighbour_rel_var in list(neighbour_rel_vars)[:5]:
                                rel1_inv = get_invrel(neighbour_rel_var)
                                if (possible_var, rel1_inv) not in to_skip['lhs']:
                                    continue
                                anchors1 = set(to_skip['lhs'][(possible_var, rel1_inv)])
                                if len(list(anchors1)) < 3:
                                    continue
                                modified_anchors1 = list(anchors1)[:3]
                                modified_anchors3 = list(anchors3)[:3]
                                if hardness == 'hard':
                                    new_raw_chain = [[modified_anchors1[0], get_invrel(rel1_inv), -1], [-1, get_invrel(rel2_inv), all_targets_hard], [modified_anchors3[0], rel3, all_targets_hard], [modified_anchors1[1], get_invrel(rel1_inv), -1], [-1, get_invrel(rel2_inv), all_targets_hard], [modified_anchors3[1], rel3, all_targets_hard], [modified_anchors1[2], get_invrel(rel1_inv), -1], [-1, get_invrel(rel2_inv), all_targets_hard], [modified_anchors3[2], get_invrel(rel3), all_targets_hard]]
                                elif hardness == 'complete':
                                    new_raw_chain = [[modified_anchors1[0], get_invrel(rel1_inv), -1], [-1, get_invrel(rel2_inv), all_targets_complete], [modified_anchors3[0], rel3, all_targets_complete], [modified_anchors1[1], get_invrel(rel1_inv), -1], [-1, get_invrel(rel2_inv), all_targets_complete], [modified_anchors3[1], rel3, all_targets_complete], [modified_anchors1[2], get_invrel(rel1_inv), -1], [-1, get_invrel(rel2_inv), all_targets_complete], [modified_anchors3[2], get_invrel(rel3), all_targets_complete]]

                                used_targets.append(new_target)
                                new_chain = Chain()
                                new_chain.data['type'] = '3chain3'
                                new_chain.data['raw_chain'] = new_raw_chain
                                new_chain.data['anchors'].append(new_raw_chain[0][0])
                                new_chain.data['anchors'].append(new_raw_chain[2][0])
                                new_chain.data['anchors'].append(new_raw_chain[3][0])
                                new_chain.data['anchors'].append(new_raw_chain[5][0])
                                new_chain.data['anchors'].append(new_raw_chain[6][0])
                                new_chain.data['anchors'].append(new_raw_chain[8][0])
                                new_chain.data['optimisable'].append(new_raw_chain[0][2])
                                new_chain.data['optimisable'].append(new_raw_chain[3][2])
                                new_chain.data['optimisable'].append(new_raw_chain[6][2])
                                new_chain.data['optimisable'].append(new_raw_chain[8][2])
                                extended_chain.type3_3chain.append(new_chain)
                                

                            
        # if chain_type == '2_3':
        #     enough = 0; not_enough = 0
        #     for j, chain in tqdm.tqdm(enumerate(data_hard.type2_3chain)):
        #         raw_chain = chain.data['raw_chain']
        #         # split to three parts
        #         chain1, chain2, chain3 = raw_chain[0], raw_chain[1], raw_chain[2]
        #         anchor1 , rel1 , targets = chain1
        #         anchor2, rel2, targets = chain2
        #         anchor3, rel3, targets = chain3
        #         if rel1 % 2 == 0:
        #             rel1_inv = rel1 + 1
        #         else:
        #             rel1_inv = rel1 - 1
        #         if rel2 % 2 == 0:
        #             rel2_inv = rel2 + 1
        #         else:
        #             rel2_inv = rel2 - 1
        #         if rel3 % 2 == 0:
        #             rel3_inv = rel3 + 1
        #         else:
        #             rel3_inv = rel3 - 1
        #         modified_anchors = []
        #         for target in targets:
        #             if (target, rel1_inv) not in to_skip['lhs'] or (target, rel2_inv) not in to_skip['lhs'] or (target, rel3_inv) not in to_skip['lhs']:
        #                 continue
        #             all_anchors_1 = to_skip['lhs'][(target, rel1_inv)]
        #             all_anchors_2 = to_skip['lhs'][(target, rel2_inv)]
        #             all_anchors_3 = to_skip['lhs'][(target, rel3_inv)]
        #             if anchor1 not in all_anchors_1 or len(all_anchors_1)<2:
        #                 continue
        #             if anchor2 not in all_anchors_2 or len(all_anchors_2)<2:
        #                 continue
        #             if anchor3 not in all_anchors_3 or len(all_anchors_3)<2:
        #                 continue

        #             all_anchors1 = [ x for x in all_anchors_1 if (x, rel1) in to_skip['rhs']]
        #             all_anchors2 = [ x for x in all_anchors_2 if (x, rel2) in to_skip['rhs']]
        #             all_anchors3 = [ x for x in all_anchors_3 if (x, rel3) in to_skip['rhs']]
        #             other_anchors1 = np.setdiff1d(all_anchors1, anchor1); other_anchors2 = np.setdiff1d(all_anchors2, anchor2); other_anchors3 = np.setdiff1d(all_anchors3, anchor3)
        #             if other_anchors1.shape[0] < 1 or other_anchors2.shape[0] < 1 or other_anchors3.shape[0] < 1:
        #                 continue
        #             new_anchors1, new_anchors2, new_anchors3 = other_anchors1[:1] , other_anchors2[:1], other_anchors3[:1]
        #             modified_anchors1, modified_anchors2, modified_anchors3 = list(new_anchors1) + [anchor1] , list(new_anchors2) + [anchor2], list(new_anchors3) + [anchor3]
        #             modified_anchors = [modified_anchors1, modified_anchors2, modified_anchors3]
                    

        #             # found enough anchors
        #             break
        #         if len(modified_anchors) < 1:
        #             not_enough += 1
        #             continue
        #         targets_this1, targets_this2, targets_this3 = set(), set(), set()
        #         for i, anch in enumerate(modified_anchors1):
        #             if hardness == 'complete':
        #                 possible_targets = set(to_skip['rhs'][(anch, rel1)])
        #             elif hardness == 'hard':
        #                 if mode == 'test':
        #                     possible_targets = set(test[np.where((test[:,0]==anch) & (test[:,1]==rel1))][:,2])
        #                 elif mode == 'valid':
        #                     possible_targets = set(valid[np.where((valid[:,0]==anch) & (valid[:,1]==rel1))][:,2])
        #             if i == 0:
        #                 targets_this1 = possible_targets
        #             else:
        #                 targets_this1 = targets_this1.intersection(possible_targets)
        #         for i, anch in enumerate(modified_anchors2):
        #             if hardness == 'complete':
        #                 possible_targets = set(to_skip['rhs'][(anch, rel2)])
        #             elif hardness == 'hard':
        #                 if mode == 'test':
        #                     possible_targets = set(test[np.where((test[:,0]==anch) & (test[:,1]==rel2))][:,2])
        #                 elif mode == 'valid':
        #                     possible_targets = set(valid[np.where((valid[:,0]==anch) & (valid[:,1]==rel2))][:,2])
        #             if i == 0:
        #                 targets_this2 = possible_targets
        #             else:
        #                 targets_this2 = targets_this2.intersection(possible_targets)
        #         for i, anch in enumerate(modified_anchors3):
        #             if hardness == 'complete':
        #                 possible_targets = set(to_skip['rhs'][(anch, rel3)])
        #             elif hardness == 'hard':
        #                 if mode == 'test':
        #                     possible_targets = set(test[np.where((test[:,0]==anch) & (test[:,1]==rel3))][:,2])
        #                 elif mode == 'valid':
        #                     possible_targets = set(valid[np.where((valid[:,0]==anch) & (valid[:,1]==rel3))][:,2])
        #             if i == 0:
        #                 targets_this3 = possible_targets
        #             else:
        #                 targets_this3 = targets_this3.intersection(possible_targets)
        #         targets_new = list(targets_this1.intersection(targets_this2).intersection(targets_this3))
        #         if len(targets_new) < 1:
        #             not_enough += 1
        #             continue
        #         else:
        #             enough += 1
                
        #         #new_raw_chain = [[modified_anchors1, rel1, targets_new], [modified_anchors2, rel2, targets_new], [modified_anchors3, rel3, targets_new]]
        #         new_raw_chain = [[modified_anchors1[0], rel1, targets_new], [modified_anchors1[1], rel1, targets_new],[modified_anchors2[0], rel2, targets_new],[modified_anchors2[1], rel2, targets_new], [modified_anchors3[0], rel3, targets_new], [modified_anchors3[1], rel3, targets_new]]
        #         new_chain = Chain()
        #         new_chain.data['type'] = '2chain3'
        #         new_chain.data['raw_chain'] = new_raw_chain
        #         new_chain.data['anchors'].append(new_raw_chain[0][0])
        #         new_chain.data['anchors'].append(new_raw_chain[1][0])
        #         new_chain.data['anchors'].append(new_raw_chain[2][0])
        #         new_chain.data['optimisable'].append(new_raw_chain[0][2])
        #         extended_chain.type2_3chain.append(new_chain) 

        # if chain_type == '2_2':
        #     enough = 0; not_enough = 0
        #     for j, chain in tqdm.tqdm(enumerate(data_hard.type2_2chain)):
        #         raw_chain = chain.data['raw_chain']

        #         # split to two parts
        #         chain1, chain2 = raw_chain[0], raw_chain[1]
        #         anchor1 , rel1 , targets = chain1
        #         anchor2, rel2, targets = chain2
        #         if rel1 % 2 == 0:
        #             rel1_inv = rel1 + 1
        #         else: 
        #             rel1_inv = rel1 - 1
        #         if rel2 % 2 == 0:
        #             rel2_inv = rel2 + 1
        #         else:
        #             rel2_inv = rel2 - 1
        #         modified_anchors = []
        #         #print(raw_chain)
        #         for target_ind, target in enumerate(targets):
        #             if (target, rel1_inv) not in to_skip['lhs'] or (target, rel2_inv) not in to_skip['lhs']:
        #                 continue

        #             all_anchors_1 = to_skip['lhs'][(target, rel1_inv)]
        #             all_anchors_2 = to_skip['lhs'][(target, rel2_inv)]
        #             #
        #             if anchor1 not in all_anchors_1 or len(all_anchors_1)<2:
        #                 continue
        #             if anchor2 not in all_anchors_2 or len(all_anchors_2)<2:
        #                 continue

        #             all_anchors1 = [ x for x in all_anchors_1 if (x, rel1) in to_skip['rhs']]
        #             all_anchors2 = [ x for x in all_anchors_2 if (x, rel2) in to_skip['rhs']]
        #             other_anchors1 = np.setdiff1d(all_anchors1, anchor1); other_anchors2 = np.setdiff1d(all_anchors2, anchor2)
        #             if other_anchors1.shape[0] < 1 or other_anchors2.shape[0] < 1:
        #                 continue
        #             #
        #             new_anchors1, new_anchors2 = other_anchors1[:1] , other_anchors2[:1]
        #             modified_anchors1, modified_anchors2 = list(new_anchors1) + [anchor1] , list(new_anchors2) + [anchor2]
        #             modified_anchors = [modified_anchors1, modified_anchors2]
        #             #
        #             if len(modified_anchors1) < 2 or len(modified_anchors2) < 2:
        #                 if target_ind == len(targets)-1:
        #                     not_enough += 1
        #                 continue
        #             targets_this1, targets_this2 = set(), set()
        #             for i, anch in enumerate(modified_anchors1):
        #                 if hardness == 'complete':
        #                     possible_targets1 = set(to_skip['rhs'][(anch, rel1)])
        #                     if mode == 'test':
        #                         possible_targets_hard = set(test[np.where((test[:,0]==anch) & (test[:,1]==rel1))][:,2])
        #                     elif mode == 'valid':
        #                         possible_targets_hard = set(valid[np.where((valid[:,0]==anch) & (valid[:,1]==rel1))][:,2])
        #                 elif hardness == 'hard':
        #                     if mode == 'test':
        #                         possible_targets1 = set(test[np.where((test[:,0]==anch) & (test[:,1]==rel1))][:,2])
        #                         possible_targets_hard = possible_targets1
        #                     elif mode == 'valid':
        #                         possible_targets1 = set(valid[np.where((valid[:,0]==anch) & (valid[:,1]==rel1))][:,2])
        #                         possible_targets_hard = possible_targets1
        #                 if i == 0:
        #                     targets_this1 = possible_targets1
        #                     targets_this1_hard = possible_targets_hard
        #                 else:
        #                     targets_this1 = targets_this1.intersection(possible_targets1)
        #                     targets_this1_hard = targets_this1_hard.intersection(possible_targets_hard)
        #             for i, anch in enumerate(modified_anchors2):
        #                 if hardness == 'complete':
        #                     possible_targets2 = set(to_skip['rhs'][(anch, rel2)])
        #                     if mode == 'test':
        #                         possible_targets_hard = set(to_skip['rhs'][(anch, rel2)])
        #                         #possible_targets_hard = set(test[np.where((test[:,0]==anch) & (test[:,1]==rel2))][:,2])
        #                     elif mode == 'valid':
        #                         possible_targets_hard = set(to_skip['rhs'][(anch, rel2)])
        #                         #possible_targets_hard = set(valid[np.where((valid[:,0]==anch) & (valid[:,1]==rel2))][:,2])
        #                 elif hardness == 'hard':
        #                     if mode == 'test':
        #                         possible_targets2 = set(to_skip['rhs'][(anch, rel2)])
        #                         #possible_targets2 = set(test[np.where((test[:,0]==anch) & (test[:,1]==rel2))][:,2])
                                
        #                     elif mode == 'valid':
        #                         possible_targets2 = set(to_skip['rhs'][(anch, rel2)])
        #                         #possible_targets2 = set(valid[np.where((valid[:,0]==anch) & (valid[:,1]==rel2))][:,2])
        #                     possible_targets_hard = possible_targets2
        #                 if i == 0:
        #                     targets_this2 = possible_targets2
        #                     targets_this2_hard = possible_targets_hard
        #                 else:
        #                     targets_this2 = targets_this2.intersection(possible_targets2)
        #                     targets_this2_hard = targets_this2_hard.intersection(possible_targets_hard)
        #             targets_new = list(targets_this1.intersection(targets_this2))
        #             targets_new_hard = list(targets_this1_hard.intersection(targets_this2_hard))
        #             # not enough targets available with these anchors
        #             if len(targets_new) < 1 or len(targets_new_hard) < 1:
        #                 continue

        #             else:
        #                 enough += 1
        #                 new_raw_chain = [[modified_anchors1[0], rel1, targets_new], [modified_anchors1[1], rel1, targets_new],[modified_anchors2[0], rel2, targets_new],[modified_anchors2[1], rel2, targets_new]]
        #                 new_chain = Chain()
        #                 new_chain.data['type'] = '2chain2'
        #                 new_chain.data['raw_chain'] = new_raw_chain
        #                 new_chain.data['anchors'].append(new_raw_chain[0][0])
        #                 new_chain.data['anchors'].append(new_raw_chain[1][0])
        #                 new_chain.data['optimisable'].append(new_raw_chain[0][2])
        #                 extended_chain.type2_2chain.append(new_chain) 
        #     print("number of extracted chains:", len(extended_chain.type2_2chain))                        



                #     # found enough anchors
                #     break                    
                # if len(modified_anchors) < 1:
                #     not_enough += 1
                #     continue
                # #targets1, targets2, targets3= set(), set(), set()
                # targets_this1, targets_this2 = set(), set()
                # for i, anch in enumerate(modified_anchors1):
                #     if hardness == 'complete':
                #         possible_targets = set(to_skip['rhs'][(anch, rel1)])
                #     elif hardness == 'hard':
                #         if mode == 'test':
                #             possible_targets = set(test[np.where((test[:,0]==anch) & (test[:,1]==rel1))][:,2])
                #         elif mode == 'valid':
                #             possible_targets = set(valid[np.where((valid[:,0]==anch) & (valid[:,1]==rel1))][:,2])
                #     if i == 0:
                #         targets_this1 = possible_targets
                #     else:
                #         targets_this1 = targets_this1.intersection(possible_targets)
                # for i, anch in enumerate(modified_anchors2):
                #     if hardness == 'complete':
                #         possible_targets = set(to_skip['rhs'][(anch, rel2)])
                #     elif hardness == 'hard':
                #         if mode == 'test':
                #             possible_targets = set(test[np.where((test[:,0]==anch) & (test[:,1]==rel2))][:,2])
                #         elif mode == 'valid':
                #             possible_targets = set(valid[np.where((valid[:,0]==anch) & (valid[:,1]==rel2))][:,2])
                #     if i == 0:
                #         targets_this2 = possible_targets
                #     else:
                #         targets_this2 = targets_this2.intersection(possible_targets)
                # targets_new = list(targets_this1.intersection(targets_this2))
                # # not enough targets available with these anchors
                # if len(targets_new) < 1:
                #     not_enough += 1
                #     continue
                # else:
                #     enough += 1
                # #new_raw_chain = [[modified_anchors1, rel1, targets_new], [modified_anchors2, rel2, targets_new]]
                # new_raw_chain = [[modified_anchors1[0], rel1, targets_new], [modified_anchors1[1], rel1, targets_new],[modified_anchors2[0], rel2, targets_new],[modified_anchors2[1], rel2, targets_new]]
                # new_chain = Chain()
                # new_chain.data['type'] = '2chain2'
                # new_chain.data['raw_chain'] = new_raw_chain
                # new_chain.data['anchors'].append(new_raw_chain[0][0])
                # new_chain.data['anchors'].append(new_raw_chain[1][0])
                # new_chain.data['optimisable'].append(new_raw_chain[0][2])
                # extended_chain.type2_2chain.append(new_chain) 

        # if chain_type == '1_2':
        #     not_enough = 0
        #     enough = 0
        #     for chain in tqdm.tqdm(data_hard.type1_2chain):
        #         # get the raw chain first
        #         raw_chain = chain.data['raw_chain']
        #         # split to two parts

        #         chain1, chain2 = raw_chain[0], raw_chain[1]
        #         _, rel2, targets = chain2 
        #         # in data_complete, each of the chains can come from all_data (targets are all possible entities)
        #         # in data_hard, chain2 comes from the test set only and doesn't cover every target in test set
        #         anchor, rel1, _ = chain1
        #         if rel1 % 2 == 0:
        #             rel1_inv = rel1 + 1
        #         else: 
        #             rel1_inv = rel1 - 1
        #         if rel2 % 2 == 0:
        #             rel2_inv = rel2 + 1
        #         else:
        #             rel2_inv = rel2 - 1
        #         modified_anchors = []

        #         for target in targets:
        #             # finding what variables are possible to reach from each target to the anchor
        #             if mode == 'test':
        #                 possible_vars_r = test[np.where((test[:, 2]==target) & (test[:,1]==rel2))[0]][:,0]
        #             elif mode == 'valid':
        #                 possible_vars_r = valid[np.where((valid[:, 2]==target) & (valid[:,1]==rel2))[0]][:,0]
        #             possible_vars_l = to_skip['rhs'][(anchor, rel1)]
        #             possible_vars = np.intersect1d(possible_vars_r, possible_vars_l).astype(int)

        #             if len(possible_vars) == 0:
        #                 continue
        #             for possible_var in possible_vars:
        #                 # there might be a possible var that has no anchors, then look for a new var
        #                 if (possible_var, rel1_inv) not in to_skip['lhs']:
        #                     continue
        #                 all_anchors = to_skip['lhs'][(possible_var, rel1_inv)]
        #                 # the current anchor must be in all_anchors list. if it isn't, then look for a new var
        #                 if anchor not in all_anchors:
        #                     continue
        #                 other_anchors = np.setdiff1d(all_anchors, anchor)
        #                 # we must have enough anchors to choose from
        #                 if other_anchors.shape[0]<2:
        #                     continue
    
        #                 new_anchors = other_anchors[:2]
        #                 modified_anchors = list(new_anchors) + [anchor]

    
        #                 break
        #             # we have enough anchors, no need to explore other targets
        #             if len(modified_anchors) == 3:
        #                 break
        #         # for this chain, we couldn't find enough anchors, thus we skip it
        #         if len(modified_anchors) != 3:
        #             not_enough += 1
        #             continue
        #         else:
        #             enough += 1 
        #             # targets are the nodes that are reachable from all the anchors and also in the test set
                    
        #             targets1, targets2, targets3 = set(), set(), set()

        #             for i, anchor in enumerate(modified_anchors):
        #                 targets_this = set()
        #                 possible_vars_backward = to_skip['rhs'][(anchor, rel1)]
        #                 for possible_var_backward in possible_vars_backward:
        #                     if (possible_var_backward, rel2) not in to_skip['rhs']:
        #                         continue
        #                     if hardness == 'complete':
        #                         targets_this = targets_this.union(set(to_skip['rhs'][(possible_var_backward, rel2)]))
        #                     elif hardness == 'hard':
        #                         if mode == 'test':
        #                             targets_this = targets_this.union(set(test[np.where((test[:,0]==possible_var_backward)&(test[:,1]==rel2))][:,2]))
        #                         elif mode == 'valid':
        #                             targets_this = targets_this.union(set(valid[np.where((valid[:,0]==possible_var_backward)&(valid[:,1]==rel2))][:,2]))
        #                 if i==0:
        #                     targets1 = targets_this
        #                 elif i==1:
        #                     targets2 = targets_this
        #                 else:
        #                     targets3 = targets_this
        #             #print("old targets", targets)
                    
        #             targets_new = targets1.intersection(targets2).intersection(targets3)
        #             targets_old = set(targets)
        #             acceptable_targets = list(targets_old.intersection(targets_new))
        #             #print("acceptable:", acceptable_targets)
        #             #print(raw_chain)
        #             # new_raw_chain = [[anch0, rel1, -1], [anch1, rel1, -1], [anch2, rel1, -1] , [-1, rel2, [targets]] ]
        #             new_raw_chain = [[modified_anchors[0], raw_chain[0][1], raw_chain[0][2]], [modified_anchors[1], raw_chain[0][1], raw_chain[0][2]], [modified_anchors[2], raw_chain[0][1], raw_chain[0][2]] , [raw_chain[1][0], raw_chain[1][1], acceptable_targets] ]
        #             new_chain = Chain()
        #             new_chain.data['type'] = '1chain2'
        #             new_chain.data['raw_chain'] = new_raw_chain
        #             new_chain.data['anchors'].append(new_raw_chain[0][0])
        #             new_chain.data['anchors'].append(new_raw_chain[1][0])
        #             new_chain.data['anchors'].append(new_raw_chain[2][0])
        #             new_chain.data['optimisable'].append(new_raw_chain[0][2])
        #             new_chain.data['optimisable'].append(new_raw_chain[3][2])
        #             extended_chain.type1_2chain.append(new_chain)
        #             #print(extended_chain.type1_2chain[0].data['raw_chain'])
        #             #sys.exit()


                    

                            

        # print(f"{enough} chains had enough anchors:")
        # print(f"{not_enough} chains did not have enough anchors:")
        # sys.exit()

        # elif chain_type == '1_3':
        #     not_enough = 0
        #     enough = 0
        #     for chain in tqdm.tqdm(data_hard.type1_3chain):
        #         raw_chain = chain.data['raw_chain']
        #         chain1, chain2, chain3 = raw_chain[0], raw_chain[1], raw_chain[2]
        #         _, rel3, targets = chain3
        #         _, rel2, _ = chain2
        #         anchor, rel1, _ = chain1
        #         if rel1 % 2 == 0:
        #             rel1_inv = rel1 + 1
        #         else: 
        #             rel1_inv = rel1 - 1
        #         if rel2 % 2 == 0:
        #             rel2_inv = rel2 + 1
        #         else:
        #             rel2_inv = rel2 - 1
        #         if rel3 % 2 == 0:
        #             rel3_inv = rel3 + 1
        #         else:
        #             rel3_inv = rel3 - 1
        #         modified_anchors = []

        #         for target in targets:
        #             # finding what variables are possible to reach from each target to the anchor
        #             if mode == 'test':
        #                 possible_vars_2 = test[np.where((test[:, 2]==target) & (test[:,1]==rel3))[0]][:,0]
        #             elif mode == 'valid':
        #                 possible_vars_2 = valid[np.where((valid[:, 2]==target) & (valid[:,1]==rel3))[0]][:,0]

        #             if len(possible_vars_2) == 0:
        #                 continue
        #             for possible_var_2 in possible_vars_2:
        #                 if (possible_var_2, rel2_inv) not in to_skip['lhs']:
        #                     continue
                    
        #                 possible_vars_1_r = to_skip['lhs'][(possible_var_2, rel2_inv)]
        #                 possible_vars_1_l = to_skip['rhs'][(anchor, rel1)]
        #                 possible_vars_1 = np.intersect1d(possible_vars_1_r, possible_vars_1_l).astype(int)
        #                 # the selected variable was not compatible with the anchor
        #                 if len(possible_vars_1) < 1:
        #                     continue
                        
        #                 for possible_var in possible_vars_1:
        #                     #print(raw_chain)
        #                     # there might be a possible var that has no anchors, then look for a new var
        #                     if (possible_var, rel1_inv) not in to_skip['lhs']:
        #                         continue
        #                     all_anchors = to_skip['lhs'][(possible_var, rel1_inv)]
        #                     # the current anchor must be in all_anchors list. if it isn't, then look for a new var
        #                     if anchor not in all_anchors:
        #                         continue
        #                     other_anchors = np.setdiff1d(all_anchors, anchor)
        #                     # we must have enough anchors to choose from
        #                     if other_anchors.shape[0]<2:
        #                         continue
        #                     new_anchors = other_anchors[:2]
        #                     modified_anchors = list(new_anchors) + [anchor]
        #                     break
        #                 # we have enough anchors, no need to explore other vars
        #                 if len(modified_anchors) == 3:
        #                     break
        #             # we have enough anchors, no need to explore other targets

        #             if len(modified_anchors) == 3:
        #                 break
        #         if len(modified_anchors) != 3:
        #             not_enough += 1
        #             continue
        #         else:
        #             enough += 1
        #             # targets are the nodes that are reachable from all the anchors and also in the test set
        #             targets1, targets2, targets3 = set(), set(), set()
        #             for i, anchor in enumerate(modified_anchors):
        #                 targets_this = set()
        #                 possible_vars_backward_1 = to_skip['rhs'][(anchor, rel1)]
        #                 for possible_var_backward_1 in possible_vars_backward_1:
        #                     if (possible_var_backward_1, rel2) not in to_skip['lhs']:
        #                         continue
        #                     possible_vars_backward_2 = to_skip['rhs'][(possible_var_backward_1, rel2)]
        #                     for possible_var_backward_2 in possible_vars_backward_2:
        #                         if (possible_var_backward_2, rel3) not in to_skip['rhs']:
        #                             continue
        #                         if hardness == 'complete':
        #                             targets_this = targets_this.union(set(to_skip['rhs'][(possible_var_backward_2, rel3)]))
        #                         elif hardness == 'hard':
        #                             if mode == 'test':
        #                                 targets_this = targets_this.union(set(test[np.where((test[:,0]==possible_var_backward_2)&(test[:,1]==rel3))][:,2]))
        #                             elif mode == 'valid':
        #                                 targets_this = targets_this.union(set(valid[np.where((valid[:,0]==possible_var_backward_2)&(valid[:,1]==rel3))][:,2]))
        #                 if i==0:
        #                     targets1 = targets_this
        #                 elif i==1:
        #                     targets2 = targets_this
        #                 else:
        #                     targets3 = targets_this
        #             #print("old targets", targets)
        #             targets_new = targets1.intersection(targets2).intersection(targets3)
        #             targets_old = set(targets)
        #             acceptable_targets = list(targets_old.intersection(targets_new))

        #             #print(raw_chain)
        #             #print(acceptable_targets)
        #             #print(modified_anchors)
        #             # new_raw_chain = [[anch0, rel1, -1], [anch1, rel1, -1], [anch2, rel1, -1] , [-1, rel2, -1], [-1, rel3, [targets]] ]
        #             new_raw_chain = [[modified_anchors[0], raw_chain[0][1], raw_chain[0][2]], [modified_anchors[1], raw_chain[0][1], raw_chain[0][2]], [modified_anchors[2], raw_chain[0][1], raw_chain[0][2]] , [raw_chain[1][0], raw_chain[1][1], raw_chain[1][2]], [raw_chain[2][0], raw_chain[2][1], acceptable_targets] ]
        #             new_chain = Chain()
        #             new_chain.data['type'] = '1chain3'
        #             new_chain.data['raw_chain'] = new_raw_chain
        #             new_chain.data['anchors'].append(new_raw_chain[0][0])
        #             new_chain.data['optimisable'].append(new_raw_chain[0][2])
        #             new_chain.data['optimisable'].append(new_raw_chain[3][2])
        #             new_chain.data['optimisable'].append(new_raw_chain[4][2])
        #             extended_chain.type1_3chain.append(new_chain)
        #             #print(extended_chain.type1_3chain[0].data['raw_chain'])
        #             #sys.exit()

        #             #print("acceptable:", acceptable_targets)
        #             #sys.exit()


        #         #sys.exit()
        # # print(f"{enough} chains had enough anchors:")
        # # print(f"{not_enough} chains did not have enough anchors:")

    save_chain_data(args.path,data_name,extended_chain)
    #sys.exit()