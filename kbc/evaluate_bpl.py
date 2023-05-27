import torch
import numpy as np
import pickle
import os, sys, tqdm

from kbc.chain_dataset_bpl import ChaineDataset
from kbc.chain_dataset_bpl import Chain
from kbc.utils import QuerDAG
from kbc.utils import preload_env
from kbc.utils import QuerDAG
from kbc.utils import preload_env

def evaluate_existential(env, scores, user_likes, non_items_array):
    part1, part2 = env.parts
    #print(part1[:5])
    #print(part2[:5])
    chains, chain_instructions = env.chains, env.chain_instructions
    nb_queries, embedding_size = chains[0][0].shape[0], chains[0][0].shape[1]
    # lists of users and item ents in each query
    users = env.users
    items = env.items
    
    non_items = set(non_items_array)

    #unique_items = set(items)
    #print(users[:5])
    #print(items[:5])
    #print(user_likes[users[0]])

    ranks = np.zeros((nb_queries//5, 5))
    for i, query in tqdm.tqdm(enumerate(scores[:-1])):
        gt_ent = items[i]

        filtered_indices = []
        filtered_ids = user_likes[users[i]]

        filtered_ids = (filtered_ids | non_items) - {gt_ent}
        #filtered_ids = (filtered_ids) - {gt_ent}
        filtered_indices = np.array(list(filtered_ids)).astype(np.int32)

        scores[i, filtered_indices] = -1e4

        gt_rank = (scores[i] > scores[i, gt_ent]).sum().item() + 1
        #print(gt_ent, gt_rank)


        row = i // 5
        col = i % 5
        ranks[row][col] = gt_rank  


    hits_at_1 = np.mean(ranks <= 1, axis=0)
    hits_at_3 = np.mean(ranks <= 3, axis=0)
    hits_at_10 = np.mean(ranks <= 10, axis=0)
    metrics = (hits_at_1, hits_at_3, hits_at_10)
    #print(ranks)
    print(metrics)    

    sys.exit()
    return metrics