#%%

import os, sys, re, pickle, torch
import numpy as np
from numpy.random import default_rng

import torch
import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm
import time
import sys, os
import pickle
import random
import tqdm
import pandas as pd
from sklearn.model_selection import train_test_split
from collections import defaultdict


def remove_new(test, val, train):
    testval = (test, val)
    axis = (0, 2)

    out = []
    # iterate through test or val triplets
    for tv in testval:    
        # remove both users and items that haven't been seen
        for a in axis:
            train_items = np.unique(train[:, a])
            tv_items = np.unique(tv[:, a])
            rm_tv = [item for item in tv_items if item not in train_items]
            for rm in rm_tv:
                tv = np.delete(tv, np.where(tv[:, a] == rm), axis=0)

        out.append(tv)
    return (out[0], out[1])




def make_critiquing_dicts(rec,kg):
    root_path = os.path.join(os.getcwd() ,'..', 'data', dataset)
    main_path = os.path.join(root_path,'BCIE')
    items = np.unique(rec[:,2])
    popularities={}
    pop_data=np.delete(kg,1,1)
    unique, counts = np.unique(pop_data, return_counts=True)
    # a dictionary containing the popularity of each object
    pop_counts=dict(zip(unique, counts))
    with open(os.path.join(main_path,'pop_counts.pkl'), 'wb') as f:
        pickle.dump(pop_counts, f)
    
    #making dictionary of facts about each item

    items_facts_head={}
    items_facts_tail={}
    for item in items:
      items_facts_head[item]=kg[np.where(kg[:, 0] == item)][:,1:]
      items_facts_tail[item]=kg[np.where(kg[:, 2] == item)][:,:-1]    
    with open(os.path.join(main_path,'items_facts_head.pkl'), 'wb') as f:
        pickle.dump(items_facts_head, f)
    with open(os.path.join(main_path,'items_facts_tail.pkl'), 'wb') as f:
        pickle.dump(items_facts_tail, f)


       #mappings from objects to items
    
    obj2items={}
    for obj in pop_counts.keys():
      if obj not in items:

        objkg = kg[np.where((kg[:, 0] == obj) | (kg[:, 2] == obj))]
        objkg = np.delete(objkg,1,1)
        mapped_items = np.intersect1d(items,objkg)
        obj2items[obj] = mapped_items
      else:
        obj2items[obj] = np.array([obj])
    with open(os.path.join(main_path,'obj2items.pkl'), 'wb') as f:
        pickle.dump(obj2items, f)


# make dictionaries "valid_heads" and "valid_tails" for realations to be used in type checking 
def make_types_dicts(rec,kg):
    data = np.concatenate([rec,kg], axis = 0)
    root_path = os.path.join(os.getcwd() ,'..', 'data', dataset)
    main_path = os.path.join(root_path,'BCIE')
    valid_heads = {}
    valid_tails = {}
    valid_heads_freq = {}
    valid_tails_freq = {}
    all_rels = np.unique(data[:,1])
    for rel in all_rels:
        heads_all = data[np.where(data[:,1]==rel)[0],0]
        tails_all = data[np.where(data[:,1]==rel)[0],2]
        heads, heads_counts = np.unique(heads_all, return_counts = True)
        tails, tails_counts = np.unique(tails_all, return_counts = True)
        valid_heads[rel] = heads
        valid_tails[rel] = tails
        valid_heads_freq[rel] = heads_counts / np.sum(heads_counts)
        valid_tails_freq[rel] = tails_counts / np.sum(tails_counts)

    with open(os.path.join(main_path, 'valid_heads.pkl'), 'wb') as f:
        pickle.dump(valid_heads, f) 
    with open(os.path.join(main_path, 'valid_tails.pkl'), 'wb') as f:
        pickle.dump(valid_tails, f) 
    with open(os.path.join(main_path, 'valid_heads_freq.pkl'), 'wb') as f:
        pickle.dump(valid_heads_freq, f)
    with open(os.path.join(main_path, 'valid_tails_freq.pkl'), 'wb') as f:
        pickle.dump(valid_tails_freq, f)


# user likes for dicts
def user_likes(test, val, train):
    tvt = (test, val, train)

    ul = []
    for data in tvt:
        user_likes = {}
        for i in range(data.shape[0]):
            if data[i,0] not in user_likes:
                user_likes.update({data[i,0]: [data[i,2]]})
            else:
                if data[i,2] not in user_likes[data[i,0]]:
                    user_likes[data[i,0]].append(data[i,2])
        ul.append(user_likes)

    return (ul[0], ul[1], ul[2]) 


def dataset_fold(rec, num_fold, val_ratio=0.005):
    # split dataset according to the split required
    root_path = os.path.join(os.getcwd() ,'..', 'data', dataset)
    main_path = os.path.join(root_path,'BCIE')
    #rec = np.load(os.path.join(main_path, 'rec.npy'))

    rec = np.random.permutation(rec) # shuffle data 
    fold_len = rec.shape[0] // num_fold # get sizes of each fold

    # make and save each of the folds
    rng = default_rng()
    for i in range(num_fold):
        if i < num_fold - 1: test_inds = np.arange(i * fold_len, (i+1) * fold_len)
        else: test_inds = np.arange(i * fold_len, rec.shape[0])
        
        test = rec[test_inds]
        other = np.delete(rec, test_inds, axis=0) # train + valid data

        # get train and valid from random split
        val_len = int(val_ratio * other.shape[0])
        val_inds = rng.choice(other.shape[0], size=val_len, replace=False)
        
        val = other[val_inds]
        train = np.delete(other, val_inds, axis=0)

        # remove users + items from test and val that aren't in train
        (test, val) = remove_new(test, val, train)

        # build user likes maps
        (ul_test, ul_val, ul_train) = user_likes(test, val, train)

        # save data
        print('saving fold: ', i)
        path = os.path.join(main_path, 'fold {}'.format(i))
        os.makedirs(path, exist_ok=True)

        np.save(os.path.join(path, 'train.npy'), train, allow_pickle=True)
        np.save(os.path.join(path, 'test.npy'), test, allow_pickle=True)
        np.save(os.path.join(path, 'val.npy'), val, allow_pickle=True)

        with open(os.path.join(path, 'ul_train.pkl'), 'wb') as f:
            pickle.dump(ul_train, f) 
        with open(os.path.join(path, 'ul_test.pkl'), 'wb') as f:
            pickle.dump(ul_test, f) 
        with open(os.path.join(path, 'ul_val.pkl'), 'wb') as f:
            pickle.dump(ul_val, f) 




#%%
def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
seed_everything(42)


dataset = 'amazon-book'
from pathlib import Path
import pickle
path = os.path.join(os.getcwd() ,'..', 'data', dataset)
root = Path(path)
print(root)
print(os.listdir(root))

# %%
kg_list = []
with open(path + '/kg_final.txt') as f:
    for line in tqdm.tqdm(f):
        line = line.strip('\n').split(' ')
        kg_list.append([int(line[0]), int(line[1]), int(line[2])])

kg = np.array(kg_list).astype(np.uint32)
n_e = len(set(kg[:, 0]) | set(kg[:, 2]))
print("number of entities: ", n_e)
n_r = len(set(kg[:, 1]))
print("number of relations: ", n_r)

kg[:,1] += 1 # offset

# %%
likes_rel = 0
rec_train_dict = {}
with open(path + '/train.txt') as f:
    for line in f:
        line = line.strip('\n').split(' ')
        rec_train_dict[int(line[0])] = set([int(x) for x in line[1:]])

rec_train_list = []
for key in rec_train_dict.keys():
    for val in rec_train_dict[key]:
        rec_train_list.append([key, likes_rel, val])


rec_test_list = []
rec_test_dict = {}
with open(path + '/test.txt') as f:
    for line in f:
        line = line.strip('\n').split(' ')
        try:
            rec_test_dict[int(line[0])] = set([int(x) for x in line[1:]])
        except:
            pass

for key in rec_test_dict.keys():
    for val in rec_test_dict[key]:
        rec_test_list.append([key, likes_rel, val])

rec_train = np.array(rec_train_list).astype(np.uint32)
rec_test = np.array(rec_test_list).astype(np.uint32)
rec = np.concatenate((rec_train, rec_test), axis = 0)

# %%
#offset the users_ids by the number of entities
rec[:, 0] += n_e
dataset_fold(rec, 5, val_ratio=0.005)
make_types_dicts(rec,kg)
make_critiquing_dicts(rec,kg)
root_path = os.path.join(os.getcwd() ,'..', 'data', dataset)
main_path = os.path.join(root_path,'BCIE')
np.save(os.path.join(main_path,'rec.npy'), rec, allow_pickle=True)
np.save(os.path.join(main_path,'kg.npy'), kg, allow_pickle=True)
# %%
