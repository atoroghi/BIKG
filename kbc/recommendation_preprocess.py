#%%
import os, pickle, sys, time
import torch
import numpy as np
import random
import tqdm
import pandas as pd
from sklearn.model_selection import train_test_split
from collections import defaultdict

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


def split_kg(kg, split = 0.2):
    num_rels = len(set(kg[:, 1]))
    num_ents = len(set(kg[:, 0]) | set(kg[:, 2]))
    test_start = int((1-split)*kg.shape[0])
    #making sure that all entities and relations are present in the train set
    while (len(set(kg[:test_start,1])) < num_rels) or (len(set(kg[:test_start, 0]) | set(kg[:test_start, 2]))<num_ents):
        np.random.shuffle(kg)
    kg_train = kg[:test_start]
    kg_test = kg[test_start:]
    return kg_train , kg_test
#%%
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
# count number of unique entities and relations
# %%
likes_rel = n_r
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

#offset the users_ids by the number of entities
rec[:, 0] += n_e
# %%
rec_train, rec_testval = split_kg(rec, split = 0.2)
# %%

rec_test, rec_valid = train_test_split(rec_testval, test_size=0.5)
# split the rec data into train, val and test
# %%
kg_train, kg_val = split_kg(kg, split = 0.2)






# %%

train = np.concatenate((rec_train, kg_train), axis = 0)
valid = np.concatenate((kg_val, rec_valid), axis = 0)
test = rec_test
# %%
# delete rows of users in the test set that are not in the train set
test = test[np.isin(test[:, 0], train[:, 0])]
# %%
valid = valid[np.isin(valid[:, 0], train[:, 0])]
# %%
# write the train, valid and test data to txt.pickle files
with open(path + '/train.txt.pickle', 'wb') as f:
    pickle.dump(train, f)
with open(path + '/valid.txt.pickle', 'wb') as f:
    pickle.dump(valid, f)
with open(path + '/test.txt.pickle', 'wb') as f:
    pickle.dump(test, f)

# %%
# forming the type checking dictionaries
all_data = np.concatenate((train, test, valid), axis = 0)
valid_heads = {}
valid_tails = {}
# a dictionary of valid heads and tails for each relation from all_data
for i in range(all_data.shape[0]):
    rel = all_data[i, 1]
    head = all_data[i, 0]
    tail = all_data[i, 2]
    if rel not in valid_heads:
        valid_heads[rel] = set()
    valid_heads[rel].add(head)
    if rel not in valid_tails:
        valid_tails[rel] = set()
    valid_tails[rel].add(tail)

for i in valid_heads.keys():
    valid_heads[i] = list(valid_heads[i])
for i in valid_tails.keys():
    valid_tails[i] = list(valid_tails[i])


out_file = open(path + '/valid_heads.pickle', 'wb')
pickle.dump(valid_heads, out_file)
out_file.close()

out_file = open(path + '/valid_tails.pickle', 'wb')
pickle.dump(valid_tails, out_file)
out_file.close()
# %%
# making the rel_id, ent_id, and to_skip dictionaries
all_data = np.concatenate((train, test, valid), axis = 0)
rel_id = {}
ent_id = {}
to_skip = {}

for line in all_data:
    if line[1] not in rel_id:
        rel_id[line[1]] = line[1]
    if line[0] not in ent_id:
        ent_id[line[0]] = line[0]
    if line[2] not in ent_id:
        ent_id[line[2]] = line[2]

#with open(path + '/entity_list.txt') as f:
#    for line in f:
#        #skip the first line
#        if line[0] != 'm':
#            continue
#        line = line.strip('\n').split(' ')
#        ent_id[line[0]] = line[0]
#
#with open(path + '/relation_list.txt') as f:
#    for line in f:
#        #skip the first line
#        if line[0] != 'h':
#            continue
#        line = line.strip('\n').split(' ')
#        rel_id[line[0]] = line[0]


# %%
out_file = open(path + '/ent_id.pickle', 'wb')
pickle.dump(ent_id, out_file)
out_file.close()

out_file = open(path + '/rel_id.pickle', 'wb')
pickle.dump(rel_id, out_file)
out_file.close()
# %%
# map train/test/valid with the ids
to_skip = {'lhs': defaultdict(set), 'rhs': defaultdict(set)}
files = [train, valid, test]

for f in files:
    for line in f:
        lhs, rel, rhs = line[0], line[1], line[2]
        to_skip['rhs'][(lhs, rel)].add(rhs)
        to_skip['lhs'][(rhs, rel)].add(lhs)

to_skip_final = {'lhs': {}, 'rhs': {}}
for kk, skip in to_skip.items():
    for k, v in skip.items():
        to_skip_final[kk][k] = sorted(list(v))
out = open(os.path.join(path , 'to_skip.pickle'), 'wb')
pickle.dump(to_skip_final, out)
out.close()
# %%
