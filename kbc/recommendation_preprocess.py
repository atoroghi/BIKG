#%%
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
    #making sure that all relations are present in the train set
    #while (len(set(kg[:test_start,1])) < num_rels) or (len(set(kg[:test_start, 0]) | set(kg[:test_start, 2]))<num_ents):
    while (len(set(kg[:test_start,1])) < num_rels):
        np.random.shuffle(kg)
    kg_train = kg[:test_start]
    kg_test = kg[test_start:]
    # eliminate rows from kg_test that have entities not present in kg_train
    kg_test = kg_test[np.isin(kg_test[:, 0], kg_train[:, 0])]
    kg_test = kg_test[np.isin(kg_test[:, 2], kg_train[:, 2])]
    return kg_train , kg_test
#%%
dataset = 'AmazonBook'
#dataset = 'yelp2018'
#dataset = 'last-fm'
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
# change the relations number and add reverse relations
#kg_new = np.zeros((2*kg.shape[0], 3)).astype(np.uint32)
#for i, row in enumerate(kg):
#    # Extract values from row1
#    val0, val1, val2 = row
#    # Create new row1 (0->0, 1->2, 2->4)
#    row1 = np.array([val0, 2*val1, val2])
#    # Create new row2 (the reverse relation (0_rev -> 1, 1_rev -> 3, 2_rev -> 5))
#    row2 = np.array([val2, 1+2*val1, val0])
#    # Add row1 and row2 to kg_new
#    kg_new[i*2] = row1
#    kg_new[i*2+1] = row2
## %%
## change the relations number and add reverse relations
#rec_new = np.zeros((2*rec.shape[0], 3)).astype(np.uint32)
#for i, row in enumerate(rec):
#    # Extract values from row1
#    val0, val1, val2 = row
#    # Create new row1 (0->0, 1->2, 2->4)
#    row1 = np.array([val0, 2*val1, val2])
#    # Create new row2 (the reverse relation (0_rev -> 1, 1_rev -> 3, 2_rev -> 5))
#    row2 = np.array([val2, 1+2*val1, val0])
#    # Add row1 and row2 to kg_new
#    rec_new[i*2] = row1
#    rec_new[i*2+1] = row2

#rec_train, rec_testval = split_kg(rec_new, split = 0.2)
#
#rec_test, rec_valid = train_test_split(rec_testval, test_size=0.5)
## split the rec data into train, val and test
#
#kg_train, kg_val = split_kg(kg_new, split = 0.2)


# %%
rec_train, rec_testval = split_kg(rec, split = 0.3)

rec_test, rec_valid = train_test_split(rec_testval, test_size=0.5)
# split the rec data into train, val and test

kg_train, kg_testval = split_kg(kg, split = 0.3)
kg_test, kg_val = train_test_split(kg_testval, test_size=0.5)




# %%

train = np.concatenate((rec_train, kg_train), axis = 0)
valid = np.concatenate((kg_val, rec_valid), axis = 0)
test = np.concatenate((rec_test, kg_test), axis = 0)
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
# load the train, valid and test txt.pickle files
with open(path + '/train.txt.pickle', 'rb') as f:
    train_loaded = pickle.load(f)
with open(path + '/valid.txt.pickle', 'rb') as f:
    valid_loaded = pickle.load(f)
with open(path + '/test.txt.pickle', 'rb') as f:
    test_loaded = pickle.load(f)
# %%
# %%
files = ['train.txt.pickle', 'valid.txt.pickle', 'test.txt.pickle']
entities, relations = set(), set()
for f in files:
    file_path = os.path.join(path, f)
    with open(file_path, 'rb') as f:
        to_read = pickle.load(f)
        for line in to_read:
            #lhs, rel, rhs = str(line[0]), str(line[1]), str(line[2])
            lhs, rel, rhs = (line[0]), (line[1]), (line[2])
            entities.add(lhs)
            entities.add(rhs)
            relations.add(rel)
            #relations.add(rel+'_reverse')
entities_to_id = {x: i for (i, x) in enumerate(sorted(entities))}
relations_to_id = {x: i for (i, x) in enumerate(sorted(relations))}


n_relations = len(relations_to_id)
n_entities = len(entities_to_id)
print(f'{n_entities} entities and {n_relations} relations')

# %%
for (dic, f) in zip([entities_to_id, relations_to_id], ['ent_id', 'rel_id']):
    pickle.dump(dic, open(os.path.join(path, f'{f}.pickle'), 'wb'))


# %%
to_skip = {'lhs': defaultdict(set), 'rhs': defaultdict(set)}

for file in files:
    file_path = os.path.join(path, file)
    with open(file_path, 'rb') as f:
        to_read = pickle.load(f)

        examples = []
        for line in to_read:
            #lhs, rel, rhs = str(line[0]), str(line[1]), str(line[2])
            lhs, rel, rhs = (line[0]), (line[1]), (line[2])
            lhs_id = entities_to_id[lhs]
            rhs_id = entities_to_id[rhs]
            rel_id = relations_to_id[rel]
            #inv_rel_id = relations_to_id[rel + '_reverse']
            examples.append([lhs_id, rel_id, rhs_id])
            to_skip['rhs'][(lhs_id, rel_id)].add(rhs_id)
            #to_skip['lhs'][(rhs_id, inv_rel_id)].add(lhs_id)
            to_skip['lhs'][(rhs_id, rel_id)].add(lhs_id)
            # Add inverse relations for training
            if file == 'train.txt.pickle':
            #    examples.append([rhs_id, inv_rel_id, lhs_id])
                examples.append([rhs_id, rel_id, lhs_id])
            #    to_skip['rhs'][(rhs_id, inv_rel_id)].add(lhs_id)
                to_skip['rhs'][(rhs_id, rel_id)].add(lhs_id)
                to_skip['lhs'][(lhs_id, rel_id)].add(rhs_id)
    out = open(os.path.join(path,'new'+ file), 'wb')
    pickle.dump(np.array(examples).astype('uint64'), out)
    out.close()

to_skip_final = {'lhs': {}, 'rhs': {}}
for kk, skip in to_skip.items():
    for k, v in skip.items():
        to_skip_final[kk][k] = sorted(list(v))
out = open(os.path.join(path, 'new_to_skip.pickle'), 'wb')
pickle.dump(to_skip_final, out)
out.close()










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








# %%
# After running till line 106 (for training on the BCIE KG trainer)

