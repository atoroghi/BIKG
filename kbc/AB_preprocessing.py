#%%  
import os, pickle, sys, time
import torch
import numpy as np
import random
import tqdm
import pandas as pd
from sklearn.model_selection import train_test_split
from collections import defaultdict
import re
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
    np.random.shuffle(kg)
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

def add_inverse(rec, kg):
    # make current kg rels to even numbers
    #inv_kg = kg[:,[2,1,0]]
    kg[:,1] = kg[:,1]*2
    #inv_kg[:,1] = inv_kg[:,1]*2 + 1
    #new_kg = np.concatenate((kg, inv_kg), axis=0)
    new_kg = kg
    # make current rec rels to even numbers
    new_likes_rel = np.max(new_kg[:,1]) + 2
    rec[:,1] = new_likes_rel
    #inv_rec = rec[:,[2,1,0]]
    #inv_rec[:,1] = inv_rec[:,1] + 1
    #new_rec = np.concatenate((rec, inv_rec), axis=0)
    new_rec = rec
    return new_rec, new_kg
    
#%%   
#dataset = 'Movielens'
#dataset = 'LastFM'
dataset = 'AmazonBook'
from pathlib import Path
import pickle
path = os.path.join(os.getcwd() ,'..', 'data', dataset)
#path = os.path.join(os.getcwd() , 'data', dataset)
root = Path(path)
print(root)
print(os.listdir(root))    
kg_path = os.path.join(root, 'kg/train.dat')
rec_path = os.path.join(root,'rs/ratings.txt')
kg = np.genfromtxt(kg_path, delimiter='\t', dtype=np.int32)
rec = np.genfromtxt(rec_path, delimiter='\t', dtype=None)
#%%
n_e = len(set(kg[:, 0]) | set(kg[:, 2]))
print("number of entities: ", n_e)
n_r = len(set(kg[:, 1]))
print("number of relations: ", n_r)

rec_users = []
rec_items = []
for i in range(rec.shape[0]):
    if int(rec[i][2]) >= 4:
        rec_users.append(re.search('(?<=\')(.*?)(?=\')', str(rec[i][0])).group(0))
        rec_items.append(re.sub("^0+", "", re.search('(?<=\')(.*?)(?=\')', str(rec[i][1])).group(0)))

#%%
TOTAL_FB_IDS = np.max(kg) # total number of default kg pairs (# rel << # entities)
# paths for converting data

#%%

#rec_items = rec_items[:10000]

#%%
item2kg_path =  os.path.join(root,'rs/i2kg_map.tsv')
emap_path = os.path.join(root,'kg/e_map.dat')
# maps movie lense id's to free base html links
ml2fb_map = {}
with open(item2kg_path) as f:
    for line in f:
        ml_id = re.search('(.+?)\t', line)
        fb_http = re.search('\t(.+?)\n', line)
        ml2fb_map.update({re.sub("^0+", "", ml_id.group(1)) : fb_http.group(1)})

#%%

# maps free base html links to free base id's (final format)
id2html_map = {}
fb2id_map = {}
with open(emap_path) as f:
    for kg_id, line in enumerate(f):
        fb_http = re.search('\t(.+?)\n', line)
        fb2id_map.update({fb_http.group(1) : kg_id})
        id2html_map.update({kg_id : fb_http.group(1)})
rec_users_kept = []
rec_items_converted = []
#%%
# convert movielens id's to freebase id's
    # convert movielens id's to freebase id's
i = 0
j = 0
while True:
    if i == len(rec_items):
        break
    if rec_items[i] in ml2fb_map: 
        # get correct freebase id from data
        fb_http = ml2fb_map[rec_items[i]]
        fb_id = fb2id_map[fb_http]
        rec_items_converted.append(fb_id)
        rec_users_kept.append(rec_users[i])
    i += 1

    j += 1
    print("1",j,i)
#%%
i = 0
j = 0
max_loop = len(rec_items_converted)
indices_to_remove = []

for item in rec_items_converted:
    if item not in kg:
        indices_to_remove.append(i)

#%%
rec_items_filtered = np.delete(rec_items_converted, indices_to_remove)
rec_users_filtered = np.delete(rec_users_kept, indices_to_remove)

#%%
umap_path = os.path.join(root,'rs/u_map.dat')
userid2fbid_map = {}
new_ids = 0
with open(umap_path) as f:
    for line in f:
        ml_id = re.search('\t(.+?)\n', line)
        #if int(ml_id.group(1)) in rec[:,0]:
        if ml_id.group(1) in rec_users_filtered:
            new_ids += 1
            #userid2fbid_map.update({int(ml_id.group(1)) : TOTAL_FB_IDS + new_ids})
            userid2fbid_map.update({ml_id.group(1) : TOTAL_FB_IDS + new_ids})
# convert movielens user id's into freebase id's
for i in range((rec_users_filtered.shape[0])):
    rec_users_filtered[i] = userid2fbid_map[rec_users_filtered[i]]
NEW_USER_IDS = new_ids

#%%
likes_rel = n_r
rec = np.column_stack((rec_users_filtered, np.full(len(rec_users_filtered), n_r), rec_items_filtered))
#%%
np.save(os.path.join(root,'rs/rec_processed.npy'), rec, allow_pickle=True)
#%%
#rec = np.load(os.path.join(root,'rs/rec_raw.npy'), allow_pickle=True)
#%%
rec, kg = add_inverse(rec, kg)

# %%
rec_train, rec_testval = split_kg(rec, split = 0.3)

rec_test, rec_valid = train_test_split(rec_testval, test_size=0.5)
# split the rec data into train, val and test

kg_train, kg_testval = split_kg(kg, split = 0.3)
kg_test, kg_val = train_test_split(kg_testval, test_size=0.5)

# %%
train = np.concatenate((rec_train, kg_train), axis = 0).astype(np.int32)
valid = np.concatenate((kg_val, rec_valid), axis = 0).astype(np.int32)
test = np.concatenate((rec_test, kg_test), axis = 0).astype(np.int32)

# delete rows of users in the test set that are not in the train set
test = test[np.isin(test[:, 0], train[:, 0])]

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
            relations.add(rel+1)
            #relations.add(rel+'_reverse')
entities_to_id = {x: i for (i, x) in enumerate(sorted(entities))}
relations_to_id = {x: i for (i, x) in enumerate(sorted(relations))}
id_to_entities = {i:x for (i, x) in enumerate(sorted(entities))}
id_to_relations = {i:x for (i, x) in enumerate(sorted(relations))}


n_relations = len(relations_to_id)
n_entities = len(entities_to_id)
print(f'{n_entities} entities and {n_relations} relations')

# %%
for (dic, f) in zip([entities_to_id, relations_to_id, id_to_entities, id_to_relations], ['ent_id', 'rel_id', 'id_ent', 'id_rel']):
    pickle.dump(dic, open(os.path.join(path, f'{f}.pickle'), 'wb'))
# %%
to_skip = {'lhs': defaultdict(set), 'rhs': defaultdict(set)}

for file in files:
    file_path = os.path.join(path, file)
    with open(file_path, 'rb') as f:
        to_read = pickle.load(f)
        examples = []
        for line in to_read:
            lhs, rel, rhs = (line[0]), (line[1]), (line[2])
            lhs_id = entities_to_id[lhs]
            rhs_id = entities_to_id[rhs]
            rel_id = relations_to_id[rel]
            examples.append([lhs_id, rel_id, rhs_id])
            to_skip['rhs'][(lhs_id, rel_id)].add(rhs_id)
            to_skip['lhs'][(rhs_id, rel_id+1)].add(lhs_id)

            if file == 'train.txt.pickle':
                examples.append([rhs_id, rel_id+1, lhs_id])
                to_skip['rhs'][(rhs_id, rel_id+1)].add(lhs_id)
                to_skip['lhs'][(lhs_id, rel_id)].add(rhs_id)
            


        ##examples = []
        #for line in to_read:
        #    #lhs, rel, rhs = str(line[0]), str(line[1]), str(line[2])
        #    lhs, rel, rhs = (line[0]), (line[1]), (line[2])
        #    lhs_id = entities_to_id[lhs]
        #    rhs_id = entities_to_id[rhs]
        #    rel_id = relations_to_id[rel]
        #    #inv_rel_id = relations_to_id[rel + '_reverse']
        #    #inv_rel_id = relations_to_id[rel*2+1]
        #    examples.append([lhs_id, rel_id, rhs_id])
        #    to_skip['rhs'][(lhs_id, rel_id)].add(rhs_id)
        #    to_skip['lhs'][(rhs_id, inv_rel_id)].add(lhs_id)
        #    #to_skip['lhs'][(rhs_id, rel_id)].add(lhs_id)
        #    # Add inverse relations for training
        #    if file == 'train.txt.pickle':
        #        examples.append([rhs_id, inv_rel_id, lhs_id])
        #    #    examples.append([rhs_id, rel_id, lhs_id])
        #        to_skip['rhs'][(rhs_id, inv_rel_id)].add(lhs_id)
        #    #    to_skip['rhs'][(rhs_id, rel_id)].add(lhs_id)
        #        to_skip['lhs'][(lhs_id, rel_id)].add(rhs_id)
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
