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
    
    # TODO: modify this line for one hop KGs
    while (len(set(kg[:test_start,1])) < num_rels-5):
        np.random.shuffle(kg)
    kg_train = kg[:test_start]
    kg_test = kg[test_start:]
    # eliminate rows from kg_test that have entities not present in kg_train
    kg_test = kg_test[np.isin(kg_test[:, 0], kg_train[:, 0])]
    kg_test = kg_test[np.isin(kg_test[:, 2], kg_train[:, 2])]
    kg_test = kg_test[np.isin(kg_test[:, 1], kg_train[:, 1])]
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
#dataset = 'Movielens (no_rev)'
#dataset = 'LastFM'
#dataset = 'AmazonBook'
#dataset = 'Movielens_twohop_nouser'
dataset = 'LastFM_twohop'
from pathlib import Path
import pickle

path = os.path.join(os.getcwd() ,'..', 'data', dataset)
if not os.path.exists(path):
    path = os.path.join(os.getcwd() , 'data', dataset)

root = Path(path)
print(root)
print(os.listdir(root))    
kg_path = os.path.join(root, 'kg/train.dat')
kg_path_test = os.path.join(root, 'kg/test.dat')
kg_path_valid = os.path.join(root, 'kg/valid.dat')
rec_path = os.path.join(root,'rs/ratings.txt')
kg = np.genfromtxt(kg_path, delimiter='\t', dtype=np.int32)


# in case of LFM, there are entities in test and valid that are not in train
if dataset == 'LastFM' or dataset=='LastFM_twohop':
    kg_test = np.genfromtxt(kg_path_test, delimiter='\t', dtype=np.int32)
    kg_valid = np.genfromtxt(kg_path_valid, delimiter='\t', dtype=np.int32)
    kg = np.concatenate((kg, kg_test, kg_valid), axis=0)

rec = np.genfromtxt(rec_path, delimiter='\t', dtype=np.int32)

#%%
# reduce the no of entities in the too large kg

column3_counts = np.bincount(kg[:, 2])
#values_to_delete = np.where((column3_counts < 3))[0] #for Movielens_twohop
values_to_delete = np.where((column3_counts < 3))[0] #for LastFM_twohop
kg = kg[ ~np.isin(kg[:, 2], values_to_delete)]

#%%
n_e = len(set(kg[:, 0]) | set(kg[:, 2]))
print("number of entities: ", n_e)
n_r = len(set(kg[:, 1]))
print("number of relations: ", n_r)

rec = rec[:,:3] # remove time col.
rec[:,2] = rec[:,2] >= 4 # binary ratings, 0 if [0, 4), 1 if [4, 5] 
rec = rec[rec[:,2] == 1] # select only positive ratings
#rec[:,2] = n_r # set redundant col to the last relationship
rec[:,2] = np.max(kg[:,1])+1 # set redundant col to the last relationship

rec = rec[:, [0,2,1]]
#%%
TOTAL_FB_IDS = np.max(kg) # total number of default kg pairs (# rel << # entities)
# paths for converting data

#%%

#rec = rec[:10000]

#%%
item2kg_path =  os.path.join(root,'rs/i2kg_map.tsv')
emap_path = os.path.join(root,'kg/e_map.dat')
# maps movie lense id's to free base html links
ml2fb_map = {}
with open(item2kg_path) as f:
    for line in f:
        ml_id = re.search('(.+?)\t', line)
        fb_http = re.search('\t(.+?)\n', line)
        if dataset == 'Movielens' or dataset == 'LastFM' or dataset == 'Movielens_twohop' or dataset=='LastFM_twohop':
            ml2fb_map.update({int(ml_id.group(1)) : fb_http.group(1)})

#%%

# maps free base html links to free base id's (final format)
id2html_map = {}
fb2id_map = {}
with open(emap_path) as f:
    for kg_id, line in enumerate(f):
        fb_http = re.search('\t(.+?)\n', line)
        try:
            fb2id_map.update({fb_http.group(1) : kg_id})
        except:
            print(f'{kg_id} not in kg')
        id2html_map.update({kg_id : fb_http.group(1)})

#%%
# convert movielens id's to freebase id's
i = 0
j = 0

while True:
    if i == rec.shape[0]:
        break
    if rec[i,2] in ml2fb_map: 
        #print(rec[i,2])
        
        # get correct freebase id from data
        fb_http = ml2fb_map[rec[i,2]]
        #print(f'{fb_http}')
        fb_id = fb2id_map[fb_http]
        #print(f'{fb_id}')
        rec[i,2] = fb_id
        i += 1
    # remove from rec (only use movies that are in kg)
    else:
        rec = np.delete(rec, i, axis=0)
    j += 1
    if j%100000 == 0:
        print("1", j)
#%%
i=0
j = 0
while True:
    if i == rec.shape[0]:
        break
    if rec[i,2] not in kg:
        rec = np.delete(rec, i, axis=0)
    i += 1
    j += 1
    if j%1000000 == 0:
        print("2", j)




#%%
umap_path = os.path.join(root,'rs/u_map.dat')
userid2fbid_map = {}
new_ids = 0
with open(umap_path) as f:
    for line in f:
        ml_id = re.search('\t(.+?)\n', line)
        if int(ml_id.group(1)) in rec[:,0]:
        #if ml_id.group(1) in rec[:,0]:
            new_ids += 1
            userid2fbid_map.update({int(ml_id.group(1)) : TOTAL_FB_IDS + new_ids})
            #userid2fbid_map.update({ml_id.group(1) : TOTAL_FB_IDS + new_ids})
# convert movielens user id's into freebase id's
for i in range(rec.shape[0]):
    rec[i,0] = userid2fbid_map[rec[i,0]]
NEW_USER_IDS = new_ids
#%%
np.save(os.path.join(root,'rs/rec_processed.npy'), rec, allow_pickle=True)
#%%
rec = np.load(os.path.join(root,'rs/rec_processed.npy'), allow_pickle=True)
#%%
# we're not inversing anymore. It's incorrect with SimplE
#rec, kg = add_inverse(rec, kg)

# %%
rec_train, rec_testval = split_kg(rec, split = 0.5)

rec_test, rec_valid = train_test_split(rec_testval, test_size=0.5)
# split the rec data into train, val and test
# %%
kg_train, kg_testval = split_kg(kg, split = 0.5)
kg_test, kg_val = train_test_split(kg_testval, test_size=0.5)

# %%

# %%
train = kg_train
valid = kg_val
test = kg_test

# %%
# at this point, train, valid and test are not all in order entities (we have to compensate later)
train = np.concatenate((rec_train, kg_train), axis = 0)
valid = np.concatenate((kg_val, rec_valid), axis = 0)
test = np.concatenate((rec_test, kg_test), axis = 0)
# %%
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
            #relations.add(rel+1)
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
# compensation for enities not being in order is taken place here

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
            # for the inverse case
            #examples.append([rhs_id, rel_id+1, lhs_id])
            to_skip['rhs'][(lhs_id, rel_id)].add(rhs_id)
            # for the inverse case
            #to_skip['lhs'][(rhs_id, rel_id+1)].add(lhs_id)

            to_skip['lhs'][(rhs_id, rel_id)].add(lhs_id)
            
            #to_skip['rhs'][(rhs_id, rel_id+1)].add(lhs_id)
            #examples.append([lhs_id, rel_id, rhs_id])
            #to_skip['rhs'][(lhs_id, rel_id)].add(rhs_id)
            #to_skip['lhs'][(rhs_id, rel_id+1)].add(lhs_id)

            #if file == 'train.txt.pickle':
            #    examples.append([rhs_id, rel_id+1, lhs_id])
            #    to_skip['rhs'][(rhs_id, rel_id+1)].add(lhs_id)
            #    to_skip['lhs'][(lhs_id, rel_id)].add(rhs_id)
            


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
# remember to replace "train.txt.pickle" with "new_train.txt.pickle" in the folder(same for valid and test)
files = ['train.txt.pickle', 'valid.txt.pickle', 'test.txt.pickle']
for file in files:
    #file_name = e.g., new_train
    file_name = file.split('.')[0]
    with open(os.path.join(path, file), 'rb') as f:
        array = pickle.load(f)
        np.savetxt(f'{file_name}.txt', array, delimiter='\t', fmt='%d')

# %%
# forming the type checking dictionaries
with open(os.path.join(path, 'train.txt.pickle'), 'rb') as f:
    train = pickle.load(f)
with open(os.path.join(path, 'valid.txt.pickle'), 'rb') as f:
    valid = pickle.load(f)
with open(os.path.join(path, 'test.txt.pickle'), 'rb') as f:
    test = pickle.load(f)

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


# TODO: this should be mixed with the above code instead of being done separately here!

with open(os.path.join(path, 'train.txt.pickle'), 'rb') as f:
    train = pickle.load(f)
with open(os.path.join(path, 'valid.txt.pickle'), 'rb') as f:
    valid = pickle.load(f)
with open(os.path.join(path, 'test.txt.pickle'), 'rb') as f:
    test = pickle.load(f)

train_kg = train[train[:, 1] != np.max(train[:, 1])]
test_kg = test[test[:, 1] != np.max(train[:, 1])]
valid_kg = valid[valid[:, 1] != np.max(train[:, 1])]

kg_all = np.concatenate((train_kg, test_kg, valid_kg), axis = 0)
test_with_kg = np.concatenate((test, kg_all), axis = 0)
np.random.shuffle(test_with_kg)
# %%
with open(os.path.join(path, 'test_with_kg.txt.pickle'), 'wb') as f:
    pickle.dump(test_with_kg, f)
# %%
np.savetxt(os.path.join(path,'test_with_kg.txt'), test_with_kg, delimiter='\t', fmt='%d')
# %%
# Making the user likes dictionary for filtered evaluation

with open(os.path.join(path, 'train.txt.pickle'), 'rb') as f:
    train = pickle.load(f)
with open(os.path.join(path, 'valid.txt.pickle'), 'rb') as f:
    valid = pickle.load(f)
with open(os.path.join(path, 'test.txt.pickle'), 'rb') as f:
    test = pickle.load(f)

train_rec = train[train[:, 1] == np.max(train[:, 1])]
test_rec = test[test[:, 1] == np.max(train[:, 1])]
valid_rec = valid[valid[:, 1] == np.max(train[:, 1])]

all_rec = np.concatenate((train_rec, test_rec, valid_rec), axis = 0)

user_likes = {}
for line in all_rec:
    user = line[0]
    item = line[2]
    if user not in user_likes:
        user_likes[user] = set()
    user_likes[user].add(item)
# %%
out_file = open(path + '/user_likes.pickle', 'wb')
pickle.dump(user_likes, out_file)
# %%
with open(os.path.join(path, 'train.txt.pickle'), 'rb') as f:
    train = pickle.load(f)
with open(os.path.join(path, 'valid.txt.pickle'), 'rb') as f:
    valid = pickle.load(f)
with open(os.path.join(path, 'test.txt.pickle'), 'rb') as f:
    test = pickle.load(f)

all_data = np.concatenate((train, test, valid), axis = 0) 
all_ents = set(all_data[:, 0]) | set(all_data[:, 2])
# %%
train_rec = train[train[:, 1] == np.max(train[:, 1])]
test_rec = test[test[:, 1] == np.max(train[:, 1])]
valid_rec = valid[valid[:, 1] == np.max(train[:, 1])]

all_rec = np.concatenate((train_rec, test_rec, valid_rec), axis = 0)
items = set(all_rec[:,2])

non_items = np.array(list(all_ents - items))
# %%
np.save(os.path.join(path, 'non_items_array.npy'), non_items)
# %%
with open(os.path.join(path, 'train.txt.pickle'), 'rb') as f:
    train = pickle.load(f)
with open(os.path.join(path, 'valid.txt.pickle'), 'rb') as f:
    valid = pickle.load(f)
with open(os.path.join(path, 'test.txt.pickle'), 'rb') as f:
    test = pickle.load(f)

train_rec = train[train[:, 1] == np.max(train[:, 1])]
test_rec = test[test[:, 1] == np.max(train[:, 1])]
valid_rec = valid[valid[:, 1] == np.max(train[:, 1])]

all_rec = np.concatenate((train_rec, test_rec, valid_rec), axis = 0)

user_likes_train = {}
for line in train_rec:
    user = line[0]
    item = line[2]
    if user not in user_likes_train:
        user_likes_train[user] = set()
    user_likes_train[user].add(item)
# %%
out_file = open(path + '/user_likes_train.pickle', 'wb')
pickle.dump(user_likes_train, out_file)
# %%
