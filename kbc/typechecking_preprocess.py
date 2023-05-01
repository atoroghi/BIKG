#%%
import os, pickle, sys, time
import torch
import numpy as np
import random
#%%
dataset = 'FB15k'

#%%
# write a function to load data from pickle files
from pathlib import Path
import pickle
path = os.path.join(os.getcwd() ,'data', dataset, 'kbc_data')
root = Path(path)
root = os.path.join(os.getcwd()) 
root = root + '/data/FB15k/kbc_data'
print(root)
print(os.listdir(root))
#%%
data = {}
for f in ['train', 'test', 'valid']:
    in_file = open(root + '/' + f + '.txt.pickle', 'rb')
    data[f] = pickle.load(in_file)
# %%
train = data['train']
test = data['test']
valid = data['valid']
all_data = np.concatenate((train, test, valid), axis = 0)
all_data.shape
# %%
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
# %%
out_file = open(root + '/valid_heads.pickle', 'wb')
pickle.dump(valid_heads, out_file)
out_file.close()

out_file = open(root + '/valid_tails.pickle', 'wb')
pickle.dump(valid_tails, out_file)
out_file.close()

# %%
# %%
