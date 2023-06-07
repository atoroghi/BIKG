import torch
import numpy as np
import os
import argparse
import pickle
import json
import sys
import random
import tqdm
import pandas as pd
from sklearn.model_selection import train_test_split
from collections import defaultdict


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

#dataset = 'Movielens'
dataset = 'LastFM'

from pathlib import Path
import pickle
path = os.path.join(os.getcwd() ,'..', 'data', dataset)

root = Path(path)
print(root)
print(os.listdir(root))  

#%%   

# opening the i2kg file
# find each of the item relation tails in the freebase
# when found, find the corresponding head or tail again in the freebase

# add the triple to the kg, ent to the entities, and rel to the relations (if not already present)