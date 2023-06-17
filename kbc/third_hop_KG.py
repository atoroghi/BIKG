#%%

import os, sys, re
import numpy as np
import gzip
import pickle
from pathlib import Path
import io
#%%
dataset = 'Movielens'

path = os.path.join(os.getcwd(),'..' , 'data', dataset)
#path = os.path.join(os.getcwd() , 'data', dataset)
root = Path(path)
print(path)
# %%
# get the set of all items to be extracted from the Freebase KG
first_tail_rdfs = set()

# open the file of second hop kg and get the set of its tails

# load e_map.dat and make a dictionary of ent_rdf to item_id (same as code)

# load r_map.dat and make a dictionary of rel_rdf to rel_id (same as code)

# write the check function that
