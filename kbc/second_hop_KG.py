
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
item_rdfs = set()
i2kg_path = os.path.join(root, 'rs/i2kg_map.tsv')
with open(i2kg_path) as f:
    for line in f:
        triple = line.strip().split("\t")
        string = triple[1]
        start_index = string.rfind("/") + 1 
        end_index = string.rfind(">")
        item_rdfs.add(string[start_index:end_index])


# %%
# load e_map.dat and make a dictionary of ent_rdf to item_id
e_map_path = os.path.join(root, 'kg/e_map.dat')
ent_rdf2id = {}
non_item_rdfs = set()
with open(e_map_path) as f:
    for line in f:
        triple = line.strip().split("\t")
        string = triple[1]
        start_index = string.rfind("/") + 1
        end_index = string.rfind(">")
        ent_rdf2id[string[start_index:end_index]] = int(triple[0])
        if string[start_index:end_index] not in item_rdfs:
            non_item_rdfs.add(string[start_index:end_index])
        


# %%
r_map_path = os.path.join(root, 'kg/r_map.dat')
rel_rdf2id = {}
with open(r_map_path) as f:
    for line in f:
        triple = line.strip().split("\t")
        string = triple[1]
        start_index = string.rfind("/") + 1
        end_index = string.rfind(">")
        rel_rdf2id[string[start_index:end_index]] = int(triple[0])

# %%
# checks if an rdf link is in the set (mids) provided
def check(rdf_link, mids):
    start_index = rdf_link.rfind("/") + 1
    end_index = rdf_link.rfind(">")
    mid = rdf_link[start_index:end_index]
    if mid in mids:
        return 1
    else:
        return 0

# %%
# gets the id of each relation or entity (if it's already an id, it returns it as it is)
def get_id_ent(string, ent_or_rel):
    start_index = string.rfind("/") + 1
    end_index = string.rfind(">")
    mid = string[start_index:end_index]
    if ent_or_rel == 'rel':
        try: return rel_rdf2id[mid]
        except: return mid
    elif ent_or_rel == 'ent':
        try: return ent_rdf2id[mid]
        except: return mid

# %%
# opens the freebase KG and writes the second hop KG
freebase_path = os.path.join(root,'..', 'freebase-rdf-latest.gz')
second_hop_kg_name = 'second_hop_kg.dat' 
f = gzip.GzipFile(freebase_path, 'r')
f2 = open(os.path.join(root, 'kg', second_hop_kg_name), 'w')
idx = 0
idx2 = 0
for line in io.TextIOWrapper(f, encoding='utf-8'):
#for line in f:
    idx += 1

    if idx % 10000000 == 0:
        print(idx , idx2)
    line = line.strip()
    if not "<http://rdf.freebase.com/ns/" in line:
        continue
    line2 = line.split("\t")

    # if the head is an item, we want the triple
    if check(line2[0], non_item_rdfs) == 1:
        h , r, t = get_id_ent(line2[0], 'ent'), get_id_ent(line2[1], 'rel'), get_id_ent(line2[2], 'ent')
        f2.write(str(h) + "\t" + str(r) + "\t" + str(t) + "\n")
        idx2 += 1
f2.close()
f.close()

# %%
# print the set of all relations in a file to choose from them the meaningful ones
f2 = open(os.path.join(root, 'kg', second_hop_kg_name), 'r')
f3 = open(os.path.join(root, 'kg', 'relation_names.dat'), 'w')

all_relations = set()
for line in f2:
    try:
        h, r, t = line.strip().split("\t")
    
        all_relations.add(r)
    except:
        continue
for r in all_relations:
    f3.write(str(r) + "\n")
f3.close()




# %%
# next, we need to update the e_map and r_map files and then, replace the kg of freebase ids to numerical ids
# get the list of selected relations
f4 = open(os.path.join(root, 'kg', 'selected_rels.dat'), 'r')
selected_rels = set()
for line in f4:
    selected_rels.add(line.strip())
f4.close()
# %%
existing_rels = {}
f = open(os.path.join(root, 'kg', 'r_map.dat'), 'r')
for line in f:
    r_id, r_rdf = line.strip().split("\t")
    start_index = r_rdf.rfind("/") + 1
    end_index = r_rdf.rfind(">")
    r_name = r_rdf[start_index:end_index]
    if r_name not in existing_rels:
        existing_rels[r_name] = r_id
f.close()
    
for new_rel in selected_rels:
    if new_rel not in existing_rels:
        existing_rels[new_rel] = str(len(existing_rels))    

# %%
f2 = open(os.path.join(root, 'kg', 'second_hop_kg_filtered.dat'), 'w')
f1 = open(os.path.join(root, 'kg', 'second_hop_kg.dat'), 'r')
i1 = 0 ; i2 = 0
for line in f1:
    try:
        h, r, t = line.strip().split("\t")
        if r in selected_rels:
            f2.write(str(h) + "\t" + str(existing_rels[r]) + "\t" + str(t) + "\n")
            i2 += 1
        i1 += 1
        if i1 % 1000000 == 0:
            print(i1, i2)
    except:
        continue
f1.close()
f2.close()
# %%
# update the e_map file

f = open(os.path.join(root, 'kg', 'second_hop_kg_filtered.dat'), 'r')
f1 = open(os.path.join(root, 'kg', 'r_map.dat'), 'a')
f2 = open(os.path.join(root, 'kg', 'e_map.dat'), 'a')
f5 = open(os.path.join(root, 'kg', 'train_secondhop.dat'), 'w')
pattern = r"\w\.\w+"
for line in f:
    if 'XML' in line:
        continue
    h , r , t = line.strip().split("\t")

    if not re.match(pattern, t):
        continue

    if h.isdigit():
        h_id = h

    elif h in ent_rdf2id.keys():
        h_id = ent_rdf2id[h]

    else:

        h_id = len(ent_rdf2id)
        ent_rdf2id[h] = h_id
        f2.write("\n" + str(h_id)+"\t" + str(h))
    if r in existing_rels.keys():
        r_id = existing_rels[r]

    elif r in existing_rels.values():
        r_id = r

        
    else:

        continue

    if t.isdigit():
        t_id = t
    elif t in ent_rdf2id.keys():
        t_id = ent_rdf2id[t]
    else:
        t_id = len(ent_rdf2id)
        ent_rdf2id[t] = t_id
        f2.write("\n" + str(t_id)+"\t" + "<http://rdf.freebase.com/ns/"+str(t)+">")
    f5.write(str(h_id) + "\t" + str(r_id) + "\t" + str(t_id) + "\n")
f.close(); f1.close(); f2.close(); f5.close()
# %%
# update the r_map file
f = open(os.path.join(root, 'kg', 'r_map_twohop.dat'), 'w')
for r in existing_rels.keys():
    f.write(str(existing_rels[r]) + "\t" + "<http://rdf.freebase.com/ns/"+ r +">" + "\n")

f.close()

# %%
onehop = os.path.join(root, 'kg', 'train_onehop.dat')
secondhop = os.path.join(root, 'kg', 'train_secondhop.dat')
all_path = os.path.join(root, 'kg', 'train.dat')

a1 = np.genfromtxt(onehop, delimiter='\t', dtype=np.int32)
a2 = np.genfromtxt(secondhop, delimiter='\t', dtype=np.int32)
a3 = np.concatenate((a1, a2), axis=0)
a3 = np.unique(a3, axis=0)

np.savetxt(all_path, a3, delimiter='\t', fmt='%d')
# %%