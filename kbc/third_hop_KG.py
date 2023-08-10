#%%

import os, sys, re
import numpy as np
import gzip
import pickle
from pathlib import Path
import io
#%%
#dataset = 'Movielens_twohop_new'
dataset = 'LastFM_twohop'

path = os.path.join(os.getcwd(),'..' , 'data', dataset)
#path = os.path.join(os.getcwd() , 'data', dataset)
root = Path(path)
print(path)
# %%
# load e_map.dat and make a dictionary of ent_rdf to ent_id and vice versa
e_map_path = os.path.join(root, 'kg/e_map.dat')
ent_rdf2id = {}
ent_id2rdf = {}
with open(e_map_path) as f:
    i  = 0
    for line in f:
        triple = line.strip().split("\t")
        try:
            string = triple[1]
            i +=1
        except:
            print(i)
            print(triple[1])
        start_index = string.rfind("/") + 1
        end_index = string.rfind(">")
        ent_rdf2id[string[start_index:end_index]] = int(triple[0])
        ent_id2rdf[int(triple[0])] = string[start_index:end_index]
# %%
# open the file of second hop kg and get the set of its tails
second_tail_ids = set()
second_tail_rdfs = set()
second_hop_path = os.path.join(root, 'kg/train_secondhop.dat')
with open(second_hop_path) as f:
    for line in f:
        triple = line.strip().split("\t")
        tail = int(triple[2])
        # just remember that the two sets aren't in sync
        second_tail_ids.add(tail)
        second_tail_rdfs.add(ent_id2rdf[tail])
# %%
# load r_map.dat and make a dictionary of rel_rdf to rel_id (same as code)
r_map_path = os.path.join(root, 'kg/r_map_twohop.dat')
rel_rdf2id = {}
with open(r_map_path) as f:
    for line in f:
        triple = line.strip().split("\t")
        string = triple[1]
        start_index = string.rfind("/") + 1
        end_index = string.rfind(">")
        rel_rdf2id[string[start_index:end_index]] = int(triple[0])
# %%
# write the check function that
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
# gets the id of each relation or entity (if it's already an id or not in the dict, it returns it as it is)
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
# opens the freebase KG and writes the third hop KG

freebase_path = os.path.join(root,'..', 'freebase-rdf-latest.gz')
third_hop_kg_name = 'third_hop_kg.dat' 
f = gzip.GzipFile(freebase_path, 'r')
f2 = open(os.path.join(root, 'kg', third_hop_kg_name), 'w')
idx = 0
idx2 = 0
for line in io.TextIOWrapper(f, encoding='utf-8'):
#for line in f:
    idx += 1

    if idx % 10000000 == 0:
        print(idx , idx2)
    line = line.strip()
    if not "<http://rdf.freebase.com/ns/" in line or "XMLSchema#" in line:
        continue
    if "XML" in line or "url" in line or "wikipedia" in line or "time" in line or "label" in line or "_id" in line or "description" in line or "alias" in line or "webpage" in line or "key" in line or "api" in line or "name" in line or "type" in line or "date" in line or "rime" in line or "rating" in line or "track" in line or "trailer" in line or "notable" in line or "value" in line or "topic" in line or "website" in line or "compos" in line:
        continue

    line2 = line.split("\t")

    # if the head is an item, we want the triple
    if check(line2[0], second_tail_rdfs) == 1:
        h , r, t = get_id_ent(line2[0], 'ent'), get_id_ent(line2[1], 'rel'), get_id_ent(line2[2], 'ent')
        f2.write(str(h) + "\t" + str(r) + "\t" + str(t) + "\n")
        idx2 += 1
f2.close()
f.close()


# remember to filter out entiites that only occur few times in the end
# %%
# print the set of all relations in a file to choose from them the meaningful ones
f2 = open(os.path.join(root, 'kg', third_hop_kg_name), 'r')
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
# %%
# next, we need to update the e_map and r_map files and then, replace the kg of freebase ids to numerical ids
# get the list of selected relations
f4 = open(os.path.join(root, 'kg', 'selected_rels.dat'), 'r')
selected_rels = set()
for line in f4:
    selected_rels.add(line.strip())
f4.close()

# %%
# this r_map that we open should be the r_map at the end of second hop
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
# %%
for new_rel in selected_rels:
    if new_rel not in existing_rels:
        existing_rels[new_rel] = str(len(existing_rels))  
# %%
f2 = open(os.path.join(root, 'kg', 'third_hop_kg_filtered.dat'), 'w')
f1 = open(os.path.join(root, 'kg', 'third_hop_kg.dat'), 'r')
i1 = 0 ; i2 = 0
for line in f1:
    try:
        h, r, t = line.strip().split("\t")
        # no new rels allowed (next line should be commented for this case)
        if r in existing_rels:
        #if r.isdigit():

            f2.write(str(h) + "\t" + str(existing_rels[r]) + "\t" + str(t) + "\n")
            #f2.write(str(h) + "\t" + str(r) + "\t" + str(t) + "\n")
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
f = open(os.path.join(root, 'kg', 'third_hop_kg_filtered.dat'), 'r')
f1 = open(os.path.join(root, 'kg', 'r_map.dat'), 'a')
f2 = open(os.path.join(root, 'kg', 'e_map.dat'), 'a')
f5 = open(os.path.join(root, 'kg', 'train_thirdhop.dat'), 'w')
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
    # not allowing new relations (next line should be commented for this case and also the elif)
    if r in existing_rels.keys():
        r_id = existing_rels[r]
    #r_id = r

    elif r in existing_rels.values():
        r_id = r

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
# instead of updating the r_map, I decided to keep the number of relations fixed and just remove new triples
#existing_rels = {}
#f = open(os.path.join(root, 'kg', 'r_map.dat'), 'r')
#for line in f:
#    r_id, r_rdf = line.strip().split("\t")
#    start_index = r_rdf.rfind("/") + 1
#    end_index = r_rdf.rfind(">")
#    r_name = r_rdf[start_index:end_index]
#    if r_name not in existing_rels:
#        existing_rels[r_name] = r_id
#f.close()
# %%
# update the r_map file. remember after this file is saved, make this the r_map.dat and previous one r_map_twohop
f = open(os.path.join(root, 'kg', 'r_map_threehop.dat'), 'w')
for r in existing_rels.keys():
    f.write(str(existing_rels[r]) + "\t" + "<http://rdf.freebase.com/ns/"+ r +">" + "\n")

f.close()
#%%
# filter out rare entities from two and three hop kgs and form the last version of the kg
secondhop = os.path.join(root, 'kg', 'train_secondhop.dat')
a2 = np.genfromtxt(secondhop, delimiter='\t', dtype=np.int32)
column3_counts = np.bincount(a2[:, 2])
#values_to_delete = np.where((column3_counts < 5))[0]   # for Movielens
values_to_delete = np.where((column3_counts < 3))[0]    # for LastFM

# %%
a2 = a2[ ~np.isin(a2[:, 2], values_to_delete)]
# %%
# filter out rare entities from two and three hop kgs and form the last version of the kg
thirdhop = os.path.join(root, 'kg', 'train_thirdhop.dat')
a3 = np.genfromtxt(thirdhop, delimiter='\t', dtype=np.int32)
column3_counts2 = np.bincount(a3[:, 2])
#values_to_delete2 = np.where((column3_counts2 < 5))[0]  # for Movielens
values_to_delete2 = np.where((column3_counts2 < 3))[0]  # for LastFM


# %%
# deleting second hop's rare entities from the third hop kg
a3 = a3[~np.isin(a3[:, 0], values_to_delete)]
# %%
# deleting rare entities from the third hop kg
a4 = a3[~np.isin(a3[:, 2], values_to_delete2)]
# %%
onehop = os.path.join(root, 'kg', 'train_onehop.dat')
all_path = os.path.join(root, 'kg', 'train.dat')

a1 = np.genfromtxt(onehop, delimiter='\t', dtype=np.int32)

a = np.vstack((a1, a2, a4))

a = np.unique(a, axis=0)

np.savetxt(all_path, a, delimiter='\t', fmt='%d')
# %%
