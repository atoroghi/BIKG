#%%
#%%
from kbc.datasets import Dataset
import itertools
import argparse
import pickle
import os,sys
import random
import numpy as np

from tqdm import tqdm as tqdm


class Chain():
    def __init__(self):
        # anchors are the anchor entities that are not optimisable
        # optimisable should be entities in between and the target entity
        self.data = {'raw_chain':[], 'anchors': [], 'optimisable': [], 'user': [],'item': [], 'type':None}

class ChaineDataset():
    def __init__(self, dataset: Dataset, threshold:int=1e6):


        if dataset is not None:
            self.threshold = threshold

            self.raw_data = dataset
            #rhs_missing is a dictionary with keys being tuples of (entity, relation) and values being the list of tails of that entity-relation pair in the test set
            self.rhs_missing = self.raw_data.to_skip['rhs']
            self.lhs_missing = self.raw_data.to_skip['lhs']
            # likes relation is the greatest rel in the dataset (rels begin from 0)
            self.likes_rel = int(np.max(self.raw_data.data['train'][:,1]))

            # merges the lhs and rhs missing into one dictionary
            self.full_missing = {**self.rhs_missing, **self.lhs_missing}

            # reforms the test triples into a set of tuples
            #self.train_set = set((tuple(triple) for triple in self.raw_data.data['train']))
            #self.valid_set = set((tuple(triple) for triple in self.raw_data.data['valid']))
            self.test_set = set((tuple(triple) for triple in self.raw_data.data['test']))
            self.general_rels = [self.likes_rel]
            #if 'amazon-book' in str(dataset.root):
            #    self.general_rels.append(0)
            #    self.general_rels.append(1)

#%%
#%%

            

        self.neighbour_relations = {}
        self.reverse_maps = {}

        self.type1_1chain = []
        self.type1_2chain = []
        self.type2_2chain = []
        self.type2_2chain_u = []

        self.type1_3chain = []
        self.type1_4chain = []
        self.type2_3chain = []
        self.type3_3chain = []
        self.type4_3chain = []
        self.type4_3chain_u = []

        self.users = []
        self.items = []



    def sample_chains(self):
        try:
            self.__get_neighbour_relations__()
            self.neighbour_relations
            self.__reverse_maps__()
            # current chains: 1_2, 2_2, 2_3
            self.__type1_2chains__()
            self.__type2_2chains__()
            #self.__type1_3chains__()
            #self.__type1_4chains__()
            self.__type2_3chains__()
            #self.__type3_3chains__()
            #self.__type4_3chains__() 

        except RuntimeError as e:
            print(e)

#this function gets relations that are connected to each entity in the test set and stores them in a list
    def __get_neighbour_relations__(self):
        try:

            for i in list(self.rhs_missing.keys()):
                if i[0] not in self.neighbour_relations:
                    self.neighbour_relations[i[0]] = []
                if i[1] not in self.general_rels:
                    self.neighbour_relations[i[0]].append(i[1])

            # for now, we're just focusing on the rhs_missing (item being the head of the triple and going from head to tail)
            # this is the majority of the triples in the kg
            #for i in list(self.lhs_missing.keys()):
            #    if i[0] not in self.neighbour_relations:
            #        self.neighbour_relations[i[0]] = []
            #    if i[1] not in self.general_rels:
            #        self.neighbour_relations[i[0]].append(i[1])

        except Exception as e:
            print(1)
# reverses keys and values in the rhs_missing dictionary
    def __reverse_maps__(self):

        for keys,vals in self.rhs_missing.items():
            for val in vals:
                if val not in self.reverse_maps:
                    self.reverse_maps[val] = []

                self.reverse_maps[val].append(keys)
#%%
#%%
    # This must be 2p
    def __type1_2chains__(self):

        try:
            # taking each triple in the test set e.g., (13, 1, 51)
            for test_triple in tqdm(self.raw_data.data['test_with_kg'][1060617:]):
            #for test_triple in tqdm(self.raw_data.data['test_with_kg'][:1060617]):
            #for test_triple in tqdm(self.raw_data.data['test_with_kg']):
                if test_triple[1] == self.likes_rel and test_triple[2] in self.reverse_maps:
                    user = test_triple[0]
                    item = test_triple[2]
                    self.users.append(user)
                    self.items.append(item)
                    #print(test_triple)

                    
                    # first part of the chain is the user and item
                    test_lhs_chain_1 = (test_triple[0], test_triple[1])
                    #print(test_lhs_chain_1)
                    #sys.exit()
                    
                    # item is added to answers
                    test_answers_chain_1 = [test_triple[2]]
                    # neighbour relations of the tail (answer) are the potential continuations of the chain
                    potential_chain_cont = [(x, self.neighbour_relations[x][:5]) for x in test_answers_chain_1]
                    #print(potential_chain_cont)                    
                    #sys.exit()
                    # potential is a tuple of the answer and the neighbour relations of each answer
                    for potential in potential_chain_cont:

                        # x is each neighbour relation
                        # segmented_list is a list of tuples of the answer and each neighbour relation

                        segmented_list = [(potential[0],x) for x in potential[1] if (x not in self.general_rels)]
                        continuations = [ [x,self.rhs_missing[x][:1]] for x in  segmented_list if x in self.rhs_missing]
                        #print(continuations)
                        #sys.exit()
                        ans_1 = [potential[0]]
                        #print(len(continuations))
                        

                        # we want to have at least 5 facts for each user, item pair
                        if len(continuations) < 5:
                            break
                    
                        # raw_chains includes both parts of the chain. the first part is the original triple and the second part is the continuation of the chain
                        raw_chains = [
                            [ list(test_lhs_chain_1) +  ans_1,  [x[0][0], x[0][1], x[1]] ]

                            for x in continuations[:5]
                        ]
                        #print(raw_chains)
                        #sys.exit()
                        # raw_chain: [ [user, likes, item], [item, relation, [tails]] ]

                        # storing raw_chains in a list of Chain objects and updating its attributes
                        for chain in raw_chains:
                            new_chain = Chain()
                            new_chain.data['type'] = '1chain2'
                            new_chain.data['raw_chain'] = chain
                            
                            new_chain.data['user']= chain[0][0]
                            new_chain.data['item'] = chain[0][2]
                            # each of the tails of the second part of the chain can be an anchor
                            new_chain.data['anchors'].append(chain[1][2])
                            new_chain.data['optimisable'].append(chain[0][2])
                            new_chain.data['optimisable'].append(chain[0][0])

                            self.type1_2chain.append(new_chain)

                            if len(self.type1_2chain) > self.threshold:
                                print(f'1_2:{len(self.type1_2chain)}')
                                #for chain in  self.type1_2chain[:10]:
                                #    print(chain.data['raw_chain']) 
                                #sys.exit()

                                print("Threshold for sample amount reached")
                                print("Finished sampling chains with legth 2 of type 1")
                                return

            print("Finished sampling chains with legth 2 of type 1")



        except RuntimeError as e:
            print(e)

    # this is 2i
    def __type2_2chains__(self):
        try:
            #print(self.reverse_maps[17379])
            #sys.exit()
            for test_triple in tqdm(self.raw_data.data['test_with_kg'][1060617:]):
            #for test_triple in tqdm(self.raw_data.data['test_with_kg'][:1060617]):
            #for test_triple in tqdm(self.raw_data.data['test_with_kg']):
                if test_triple[1] == self.likes_rel and test_triple[2] in self.reverse_maps:
                    #print(test_triple)
                    #sys.exit()
                    user = test_triple[0]
                    item = test_triple[2]
                    ans = item          
                    #print(ans)
                    #print(self.reverse_maps[ans])
                    #sys.exit()
                    

            #for ans in tqdm(self.reverse_maps):

                    # we're not using the common lhs anymore, but we're interested in the item to be the head entity
                    #common_lhs = [x for x in self.reverse_maps[ans] if x[1] not in self.general_rels]
                    #print(common_lhs)
                    #print(len(common_lhs))
                    potential_chain_rel = [(ans, x) for x in self.neighbour_relations[ans][:5]]
                    potential_chain_cont = []
                    for potential_rel in potential_chain_rel:
                        for potential_tail in self.rhs_missing[potential_rel]:
                            potential_chain_cont.append(potential_rel + (potential_tail,)) 
                    potential_chain_cont_clean = random.sample(potential_chain_cont, 5)
                    #print(potential_chain_cont_clean)
                    # ensuring that we'll have at least 5 facts for each user, item pair
                    if len(potential_chain_cont_clean)<5:
                        continue
                    
                    common_lhs_clean = list(itertools.combinations(potential_chain_cont_clean, 2))
                    #print(common_lhs)
                    #sys.exit()
                    # we don't care about the chains to be only from the test set anymore
                    #common_lhs_clean = []
                    #for segments in common_lhs:
                    #    for s in segments:
                    #        if s + (ans,) in self.test_set:
                    #            common_lhs_clean.append(segments)
                    #            break
                    
                    #print(len(common_lhs_clean))
                    #sys.exit()
                    
                    if len(common_lhs_clean) == 0:
                        continue
                    elif len(common_lhs_clean) > 5:
                        common_lhs_clean = random.sample(common_lhs_clean, 5)
                    # raw_chains = [[[user, likes, item], [item, rel, anchor], [item, rel, anchor]]]
                    raw_chains = [[[user, self.likes_rel, item], list(x[0]), list(x[1])] for x in common_lhs_clean]

                    #raw_chains = [ [ list(x[0])+[ans], list(x[1])+[ans], [user,self.likes_rel, item] ]  for x in common_lhs_clean]
                    
                    #print(raw_chains)
                    for chain in raw_chains:
                        new_chain = Chain()

                        new_chain.data['type'] = '2chain2'

                        new_chain.data['raw_chain'] = chain
                        new_chain.data['anchors'].append(chain[0][0])
                        new_chain.data['anchors'].append(chain[1][0])

                        new_chain.data['optimisable'].append(chain[0][2])
                        new_chain.data['optimisable'].append(chain[2][0])

                        new_chain.data['user'] = chain[2][0]
                        new_chain.data['item'] = chain[2][2]

                        self.type2_2chain.append(new_chain)
                        #print(len(self.type2_2chain))

                        if len(self.type2_2chain) > self.threshold:
                            print(f'2_2:{len(self.type2_2chain)}')
                            
                            #for chain in self.type2_2chain:
                            #    print(chain.data['raw_chain'])

                            print("Threshold for sample amount reached")
                            print("Finished sampling chains with legth 2 of type 2")
                            return

            print("Finished sampling chains with legth 2 of type 2")

        except RuntimeError as e:
            print(e)

# this is 3p
    def __type1_3chains__(self):
        try:
            #Appending routine

            potential_chain_cont_records = {}
            # we can't use chains 1_2 anymore because they might be insufficient for making 5 continuations
            #for chain in tqdm(self.type1_2chain):
            for test_triple in tqdm(self.raw_data.data['test_with_kg'][:1060617]):
                if test_triple[1] == self.likes_rel and test_triple[2] in self.reverse_maps:
                    #print(test_triple)
                    #sys.exit()
                    user = test_triple[0]
                    item = test_triple[2]
                    ans = item          
                    #print(ans)
                    #print(self.reverse_maps[ans])
                    #sys.exit()
                    # first part of the chain is the user and item
                    test_lhs_chain_1 = (test_triple[0], test_triple[1])
                    # item is added to answers
                    test_answers_chain_1 = [test_triple[2]]
                    potential_chain_cont = [(x, self.neighbour_relations[x]) for x in test_answers_chain_1]

                    for potential in potential_chain_cont:

                        segmented_list = [(potential[0],x) for x in potential[1] if (x not in self.general_rels)]
                        continuations = []
                        for x in segmented_list:
                            if x in self.rhs_missing:
                                potential_tails = [y for y in self.rhs_missing[x] if y in self.general_rels]
                        
                        # our current kg is not deep enough to make a 3p chain
                        if len(potential_tails) == 0:
                            continue
                        print(potential_tails)
                        sys.exit()
                        continuations = [ [x,self.rhs_missing[x]] for x in  segmented_list if x in self.rhs_missing]
                        print(continuations)
                        sys.exit()

                
                
                ans_2chain = [x for x in ans_2chain if x != raw_chain[0][0]]
                print(ans_2chain)
                print(self.neighbour_relations[ans_2chain[0]])
                sys.exit()
                
                # TODO: this should be problemmatic
                potential_chain_cont = [random.choice([(x, list(set(self.neighbour_relations[x])-set(self.general_rels))) for x in ans_2chain])]
                print(potential_chain_cont)
                #sys.exit()

                while (potential_chain_cont in potential_chain_cont_records[tuple(raw_chain[0])]):
                    potential_chain_cont = [random.choice([(x, list(set(self.neighbour_relations[x])-set(self.general_rels))) for x in ans_2chain])]
                potential_chain_cont_records[tuple(raw_chain[0])].append(potential_chain_cont)
                print(potential_chain_cont_records)
                #sys.exit()
                

                for potential in potential_chain_cont:
                    print(potential)
                    sys.exit()

                    segmented_list = [(potential[0],x) for x in potential[1]]

                    continuations = [ [x,self.rhs_missing[x][:1]] for x in  segmented_list
                                     if x in self.rhs_missing][:1]
                    print(continuations)
                    sys.exit()


                    # we want to have at least 5 facts for each user, item pair
                    #if len(continuations) < 5:
                    #    break

                    ans_connector = potential[0]   
                    
                                     
                    # new_chains = [[[user, likes, item], [item, relation, tail], [tail, relation, [tails]]]]

                    new_chains = \
                    [
                        [
                            raw_chain[0], [raw_chain[1][0], raw_chain[1][1], ans_connector],
                            [ans_connector,x[0][1], x[1]]

                        ]
                        for x in continuations
                    ]                    
                    print(new_chains)

                    for chain in new_chains:
                        new_chain = Chain()

                        new_chain.data['type'] = '1chain3'

                        new_chain.data['raw_chain'] = chain

                        new_chain.data['anchors'].append(chain[2][2])

                        new_chain.data['optimisable'].append(chain[0][2])
                        new_chain.data['optimisable'].append(chain[1][2])
                        # IMPORTANT: this must be chain[2][2] and not chain[1][2]
                        #new_chain.data['optimisable'].append(chain[1][2])
                        new_chain.data['optimisable'].append(chain[0][2])

                        new_chain.data['user'] = chain[0][0]
                        new_chain.data['item'] = chain[0][2]


                        self.type1_3chain.append(new_chain)

                        if len(self.type1_3chain) > self.threshold:


                            print("Threshold for sample amount reached")
                            print("Finished sampling chains with legth 3 of type 1")

                            return

            print("Finished sampling chains with legth 3 of type 1")


        except RuntimeError as e:
            print(e)


# this is 4p
    def __type1_4chains__(self):
        try:
            #Appending routine
            # chain = [ [user,likes, item], [item, rel, tail], [trail, rel2, [tails]] ]
            potential_chain_cont_records = {}
            for chain in tqdm(self.type1_3chain):
                raw_chain = chain.data['raw_chain']
                raw_chain_rels = [x[1] for x in raw_chain]

                potential_chain_cont_records[tuple(raw_chain[0])] = []

                ans_2chain = raw_chain[2][2]

                ans_2chain = [x for x in ans_2chain if x != raw_chain[1][0]]

                #potential_chain_cont = [(x, self.neighbour_relations[x]) for x in ans_2chain][:5]
                # randomly select the chain continuation to avoid bias
                potential_chain_cont_candidates = [[(x, list(set(self.neighbour_relations[x])-set(self.general_rels))) for x in ans_2chain]]
                if len(potential_chain_cont_candidates[0]) == 0:
                    continue
                potential_chain_cont_clean = []

                # make sure the same rel is not selected twice
                for x in potential_chain_cont_candidates:
                    try:
                        candidate_rels = [y for y in x[0][1] if y not in raw_chain_rels]
                    except:
                        print(potential_chain_cont_candidates)
                potential_chain_cont_clean = [(potential_chain_cont_candidates[0][0][0], candidate_rels)]
                
                # in this case all clean candidates are already in the records
                if len(potential_chain_cont_clean) == potential_chain_cont_records[tuple(raw_chain[0])]:
                    continue

                potential_chain_cont = [random.choice(potential_chain_cont_clean)]

                # make sure the same continuation is not sampled twice
                while (potential_chain_cont in potential_chain_cont_records[tuple(raw_chain[0])]):
                    potential_chain_cont = [random.choice(potential_chain_cont_clean)]

                potential_chain_cont_records[tuple(raw_chain[0])].append(potential_chain_cont)

                for potential in potential_chain_cont:

                    segmented_list = [(potential[0],x) for x in potential[1]]

                    continuations = [ [x,self.rhs_missing[x]] for x in  segmented_list
                                     if (x in self.rhs_missing and len(self.rhs_missing)>0)][-1]
                    # we want to have at least 5 facts for each user, item pair
                    #if len(continuations) < 5:
                    #    break

                    ans_connector = potential[0]


                    new_chains = \
                    [
                        [
                            raw_chain[0], raw_chain[1], [raw_chain[2][0], raw_chain[2][1], ans_connector],
                            [ans_connector,continuations[0][1], continuations[1]]

                        ]
                        
                    ]
                    # new_chains = [[[user, likes, item], [item, relation, tail], [tail, relation, tail], [tail, relation, [tails]]

                    for chain in new_chains:
                        new_chain = Chain()

                        new_chain.data['type'] = '1chain4'

                        new_chain.data['raw_chain'] = chain

                        new_chain.data['anchors'].append(chain[3][0])

                        new_chain.data['optimisable'].append(chain[0][2])
                        new_chain.data['optimisable'].append(chain[1][2])
                        # IMPORTANT: this must be chain[2][2] and not chain[1][2]
                        #new_chain.data['optimisable'].append(chain[1][2])
                        new_chain.data['optimisable'].append(chain[2][2])
                        new_chain.data['optimisable'].append(chain[0][0])

                        new_chain.data['user'] = chain[0][0]
                        new_chain.data['item'] = chain[0][2]


                        self.type1_4chain.append(new_chain)

                        if len(self.type1_4chain) > self.threshold:

                            print("Threshold for sample amount reached")
                            print("Finished sampling chains with legth 4 of type 1")

                            return

            print("Finished sampling chains with legth 4 of type 1")


        except RuntimeError as e:
            print(e)



# this is 3i
    def __type2_3chains__(self):
        try:
            for test_triple in tqdm(self.raw_data.data['test_with_kg'][1060617:]):
            #for test_triple in tqdm(self.raw_data.data['test_with_kg'][:1060617]):
            #for test_triple in tqdm(self.raw_data.data['test_with_kg']):
                if test_triple[1] == self.likes_rel and test_triple[2] in self.reverse_maps:
                    #print(test_triple)
                    user = test_triple[0]
                    item = test_triple[2]
                    ans = item

                    potential_chain_rel = [(ans, x) for x in self.neighbour_relations[ans]]
                    potential_chain_cont = []

                    for potential_rel in potential_chain_rel:
                        for potential_tail in self.rhs_missing[potential_rel]:
                            potential_chain_cont.append(potential_rel + (potential_tail,)) 

                    if len(potential_chain_cont)<5:
                        continue

                    common_lhs_clean = list(itertools.combinations(potential_chain_cont, 3))


                    common_lhs_clean = random.sample(common_lhs_clean, 5)

            #for ans in tqdm(self.reverse_maps):
                



                    #common_lhs = [x for x in self.reverse_maps[ans] if x[1] not in self.general_rels]
                    # ensure that we have at least 5 facts for each user, item pair
                    #if len(common_lhs)<6:
                    #    continue
                    #elif len(common_lhs) >15:
                    #    common_lhs = common_lhs[:15]

                    #common_lhs = list(itertools.combinations(common_lhs, 3))
#
                    #common_lhs_clean = []
                    #for segments in common_lhs:
                    #    for s in segments:
                    #        if s + (ans,) in self.test_set:
                    #            common_lhs_clean.append(segments)
                    #            break

                    #if len(common_lhs_clean) == 0:
                    #    continue

                    #elif len(common_lhs_clean) > 10:
                    #    common_lhs_clean = common_lhs_clean[:10]

                        # raw_chains = [[tail, rel, item],[tail, rel, item],[tail, rel, item], [user, likes, item]]
                    # raw_chains = [[user, likes, item], [item, rel, tail], [item, rel, tail], [item, rel, tail]]

                    raw_chains = [[[user, self.likes_rel, item] , list(x[0]), list(x[1]), list(x[2])] for x in common_lhs_clean]

                    #raw_chains = [ [ list(x[0])+[ans], list(x[1])+[ans], list(x[2]) + [ans], [user,self.likes_rel, item]]
                    #              for x in common_lhs_clean]

                    for chain in raw_chains:
                        new_chain = Chain()

                        new_chain.data['type'] = '2chain3'

                        new_chain.data['raw_chain'] = chain

                        new_chain.data['anchors'].append(chain[1][2])
                        new_chain.data['anchors'].append(chain[2][2])
                        new_chain.data['anchors'].append(chain[3][2])


                        new_chain.data['optimisable'].append(chain[0][0])
                        new_chain.data['optimisable'].append(chain[0][2])

                        new_chain.data['user'] = chain[0][0]
                        new_chain.data['item'] = chain[0][2]

                        self.type2_3chain.append(new_chain)

                        if len(self.type2_3chain) > self.threshold:
                            print((f'2_3:{len(self.type2_3chain)}'))
                            # for chain in self.type2_3chain:
                            #     print(chain.data['raw_chain'])

                            print("Threshold for sample amount reached")
                            print("Finished sampling chains with legth 3 of type 2")

                            return

            print("Finished sampling chains with legth 3 of type 2")


        except RuntimeError as e:
            print(e)
# this is pi
    def __type3_3chains__(self):

        try:
            for chain in tqdm(self.type2_2chain):

                raw_chain_initial = chain.data['raw_chain']
                connector_node_opt0 = raw_chain_initial[0][0]
                connector_node_opt1 = raw_chain_initial[1][0]

                if connector_node_opt0 in self.reverse_maps:
                    common_lhs_0 = [x for x in self.reverse_maps[connector_node_opt0] if x[1] not in self.general_rels]

                else:
                    common_lhs_0 = []
                if len(common_lhs_0) >10:
                    common_lhs_0 = common_lhs_0[:10]


                if connector_node_opt1 in self.reverse_maps:
                    common_lhs_1 = [x for x in self.reverse_maps[connector_node_opt1] if x[1] not in self.general_rels]
                else:
                    common_lhs_1 = []


                if len(common_lhs_1) >10:
                    common_lhs_1 = common_lhs_1[:10]
                # ensure that we have at least 5 facts for each user, item pair
                if len(common_lhs_0) + len(common_lhs_1) < 5:
                    continue


                potential_additions_0 = [ list(x)+[connector_node_opt0] for x in common_lhs_0]
                potential_additions_1 = [ list(x)+[connector_node_opt1] for x in common_lhs_1]
               
                

                raw_chains_0 = \
                [
                    [
                        x, raw_chain_initial[0], raw_chain_initial[1], raw_chain_initial[2]
                    ]

                    for x in potential_additions_0
                ]

                raw_chains_1 = \
                [
                    [
                        x, raw_chain_initial[1],raw_chain_initial[0], raw_chain_initial[2]
                    ]

                    for x in potential_additions_1
                ]
                #print(raw_chains_1)
                #sys.exit()
                # raw_chains = [[tail3, rel, tail],[tail2, rel, item],[tail1, rel, item], [user, likes, item]]

                raw_chains = raw_chains_0 + raw_chains_1

                for chain in raw_chains:

                    new_chain = Chain()

                    new_chain.data['type'] = '3chain3'

                    new_chain.data['raw_chain'] = chain

                    new_chain.data['anchors'].append(chain[0][0])
                    new_chain.data['anchors'].append(chain[2][0])

                    new_chain.data['optimisable'].append(chain[0][2])
                    new_chain.data['optimisable'].append(chain[1][2])
                    new_chain.data['optimisable'].append(chain[2][2])
                    new_chain.data['optimisable'].append(chain[3][0])


                    new_chain.data['user'].append(chain[3][0])
                    new_chain.data['item'].append(chain[3][2])

                    self.type3_3chain.append(new_chain)

                    if len(self.type3_3chain) > self.threshold:


                        print("Threshold for sample amount reached")
                        print("Finished sampling chains with legth 3 of type 3")
                        return

            print("Finished sampling chains with legth 3 of type 3")

        except RuntimeError as e:
            print(e)

# this is ip
    def __type4_3chains__(self):
        chains_recorder = []
        try:
            # first part of the chain is the same as in 2p (user, item, sth)
            for chain in tqdm(self.type1_2chain):
                # the rest is like a normal 2 i (not with item as the connector node)

                raw_chain_initial = chain.data['raw_chain']
                #print(raw_chain_initial)
                first_part_raw = raw_chain_initial[0]
                anss = raw_chain_initial[1][2][:15]
                #print(anss)
                for ans in anss:


                    second_part_raw = [raw_chain_initial[1][0], raw_chain_initial[1][1], ans]
                    #print(second_part_raw)
                    

                
                    #chain_top_initial = raw_chain_initial[0][2]

                    #chain_potential_predicates = self.neighbour_relations[chain_top_initial]

                    #chain_potential_lhs = [(chain_top_initial,x) for x in chain_potential_predicates]

                    common_lhs = [x for x in self.reverse_maps[ans] if x[1] not in self.general_rels]



                    if len(common_lhs) < 5:
                        continue
                    elif len(common_lhs) > 5:
                        common_lhs = common_lhs[:10]
        
                    common_lhs = list(itertools.combinations(common_lhs, 2))


                    common_lhs_clean = []
                    for segments in common_lhs:
                        for s in segments:
                            if s + (ans,) in self.test_set:
                                common_lhs_clean.append(segments)
                                break


                    if len(common_lhs_clean) == 0:
                        continue
                    #elif len(common_lhs_clean) > 5:
                    #    common_lhs_clean = common_lhs_clean[:2]
                    else:
                        common_lhs_clean = common_lhs_clean[:1]

                    # raw_chains = [[tail2, rel, tail1],[tail3, rel, tail1],[item, rel, tail1], [user, likes, item]]
                    raw_chains = [ [ list(x[0])+[ans], list(x[1])+[ans], second_part_raw, first_part_raw]  for x in common_lhs_clean]

                    for chain in raw_chains:


                        new_chain = Chain()
                        new_chain.data['type'] = '4chain3'
                        new_chain.data['raw_chain'] = chain
                        new_chain.data['anchors'].append(chain[0][0])
                        new_chain.data['anchors'].append(chain[1][0])
                        new_chain.data['optimisable'].append(chain[0][2])
                        new_chain.data['optimisable'].append(chain[2][0])
                        new_chain.data['optimisable'].append(chain[3][0])
                        new_chain.data['user'].append(chain[3][0])
                        new_chain.data['item'].append(chain[3][2])
                        self.type4_3chain.append(new_chain)
                        #print(len(self.type4_3chain))
                        if len(self.type4_3chain) >= self.threshold:
                            for chain in self.type4_3chain:
                                print(chain.data['raw_chain'])
                            

                            print("Threshold for sample amount reached")
                            print("Finished sampling chains with legth 3 of type 4")
                            return

            #TODO: remove records with less than 5 chains
            print("Finished sampling chains with legth 3 of type 2")



        except RuntimeError as e:
            print(e)



def save_chain_data(save_path, dataset_name, data):
    try:

        full_path = os.path.join(save_path,dataset_name+".pkl")

        with open(full_path, 'wb') as f:
            pickle.dump(data,f,-1)

        print("Chain Dataset for {} saved at {}".format(dataset_name,full_path))

    except RuntimeError as e:
        print(e)

def load_chain_data(data_path):
    data = None
    try:
        with open(data_path,'rb') as f:
            data= pickle.load(f)
    except RuntimeError as e:
        print(e)
        return data
    return data


if __name__ == "__main__":

    big_datasets = ['amazon-book', 'yelp2018', 'Movielens', 'last-fm', 'FB15k']
    datasets = big_datasets

    parser = argparse.ArgumentParser(
    description="Chain Dataset Sampling"
    )


    parser.add_argument(
    '--dataset', choices=datasets,
    help="Dataset in {}".format(datasets)
    )

    parser.add_argument(
    '--threshold',default = 1e5,type=int,
    help="Threshold for maximum amount sampled per chain type"
    )

    parser.add_argument(
    '--save_path',default = os.getcwd(),
    help="Path to save the chained dataset"
    )

    args = parser.parse_args()

    chained_dataset_sampler = ChaineDataset( Dataset(os.path.join('data',args.dataset,'kbc_data')),args.threshold)
    chained_dataset_sampler.sample_chains()

    save_chain_data(args.save_path,args.dataset,chained_dataset_sampler)
