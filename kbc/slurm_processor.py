import os, sys, re
import ast
from prettytable import PrettyTable

if __name__ == '__main__':
    directory = os.getcwd()
    """
        Iterates through all files with ".out" extension in the given directory and looks for the string written in front of 'model':
        and prints it.
        """
    results = {'FB15k-237':{}, 'NELL':{}, 'FB15k':{}}
    for key in list(results.keys()):
        results[key]['cov_anchors'] = []
        results[key]['cov_var'] = []
        results[key]['cov_targets'] = []
        results[key]['hit3s'] = []

    i = 0
    for filename in os.listdir(directory):

        consider_file = False
        
        if filename.endswith(".out"):
            
            with open(os.path.join(directory, filename)) as f:
                for line in f.readlines():
                    if "BPL" in line:
                        consider_file = True
                    if consider_file:
                        
                        if "dataset" in line:
                            start_index = line.index(": ") + 2
                            dataset_name = line[start_index:-1]
                        if "cov_anchor" in line:
                            start_index = line.index(": ") + 2
                            cov_anchor = line[start_index:-1]
                            results[dataset_name]['cov_anchors'].append(float(cov_anchor))
                        if "cov_var" in line:
                            start_index = line.index(": ") + 2
                            cov_var = line[start_index:-1]
                            results[dataset_name]['cov_var'].append(float(cov_var))
                        if "cov_target" in line:
                            start_index = line.index(": ") + 2
                            cov_target = line[start_index:-1]
                            results[dataset_name]['cov_targets'].append(float(cov_target))
                        if "HITS@3m_new'" in line:
                            dict_ = ast.literal_eval(line)
                            hits_3m_new = dict_["HITS@3m_new"]
                            results[dataset_name]['hit3s'].append(hits_3m_new)

    for dataset in ['FB15k-237', 'NELL', 'FB15k']:
        dataset_results = results[dataset]
        dataset_results_sorted = sorted(zip(dataset_results['cov_anchors'],dataset_results['cov_var'], dataset_results['cov_targets'],
                                            dataset_results['hit3s']), key=lambda x: x[3], reverse=True)
        
        sorted_dict = {
            'cov_anchors': [x[0] for x in dataset_results_sorted],
            'cov_var': [x[1] for x in dataset_results_sorted],
            'cov_targets': [x[2] for x in dataset_results_sorted],
            'hit3s': [x[3] for x in dataset_results_sorted]
        }

        table = PrettyTable(['cov_anchors', 'cov_var', 'cov_targets', 'hit3s'])

        for i in range(len(sorted_dict['hit3s'])):
            table.add_row([sorted_dict['cov_anchors'][i], sorted_dict['cov_var'][i], sorted_dict['cov_targets'][i],
                           sorted_dict['hit3s'][i]])
        
        print(table)





                    
                