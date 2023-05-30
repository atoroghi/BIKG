import os, sys, re
import numpy as np
from prettytable import PrettyTable

if __name__ == '__main__':
    directory = os.getcwd()

    """
        Iterates through all files with ".out" extension in the given directory and looks for the result with
        the greatest hits@10.
        """
    results = {'existential':{}, 'marginal UI':{}, 'instantiated':{}}
    for key in list(results.keys()):
        results[key]['cov_anchor'] = []
        results[key]['cov_var'] = []
        results[key]['cov_target'] = []
        results[key]['hits10_0'] = []; results[key]['hits10_1'] = []; results[key]['hits10_3'] = []
        results[key]['hits10_4'] = []; results[key]['hits10_5'] = []; results[key]['hits10_6'] = []

    hitsone_existential_all = np.array([]); hitsone_marginal_all = np.array([]); hitsone_instantiated_all = np.array([])
    hitsthree_existential_all = np.array([]); hitsthree_marginal_all = np.array([]); hitsthree_instantiated_all = np.array([])
    hitsten_existential_all = np.array([]); hitsten_marginal_all = np.array([]); hitsten_instantiated_all = np.array([])


    i = 0
    pattern = r"array\((\[.*?\])\)"
    pattern2 = r"\d+(?:\.\d+)?(?:[eE][-+]?\d+)?"

    for filename in os.listdir(directory):
        consider_file = False
        if filename.endswith(".out"):
            with open(os.path.join(directory, filename)) as f:
                for new_line in f.readlines():
                    line = new_line.lower()
                    if "beam" in line:
                        consider_file = True
                    if consider_file:

                        if "existential" in line:
                            if "cov_anchor" in line:
                                matches = re.findall(pattern2, line)
                                cov_anchor = float(matches[0])
                                results['existential']['cov_anchor'].append(cov_anchor)
                            elif "cov_var" in line:
                                matches = re.findall(pattern2, line)
                                cov_var = float(matches[0])
                                results['existential']['cov_var'].append(cov_var)
                            elif "cov_target" in line:
                                matches = re.findall(pattern2, line)
                                cov_target = float(matches[0])
                                results['existential']['cov_target'].append(cov_target)
                            else:
                                matches1 = re.findall(pattern, line)
                                hitsone_existential = np.array(eval(matches1[0]))
                                hitsthree_existential = np.array(eval(matches1[1]))
                                hitsten_existential = np.array(eval(matches1[2]))
                                results['existential']['hits10_0'].append(hitsten_existential[0])
                                results['existential']['hits10_1'].append(hitsten_existential[1])
                                results['existential']['hits10_3'].append(hitsten_existential[2])
                                results['existential']['hits10_4'].append(hitsten_existential[3])
                                results['existential']['hits10_5'].append(hitsten_existential[4])
                                results['existential']['hits10_6'].append(hitsten_existential[5])

                        elif "marginal" in line:
                            if "cov_anchor" in line:
                                print(line)

                                matches = re.findall(pattern2, line)
                                print(matches)
                                cov_anchor = float(matches[0])
                                results['marginal UI']['cov_anchor'].append(cov_anchor)
                            elif "cov_var" in line:
                                matches = re.findall(pattern2, line)
                                cov_var = float(matches[0])
                                results['marginal UI']['cov_var'].append(cov_var)
                            elif "cov_target" in line:
                                matches = re.findall(pattern2, line)
                                cov_target = float(matches[0])
                                results['marginal UI']['cov_target'].append(cov_target)
                            else:
                                matches1 = re.findall(pattern, line)
                                hitsone_marginal = np.array(eval(matches1[0]))
                                hitsthree_marginal = np.array(eval(matches1[1]))
                                hitsten_marginal = np.array(eval(matches1[2]))
                                results['marginal UI']['hits10_0'].append(hitsten_marginal[0])
                                results['marginal UI']['hits10_1'].append(hitsten_marginal[1])
                                results['marginal UI']['hits10_3'].append(hitsten_marginal[2])
                                results['marginal UI']['hits10_4'].append(hitsten_marginal[3])
                                results['marginal UI']['hits10_5'].append(hitsten_marginal[4])
                                results['marginal UI']['hits10_6'].append(hitsten_marginal[5])
                        elif "instantiated" in line:
                            if "cov_anchor" in line:
                                matches = re.findall(pattern2, line)
                                cov_anchor = float(matches[0])
                                results['instantiated']['cov_anchor'].append(cov_anchor)
                            elif "cov_var" in line:
                                matches = re.findall(pattern2, line)
                                cov_var = float(matches[0])
                                results['instantiated']['cov_var'].append(cov_var)
                            elif "cov_target" in line:
                                matches = re.findall(pattern2, line)
                                cov_target = float(matches[0])
                                results['instantiated']['cov_target'].append(cov_target)
                            else:
                                matches1 = re.findall(pattern, line)
                                hitsone_marginal = np.array(eval(matches1[0]))
                                hitsthree_marginal = np.array(eval(matches1[1]))
                                hitsten_marginal = np.array(eval(matches1[2]))
                                results['instantiated']['hits10_0'].append(hitsten_marginal[0])
                                results['instantiated']['hits10_1'].append(hitsten_marginal[1])
                                results['instantiated']['hits10_3'].append(hitsten_marginal[2])
                                results['instantiated']['hits10_4'].append(hitsten_marginal[3])
                                results['instantiated']['hits10_5'].append(hitsten_marginal[4])
                                results['instantiated']['hits10_6'].append(hitsten_marginal[5])

    for quantifier in ['existential', 'marginal UI', 'instantiated']:
        print(quantifier)
        quantifier_results = results[quantifier]
        quantifier_results_sorted = sorted(zip(quantifier_results['cov_anchor'], quantifier_results['cov_var'], quantifier_results['cov_target'],
                                                  quantifier_results['hits10_0'], quantifier_results['hits10_1'], quantifier_results['hits10_3'],
                                                    quantifier_results['hits10_4'], quantifier_results['hits10_5'], quantifier_results['hits10_6']), key=lambda x: x[-1], reverse=True)



        sorted_dict = {
            'cov_anchor': [x[0] for x in quantifier_results_sorted],
            'cov_var': [x[1] for x in quantifier_results_sorted],
            'cov_target': [x[2] for x in quantifier_results_sorted],
            'hits10_0': [x[3] for x in quantifier_results_sorted],
            'hits10_1': [x[4] for x in quantifier_results_sorted],
            'hits10_3': [x[5] for x in quantifier_results_sorted],
            'hits10_4': [x[6] for x in quantifier_results_sorted],
            'hits10_5': [x[7] for x in quantifier_results_sorted],
            'hits10_6': [x[8] for x in quantifier_results_sorted]

        }

        table = PrettyTable(['cov_anchor', 'cov_var', 'cov_target', 'hits10_0', 'hits10_1', 'hits10_3', 'hits10_4', 'hits10_5', 'hits10_6'])

        for i in range(len(sorted_dict['cov_anchor'])):
            table.add_row([sorted_dict['cov_anchor'][i], sorted_dict['cov_var'][i], sorted_dict['cov_target'][i], sorted_dict['hits10_0'][i], sorted_dict['hits10_1'][i],
                           sorted_dict['hits10_3'][i], sorted_dict['hits10_4'][i], sorted_dict['hits10_5'][i], sorted_dict['hits10_6'][i]])

        print(table)

