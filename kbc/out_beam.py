import os, sys, re
import numpy as np
from prettytable import PrettyTable
import seaborn as sns
import matplotlib.pyplot as plt
def plot_sanity_check(sorted_dict, k):
    sns.set_theme(style="darkgrid")
    ratios = []
    values = []
    for i in range(len(sorted_dict['cov_anchor'])):
        ratio_float = sorted_dict['cov_target'][i] / sorted_dict['cov_anchor'][i]
        ratio = int(np.log10(ratio_float))
        if ratio in ratios:
            continue
        
        ratios.append(ratio)
        value = [sorted_dict[f'hits{k}_{j}'][i] for j in range(6)]
        values.append(value)
    num_colors = len(values)
    # sort ratios

    range_ratios = np.arange(-(num_colors)//2+1, (num_colors)//2 + 1)
    sorted_ratios, sorted_values = [], []
    for i in range_ratios:
        sorted_ratios.append(i)
        ind = np.where(np.array(ratios) == i)[0][0]
        sorted_values.append(values[ind])
    sorted_ratios.reverse(); sorted_values.reverse()
    sorted_ratios = sorted_ratios[7:14]
    sorted_values = sorted_values[7:14]

    color_palette = sns.color_palette("tab10", n_colors=num_colors)
    sorted_legends = [f'Ratio: 1e{ratio}' for ratio in sorted_ratios]
    #sorted_legends.sort(key=lambda x: float(x.split(':')[1]), reverse=True)
    fig, ax = plt.subplots()
    for i, value in enumerate(sorted_values):
        sns.lineplot(x=range(6), y=value, marker='o', dashes=True, label=sorted_legends[i], color=color_palette[i])
    ax.legend(loc='center right')
    #ax.set_xlim(-1, 6) 
    ax.set_xlabel("Step", fontdict={'fontsize': 18})
    ax.set_ylabel(f"Hits@{k}", fontdict={'fontsize': 18})
    
    plt.savefig(f'sanity_hits_{k}.pdf', format='pdf', dpi=1200)
    plt.show()

if __name__ == '__main__':
    directory = os.getcwd()

    """
        Iterates through all files with ".out" extension in the given directory and looks for the result with
        the greatest hits@10.
        """
    k = 10
    results = {'existential':{}, 'marginal UI':{}, 'instantiated':{}}
    for key in list(results.keys()):
        results[key]['cov_anchor'] = []
        results[key]['cov_var'] = []
        results[key]['cov_target'] = []
        results[key][f'hits{k}_0'] = []; results[key][f'hits{k}_1'] = []; results[key][f'hits{k}_3'] = []
        results[key][f'hits{k}_4'] = []; results[key][f'hits{k}_5'] = []; results[key][f'hits{k}_6'] = []

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
                                if k == 1:
                                    hits = hitsone_existential
                                elif k == 3:
                                    hits = hitsthree_existential
                                elif k == 10:
                                    hits = hitsten_existential
                                results['existential'][f'hits{k}_0'].append(hits[0])
                                results['existential'][f'hits{k}_1'].append(hits[1])
                                results['existential'][f'hits{k}_3'].append(hits[2])
                                results['existential'][f'hits{k}_4'].append(hits[3])
                                results['existential'][f'hits{k}_5'].append(hits[4])
                                results['existential'][f'hits{k}_6'].append(hits[5])

                        elif "marginal_ui" in line or "marginal ui" in line:

                            if "cov_anchor" in line:

                                matches = re.findall(pattern2, line)
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
                                if k == 1:
                                    hits = hitsone_marginal
                                elif k == 3:
                                    hits = hitsthree_marginal
                                elif k == 10:
                                    hits = hitsten_marginal
                                results['marginal UI'][f'hits{k}_0'].append(hits[0])
                                results['marginal UI'][f'hits{k}_1'].append(hits[1])
                                results['marginal UI'][f'hits{k}_3'].append(hits[2])
                                results['marginal UI'][f'hits{k}_4'].append(hits[3])
                                results['marginal UI'][f'hits{k}_5'].append(hits[4])
                                results['marginal UI'][f'hits{k}_6'].append(hits[5])

                        elif "instantiated" in line or 'marginal_i' in line:
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
                                if k == 1:
                                    hits = hitsone_marginal
                                elif k == 3:
                                    hits = hitsthree_marginal
                                elif k == 10:
                                    hits = hitsten_marginal
                                results['instantiated'][f'hits{k}_0'].append(hits[0])
                                results['instantiated'][f'hits{k}_1'].append(hits[1])
                                results['instantiated'][f'hits{k}_3'].append(hits[2])
                                results['instantiated'][f'hits{k}_4'].append(hits[3])
                                results['instantiated'][f'hits{k}_5'].append(hits[4])
                                results['instantiated'][f'hits{k}_6'].append(hits[5])


    for quantifier in ['existential', 'marginal UI', 'instantiated']:
        quantifier_results = results[quantifier]
        
        quantifier_results_sorted = sorted(zip(quantifier_results['cov_anchor'], quantifier_results['cov_var'], quantifier_results['cov_target'],
                                                  quantifier_results[f'hits{k}_0'], quantifier_results[f'hits{k}_1'], quantifier_results[f'hits{k}_3'],
                                                    quantifier_results[f'hits{k}_4'], quantifier_results[f'hits{k}_5'], quantifier_results[f'hits{k}_6']), key=lambda x: x[-1], reverse=True)

        sorted_dict = {
            'cov_anchor': [x[0] for x in quantifier_results_sorted],
            'cov_var': [x[1] for x in quantifier_results_sorted],
            'cov_target': [x[2] for x in quantifier_results_sorted],
            f'hits{k}_0': [x[3] for x in quantifier_results_sorted],
            f'hits{k}_1': [x[4] for x in quantifier_results_sorted],
            f'hits{k}_2': [x[5] for x in quantifier_results_sorted],
            f'hits{k}_3': [x[6] for x in quantifier_results_sorted],
            f'hits{k}_4': [x[7] for x in quantifier_results_sorted],
            f'hits{k}_5': [x[8] for x in quantifier_results_sorted]
        }
        table = PrettyTable(['cov_anchor', 'cov_var', 'cov_target', f'hits{k}_0', f'hits{k}_1', f'hits{k}_3', f'hits{k}_4', f'hits{k}_5', f'hits{k}_6'])

        for i in range(len(sorted_dict['cov_anchor'])):
            table.add_row([sorted_dict['cov_anchor'][i], sorted_dict['cov_var'][i], sorted_dict['cov_target'][i], sorted_dict[f'hits{k}_0'][i], sorted_dict[f'hits{k}_1'][i],
                           sorted_dict[f'hits{k}_2'][i], sorted_dict[f'hits{k}_3'][i], sorted_dict[f'hits{k}_4'][i], sorted_dict[f'hits{k}_5'][i]])

        print(table)
        #if quantifier == 'marginal UI':
        #    plot_sanity_check(sorted_dict, k)
    
        