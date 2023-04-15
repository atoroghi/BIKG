import os, sys, re

if __name__ == '__main__':
    directory = os.getcwd()
    """
        Iterates through all files with ".out" extension in the given directory and looks for the string written in front of 'model':
        and prints it.
        """
    best_simple_fb = 0; best_simple_batch_size_fb = 0; best_simple_reg_fb = 0; best_simple_rank_fb = 0
    best_simple_nell = 0; best_simple_batch_size_nell = 0; best_simple_reg_nell = 0; best_simple_rank_nell = 0
    best_complex_fb = 0; best_complex_batch_size_fb = 0; best_complex_reg_fb = 0; best_complex_rank_fb = 0
    best_complex_nell = 0; best_complex_batch_size_nell = 0; best_complex_reg_nell = 0; best_complex_rank_nell = 0
    i = 0
    for filename in os.listdir(directory):
        
        if filename.endswith(".out"):

            with open(os.path.join(directory, filename)) as f:
                for line in f.readlines():
                    if "model" in line:
                        if "SimplE" in line:
                            model_name = "SimplE"
                            
                        elif "ComplEx" in line:
                            model_name = "ComplEx"
                            
                    if "dataset" in line:
                        if "FB15k" in line and "237" in line:
                            dataset_name = "FB15k-237"
                            print(filename)
                        elif "NELL" in line:
                            dataset_name = "NELL"
                        else:
                            dataset_name = "other"
                    if "rank" in line:
                        rank = re.search(r'\d+', line).group()
                    if "reg" in line and "zer" not in line:
                        reg = line
                    if "batch_size" in line:
                        batch_size = line
##
                    if "TEST" in line:
                        match = re.search(r"tensor\(\[(\d+\.\d+), (\d+\.\d+), (\d+\.\d+)\]\)", line)
#
                        if match:
                            hit3 = float(match.group(2))
                            if dataset_name == 'FB15k-237':
                                if model_name == 'SimplE':
                                    if hit3 > best_simple_fb:
                                        best_simple_fb = hit3
                                        best_simple_batch_size_fb = batch_size
                                        best_simple_reg_fb = reg
                                        best_simple_rank_fb = rank
                                        best_simple_match_fb = line
                                        best_simple_name_fb = filename
                                elif model_name == 'ComplEx':
                                    if hit3 > best_complex_fb:
                                        best_complex_fb = hit3
                                        best_complex_batch_size_fb = batch_size
                                        best_complex_reg_fb = reg
                                        best_complex_rank_fb = rank
                                        best_complex_match_fb = line
                                        best_complex_name_fb = filename
                        
                            elif dataset_name == 'NELL':
                                if model_name == 'SimplE':
                                    if hit3 > best_simple_nell:
                                        best_simple_nell = hit3
                                        best_simple_batch_size_nell = batch_size
                                        best_simple_reg_nell = reg
                                        best_simple_rank_nell = rank
                                        best_simple_match_nell = line
                                        best_simple_name_nell = filename
                                elif model_name == 'ComplEx':
                                    if hit3 > best_complex_nell:
                                        best_complex_nell = hit3
                                        best_complex_batch_size_nell = batch_size
                                        best_complex_reg_nell = reg
                                        best_complex_rank_nell = rank
                                        best_complex_match_nell = line
                                        best_complex_name_nell = filename


    print("Best SimplE FB15k-237: ", best_simple_fb)
    print("Best SimplE FB15k-237 batch size: ", best_simple_batch_size_fb)
    print("Best SimplE FB15k-237 reg: ", best_simple_reg_fb)
    print("Best SimplE FB15k-237 rank: ", best_simple_rank_fb)
    print("Best SimplE FB15k-237 results: ", best_simple_match_fb)
    print("Best SimplE Name FB15k-237: ", best_simple_name_fb)
    print("Best SimplE NELL: ", best_simple_nell)
    print("Best SimplE NELL batch size: ", best_simple_batch_size_nell)
    print("Best SimplE NELL reg: ", best_simple_reg_nell)
    print("Best SimplE NELL rank: ", best_simple_rank_nell)
    print("Best SimplE NELL results: ", best_simple_match_nell)
    print("Best SimplE Name NELL: ", best_simple_name_nell)
    print("Best ComplEx FB15k-237: ", best_complex_fb)
    print("Best ComplEx FB15k-237 batch size: ", best_complex_batch_size_fb)
    print("Best ComplEx FB15k-237 reg: ", best_complex_reg_fb)
    print("Best ComplEx FB15k-237 rank: ", best_complex_rank_fb)
    print("Best ComplEx FB15k-237 results: ", best_complex_match_fb)
    print("Best ComplEx Name FB15k-237: ", best_complex_name_fb)
    print("Best ComplEx NELL: ", best_complex_nell)
    print("Best ComplEx NELL batch size: ", best_complex_batch_size_nell)
    print("Best ComplEx NELL reg: ", best_complex_reg_nell)
    print("Best ComplEx NELL rank: ", best_complex_rank_nell)
    print("Best ComplEx NELL results: ", best_complex_match_nell)
    print("Best ComplEx Name NELL: ", best_complex_name_nell)