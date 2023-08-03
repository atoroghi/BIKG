import os, sys
import subprocess


if __name__ == '__main__':

    # Save original stdout for later restoration
    original_stdout = sys.stdout

    # Open a file for writing

    
    #cov_anchors = [1e5]
    #cov_vars = [1e5]
    #cov_targets = [1e5]
    cov_anchors = [1e-3,1e-2, 1e-1,1e1,1e2,1e3]
    cov_vars = [1e-3,1e-2, 1e-1,1e1,1e2,1e3]
    cov_targets = [1e-3,1e-2, 1e-1,1e1,1e2,1e3]

    #cov_anchors = [1e-2, 1e-1,1e1, 1e2]
    #cov_vars = [1e-2, 1e-1,1e1, 1e2]
    #cov_targets = [1e-2, 1e-1,1e1, 1e2]
    experiment_name = '1_3_marginal_inst_new'
    chain_type = '1_3'
    os.mkdir(experiment_name)


    for cov_anchor in cov_anchors:
        for cov_var in cov_vars:
            for cov_target in cov_targets:
                output_file = f"slurm_{cov_anchor}_{cov_var}_{cov_target}.out"

                #command = ["python3", "-m", "kbc.cqd_beam_bpl", "data/Movielens", "--model_path", "models/Movielens-SimplE-model-rank-50-epoch-40-1684629098.pt", "--dataset", "Movielens", "--candidates", "3", "--quantifier", "marginal_ui", "--cov_target", str(cov_target), "--cov_var", str(cov_var), "--cov_anchor", str(cov_anchor)]
                #command = ["python3", "-m", "kbc.cqd_beam_bpl", "data/Movielens", "--model_path", "models/Movielens-SimplE-model-rank-50-epoch-40-1684629098.pt", "--dataset", "Movielens", "--candidates", "10", "--quantifier", "marginal_i", "--cov_target", str(cov_target), "--cov_var", str(cov_var), "--cov_anchor", str(cov_anchor)]
                #command = ["python3", "-m", "kbc.cqd_beam_bpl", "data/LastFM", "--model_path", "models/LastFM-SimplE-model-rank-100-epoch-80-1686020763.pt", "--dataset", "LastFM", "--candidates", "10", "--quantifier", "marginal_i", "--cov_target", str(cov_target), "--cov_var", str(cov_var), "--cov_anchor", str(cov_anchor)]
                #command = ["python3", "-m", "kbc.cqd_beam_bpl", "data/Movielens_twohop", "--model_path", "models/Movielens_twohop-SimplE-model-rank-20-epoch-50-1687626870.pt", "--dataset", "Movielens_twohop", "--candidates", "3", "--quantifier", "marginal_ui", "--cov_target", str(cov_target), "--cov_var", str(cov_var), "--cov_anchor", str(cov_anchor)]
                #command = ["python3", "-m", "kbc.cqd_beam_bpl", "data/Movielens_twohop", "--model_path", "models/Movielens_twohop-SimplE-model-rank-20-epoch-50-1687626870.pt", "--dataset", "Movielens_twohop", "--candidates", "3", "--quantifier", "marginal_ui", "--cov_target", str(cov_target), "--cov_var", str(cov_var), "--cov_anchor", str(cov_anchor), "--chain_type", str(chain_type)]
                command = ["python3", "-m", "kbc.cqd_beam_bpl", "data/Movielens_twohop", "--model_path", "models/Movielens_twohop-SimplE-model-rank-50-epoch-30-1687217986.pt", "--dataset", "Movielens_twohop", "--candidates", "1", "--quantifier", "marginal_i", "--cov_target", str(cov_target), "--cov_var", str(cov_var), "--cov_anchor", str(cov_anchor), "--chain_type", str(chain_type), '--mode', 'valid']
                
                with open(os.path.join(experiment_name, output_file), "w") as f:
                #with open(output_file, "w") as f:
                    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    stdout, stderr = process.communicate()
                    f.write(stdout.decode())
                    if stderr:
                        f.write(stderr.decode())
                return_code = process.returncode
                print(f"Command returned with code {return_code} for cov_target={cov_target}, cov_var={cov_var}, and cov_anchor={cov_anchor}")
