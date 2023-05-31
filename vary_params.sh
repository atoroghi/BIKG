#!/bin/bash
#SBATCH --job-name=zero
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --exclusive
#SBATCH --time=00:10:00
#SBATCH --account=def-ssanner
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --mail-user=<armin.toroghi@mail.utoronto.ca>
#SBATCH --mail-type=ALL

#wandb offline
#wandb online -num_run 
#wandb agent atoroghi/pre-critiquing/xydr8uhv
#python3 launch.py -epochs 1 -save_each 1 -kg no_kg
source ~/projects/def-ssanner/atoroghi/project/ENV2/bin/activate
cd ~/projects/def-ssanner/atoroghi/project/cqd
#python3 nested_cv_zero.py -tune_name soft_best -reg_type gauss -loss_type softplus -reduce_type sum -optim_type adagrad -sample_type split_rev -init_type uniform -kg kg 
#python3 nested_cv_zero.py -tune_name svd_neg_soft -reg_type gauss -loss_type softplus -reduce_type sum -optim_type adagrad -sample_type split_reg -init_type uniform -learning_rel freeze -kg no_kg 
#python3 inner_cv.py -tune_name gaussindirect -fold 0
for dataset in Movielens; do
	for rank in 50 100; do
		for reg in 0.001 0.01 0.1; do
			for batch_size in 300 500 1000; do
				sbatch --export=DATA=${dataset},RANK=${rank},REG=${reg},BATCH_SIZE=${batch_size} NewPA_job.sh
			done
		done
	done
done
