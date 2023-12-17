
This repository contains the implementation of our AAAI-24 paper **Bayesian Inference with Complex Knowledge Graph Evidence (BIKG)**.

In order to use the code, please follow these steps:

## 1- Install requirements
~~~
pip install -r requirements.txt
~~~
## 2- Download the Datasets
You can download the five datasets used in the paper from [here](https://drive.google.com/drive/folders/1pz6qYObdTdw4KprXZ0oE3OOG4TDrE8wW?usp=drive_link) and de-compress them.

## 3- Download Embedding Models or Train them
You can download the KGE models from [here](https://drive.google.com/drive/folders/1joInw77FnnbEy2qeQs1qg6kzqQ8hLJKQ?usp=sharing) and de-compress them. Alternatively, you can train your own models using the following code with your desired arguments.
~~~
python -m kbc.learn data/Movielens_twohop --rank 100 --max_epochs 300 --batch_size 128 --model SimplE --valid 1 --model_save_schedule 10
~~~

## 4- Sequential Knowledge Graph-based Query Answering (KGQA)
In order to run the KGQA experiments, please run the following command with your desired arguments.
~~~
python3 -m kbc.cqd_beam data/NELL --model_path models/NELL-DistMult-model-rank-300-epoch-300-1690457575.pt --dataset NELL --candidates 1 --reasoning_mode bayesian1 --mode test --seq yes --chain_type 1_2_seq
~~~
## 5- Critiquing with Complex Evidence
In order to perform critiquing with complex evidence experiments, please run the following files with your desired arguments. Please note that this experiment requires hyperparameter tuning.
~~~
python3 -m kbc.cqd_beam_bpl data/Movielens_twohop --model_path models/Movielens_twohop-SimplE-model-rank-50-epoch-30-1687217986.pt --dataset Movielens_twohop --candidates 3 --quantifier marginal_ui --cov_anchor 1e-2 --cov_var 1e-2 --cov_target 1e-2
~~~
You can also do the hyperparameter tuning by running the following code that performs grid search:
~~~
python kbc.CC_simulator
~~~

Thank you for your attention!
