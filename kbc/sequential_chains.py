import torch
import numpy as np
import os.path as osp
import argparse
import pickle
import json
import sys
from kbc.chain_dataset_bpl import ChaineDataset
from kbc.chain_dataset_bpl import Chain

from kbc.utils import QuerDAG
from kbc.utils import preload_env
from kbc.utils import QuerDAG
from kbc.utils import DynKBCSingleton

from kbc.metrics import evaluation
from kbc.evaluate_bpl import evaluate_existential
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
    description="Complex Query Decomposition - Beam"
    )
    parser.add_argument('path', help='Path to directory containing queries')
    parser.add_argument(
    '--dataset',
    help="The pickled Dataset name containing the chains"
    )
    parser.add_argument(
    '--mode', default='test'
    )
    args = parser.parse_args()
    print(args)