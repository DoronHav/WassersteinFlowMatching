#!/usr/bin/env python3
"""
Riemannian Wasserstein Flow Matching on Scimilarity Point Clouds

This script demonstrates how to run RWFM for point-cloud generation on the Scimilarity manifold.

usage: uv run python scimilarity_train.py

The script loads all point clouds in memory so large memory jobs are required.
The point clouds are batched on the fly to avoid additional memory overhead due to padding.
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import jax
import jax.numpy as jnp
from jax import random, lax
import numpy as np
import matplotlib.pyplot as plt
import skimage
from tqdm import tqdm
from scipy.spatial.distance import cdist
import multiprocessing
from itertools import product
import torch

import ott
from ott.solvers import linear
import ot

import json

from wassersteinflowmatching.riemannian_wasserstein import PascientFM

class rwfm_config:
    """Configuration for Riemannian Wasserstein Flow Matching"""
    geom: str = 'sphere'
    monge_map: str = 'wasserstein_eps'
    wasserstein_eps: float = 0.005
    wasserstein_lse: bool = True
    num_sinkhorn_iters: int = -1
    mini_batch_ot_mode: bool = True
    mini_batch_ot_solver: str = 'chamfer'
    minibatch_ot_eps: float = 0.01
    minibatch_ot_lse: bool = True
    noise_type: str = 'ambient_gaussian'
    scaling: str = 'None'
    factor: float = 1.0
    embedding_dim: int = 512
    num_layers: int = 6
    num_heads: int = 4
    dropout_rate: float = 0.1
    mlp_hidden_dim: int = 512
    cfg: bool = True
    p_cfg_null: float = 0.1
    w_cfg: float = 2.0
    normalized_condition: bool = False



class ArrayDataset(torch.utils.data.Dataset):
    def __init__(self, pcs, wts, cond=None):
        self.pcs = pcs   # numpy arrays
        self.wts = wts
        self.cond = cond
    def __len__(self): return self.pcs.shape[0]
    def __getitem__(self, i):
        x = torch.from_numpy(self.pcs[i])       # CPU tensors
        w = torch.from_numpy(self.wts[i])
        if self.cond is None:
            return {"pc": x, "w": w}
        c = torch.from_numpy(self.cond[i])
        return {"pc": x, "w": w, "cond": c}

def torch_loader(point_clouds, weights, conditioning, global_bs, num_workers=8):
    ds = ArrayDataset(point_clouds, weights, conditioning)
    return DataLoader(
        ds,
        batch_size=global_bs, shuffle=True,
        num_workers=num_workers, persistent_workers=True,
        prefetch_factor=4, pin_memory=False,
        drop_last=True
    )

import jax
import os
import glob
import pandas as pd

def main():
    """Main execution function."""

    # Load and preprocess data
    max_files = 5  # For testing, limit number of files to load
    data_dir = "/braid/cellm/2025Q1_scimilarity_embeddings/"
    split_path = "/braid/cellm/zarrs/2025Q1_sample/splits/split_v1.json"
    embedding_prefix = "emb"
    files = sorted(glob.glob(os.path.join(data_dir, "*.parquet")))
    df_list = []
    
    for i,f in tqdm(enumerate(files), desc="Loading parquet files", total=len(files)):
        try:
            df_list.append(pd.read_parquet(f.replace("/braid/","s3://prescient-braid-data/"), engine='pyarrow'))
        except Exception as e:
            breakpoint()
        if i+1 >= max_files:
            break
    full_df = pd.concat(df_list, ignore_index=True)
    del df_list

    with open(split_path, 'r') as f:
        splits = json.load(f)
    df_train = full_df[full_df['dsid'].isin(splits['train'])]
    df_val = full_df[full_df['dsid'].isin(splits['val'])]
    df_test = full_df[full_df['dsid'].isin(splits['test'])]

    df_train["pc_index"] = df_train["dsid"].astype(str) + ":::" + df_train["sample"].astype(str)

    df_train.drop(columns=["dsid","sample"], inplace=True)
    
    # Map Tissue and Disease to numerical values
    # We set -1 for unknown/nans
    
    df_train[["tissue", "disease"]] = df_train[["tissue", "disease"]].fillna("nan")
    tissue_dict = dict(zip(df_train["tissue"].unique(), range(df_train["tissue"].nunique())))
    tissue_dict["nan"] = -1
    disease_dict = dict(zip(df_train["disease"].unique(), range(df_train["disease"].nunique())))
    disease_dict["nan"] = -1

    df_train["tissue"] = df_train["tissue"].map(tissue_dict)
    df_train["disease"] = df_train["disease"].map(disease_dict)

    conditioning = df_train[["pc_index","tissue","disease"]].drop_duplicates()
    
    # Initialize model
    print("\nInitializing Flow Matching Model...")
    flow_model = PascientFM(
        point_clouds=df_train,
        config=rwfm_config,
        conditioning = conditioning
    )

    batch_size = 32
    training_steps = 200000
    decay_steps = 50000 
    flow_model.train(
        batch_size=batch_size,
        training_steps=training_steps,
        decay_steps=decay_steps,
        shape_sample= 100,
        use_wandb=True,
        wandb_config = {"project": "WFM"}
    )
    
    #save model
    flow_model.save("./scimilarity_flow_model.pkl")


if __name__ == "__main__":
    main()
