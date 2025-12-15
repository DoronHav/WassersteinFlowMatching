#!/usr/bin/env python3
"""
Riemannian Wasserstein Flow Matching on Scimilarity Point Clouds

This script demonstrates how to run RWFM for point-cloud generation on spherical manifolds
using the spherical MNIST dataset.
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
    print("Generate data")
    pc_train, pc_test = load_and_preprocess_mnist()
    print(f"Training samples: {len(pc_train)}, Test samples: {len(pc_test)}")

    max_files = 10  # For testing, limit number of files to load
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
    # Initialize model
    print("\nInitializing Flow Matching Model...")
    flow_model = PascientFM(
        point_clouds=df_train,
        config=rwfm_config
    )

    batch_size = 32
    training_steps = 200000
    decay_steps = 2000 
    flow_model.train(
        batch_size=batch_size,
        training_steps=training_steps,
        decay_steps=decay_steps,
        shape_sample= 100,
        use_wandb=True,
        wandb_config = {"project": "WFM"}
    )


    

    pc_train_processed = flow_model.point_clouds 

    data_module = PascientDataLoader(
        data_dir="/braid/cellm/zarrs/2025Q1_sample/",
        gene_order_path="/braid/scimilarity/gene_order.tsv",
        batch_size=16,
        num_workers=8,
        split_path="/braid/cellm/zarrs/2025Q1_sample/splits/split_v1.json",
        sample_metadata_path="/braid/cellm/zarrs/2025Q1_sample/harmonized_sample_list.csv",
        cell_metadata_paths=None,
        cell_metadata_columns=None,
        sample_metadata_columns=["tissue", "disease"],
        in_mem=True,
        return_sparse_tensor=True,
        sparse_collate=False,
        max_cells_per_sample=100,
        sampling_strategy="random",
        sampling_seed=42,
    )
    
    # Setup data
    data_module.setup()

    train_dl = data_module.train_dataloader()
    breakpoint()
    
    pc_train_loader = torch_loader(pc_train_processed, 
                                  weights=flow_model.weights,
                                  conditioning=None,
                                  global_bs=32,
                                  num_workers=8)


    
    flow_model.train_with_dl(
            pc_train_loader,
            training_steps=200000,
            verbose=8,
            learning_rate = 2e-4,
            decay_steps = 1000,
            saved_state=None,
            key=random.key(0),
            use_wandb=True,
            wandb_config={"project": "WFM"},
    )
    
    # Plot loss curve
    loss_smooth = np.convolve(np.log(flow_model.losses), np.ones(100) / 100, mode='valid')
    
    breakpoint()



if __name__ == "__main__":
    main()
