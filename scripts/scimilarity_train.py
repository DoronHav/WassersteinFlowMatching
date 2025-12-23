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
import pickle

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
    max_files = 50000  # For testing, limit number of files to load
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
    print(f"Loaded parquet files.")

    #with open(split_path, 'r') as f:
    #    splits = json.load(f)
    #df_train = full_df[full_df['dsid'].isin(splits['train'])]
    #df_val = full_df[full_df['dsid'].isin(splits['val'])]
    #df_test = full_df[full_df['dsid'].isin(splits['test'])]

    full_df["pc_index"] = full_df["dsid"].astype(str) + ":::" + full_df["sample"].astype(str)

    full_df.drop(columns=["dsid","sample"], inplace=True)
    
    # Map Tissue and Disease to embeddings
    print("Mapping tissue and disease to embeddings...")

    cond_df = full_df[["pc_index","tissue","disease"]].drop_duplicates()

    # cleaned disease map
    clean_disease_map = pd.read_csv('/data/debroue1/cellm/2025Q1/clean_disease_map.csv')
    clean_disease_map = dict(zip(clean_disease_map['disease'], clean_disease_map['HarmonizedLabel']))

    cond_df["disease"] = cond_df["disease"].map(clean_disease_map)
    cond_df.loc[cond_df["disease"]=="Other", "disease"] = np.nan

    # cleaned tissue map
    clean_tissue_map = pd.read_csv('/data/debroue1/cellm/2025Q1/clean_tissue_map.csv')
    clean_tissue_map = dict(zip(clean_tissue_map['tissue'], clean_tissue_map['tissue_clean']))

    cond_df["tissue"] = cond_df["tissue"].map(clean_tissue_map)
    
    #load embedding dicts
    with open('/data/debroue1/cellm/2025Q1/disease_embedding_dict.pkl', 'rb') as f:
        disease_dict = pickle.load(f)
    with open('/data/debroue1/cellm/2025Q1/tissue_embedding_dict.pkl', 'rb') as f:
        tissue_dict = pickle.load(f)

    tissue_embds = cond_df["tissue"].map(tissue_dict)
    disease_embds = cond_df["disease"].map(disease_dict)
    
    tissue_list = [x if isinstance(x, (np.ndarray, list)) else np.zeros(16) for x in tissue_embds]
    disease_list = [x if isinstance(x, (np.ndarray, list)) else np.zeros(11) for x in disease_embds]
 
    tissue_df = pd.DataFrame(tissue_list, columns = [f"tissue_embd_{i}" for i in range(16)], index = cond_df.index)
    disease_df = pd.DataFrame(disease_list, columns=[f"disease_embd_{i}" for i in range(11)], index = cond_df.index)
    cond_df = pd.concat([cond_df[["pc_index"]], tissue_df, disease_df], axis=1)
    # Initialize model
    print("\nInitializing Flow Matching Model...")
    flow_model = PascientFM(
        point_clouds=full_df,
        config=rwfm_config,
        conditioning = cond_df
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
