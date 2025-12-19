from functools import partial # type: ignore
import types # type: ignore
 
import jax # type: ignore
import jax.numpy as jnp # type: ignore
import numpy as np  # type: ignore
import optax # type: ignore
from jax import jit, random# type: ignore
from tqdm import trange, tqdm # type: ignore
from flax.training import train_state # type: ignore
import pickle # type: ignore
import pandas as pd
import time

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

import wassersteinflowmatching.riemannian_wasserstein.utils_OT as utils_OT # type: ignore
import wassersteinflowmatching.riemannian_wasserstein.utils_Geom as utils_Geom # type: ignore  # noqa: F401
import wassersteinflowmatching.riemannian_wasserstein.utils_Noise as utils_Noise # type: ignore
from wassersteinflowmatching.riemannian_wasserstein._utils_Transformer import AttentionNN # type: ignore
from wassersteinflowmatching.riemannian_wasserstein.DefaultConfig import DefaultConfig # type: ignore
from wassersteinflowmatching.riemannian_wasserstein._utils_Processing import pad_pointclouds # type: ignore

from flax import jax_utils
from torch.utils.data import DataLoader
from typing import Union

from jax import lax

from wassersteinflowmatching.riemannian_wasserstein import RiemannianWassersteinFlowMatching


class PascientFM(RiemannianWassersteinFlowMatching):
    def __init__(
        self,
        point_clouds:pd.DataFrame,
        conditioning = None,
        config = DefaultConfig,
        **kwargs,
    ): 
    
        print("Initializing WassersteinFlowMatching")

        self.config = config

        for key, value in kwargs.items():
            setattr(self.config, key, value)
        
        self.geom = self.config.geom
        self.scaling = self.config.scaling
        self.factor = self.config.factor


        self.monge_map = self.config.monge_map
        self.num_sinkhorn_iters = self.config.num_sinkhorn_iters

        self.geom_utils = getattr(utils_Geom, self.geom)()
        
        print(f'Using {self.geom} geometry')

        self.interpolant_vmap = jax.vmap(jax.vmap(self.geom_utils.interpolant, in_axes=(0, 0, None), out_axes=0), in_axes=(0, 0, 0), out_axes=0)
        self.interpolant_velocity_vmap = jax.vmap(jax.vmap(self.geom_utils.velocity, in_axes=(0, 0, None), out_axes=0), in_axes=(0, 0, 0), out_axes=0)
        self.exponential_map_vmap = jax.vmap(jax.vmap(self.geom_utils.exponential_map, in_axes=(0, 0, None), out_axes=0), in_axes=(0, 0, None), out_axes=0)
        self.loss_func_vmap = jax.vmap(jax.vmap(self.geom_utils.tangent_norm, in_axes=(0, 0, 0), out_axes=0), in_axes=(0, 0, 0), out_axes=0)
        self.project_to_geometry = self.geom_utils.project_to_geometry

        embedding_prefix = "emb"
        emb_cols = [c for c in point_clouds.columns if c.startswith(embedding_prefix)]
        meta_cols = [c for c in point_clouds.columns if c not in emb_cols]
        point_clouds_ = point_clouds[emb_cols].values.astype(np.float32)
        metadata = point_clouds[meta_cols].reset_index(drop=True)

        pc_length_dict = metadata.groupby("pc_index").size().to_dict()
        pc_start_dict = metadata.reset_index().groupby("pc_index")["index"].min().to_dict()
        
        self.pcid_dict = {k:i for i,k in enumerate(pc_length_dict.keys())}
        #start and end index of each point cloud.
        self.pc_idx_dict_ = {k:(pc_start_dict[k], pc_start_dict[k]+pc_length_dict[k]) for k in pc_length_dict.keys()}
        self.pc_idx_dict = {self.pcid_dict[k]:v for k,v in self.pc_idx_dict_.items()}

        self.point_clouds = self.project_to_geometry(point_clouds_)
        self.weights = np.ones(self.point_clouds.shape[0])
        for k,v in self.pc_idx_dict.items():
            self.weights[v[0]:v[1]] /= (v[1]-v[0])

        sampled_idx = np.random.choice(list(self.pc_idx_dict.keys()), size=min(100, len(self.pc_idx_dict)), replace=False)
        sampled_pcs = [self.point_clouds[self.pc_idx_dict[s][0]:self.pc_idx_dict[s][1]] for s in sampled_idx]
        sampled_weights = [self.weights[self.pc_idx_dict[s][0]:self.pc_idx_dict[s][1]] for s in sampled_idx]
        

        self.sampled_point_clouds, self.sampled_weights = pad_pointclouds(
            sampled_pcs, 
            sampled_weights
        )
        self.space_dim = self.sampled_point_clouds.shape[-1]

        self.noise_config = types.SimpleNamespace()
        self.noise_type = self.config.noise_type
        
        
        self.noise_geom = self.config.noise_geom
        if(self.noise_geom != self.geom):
            print(f"Using {self.noise_geom} geometry for noise instead of {self.geom}")
            self.noise_geom = self.config.noise_geom
            self.noise_proj_to_geometry = getattr(utils_Geom, self.noise_geom)().project_to_geometry 
        else:
            print(f"Using {self.noise_geom} geometry for noise")
            self.noise_proj_to_geometry = self.project_to_geometry  
        # Get noise functions from the factory
        self.noise_func, param_estimator = utils_Noise.get_noise_functions(self.noise_type, self.noise_proj_to_geometry
        )

        self.matched_noise = False 

        # Estimate parameters if an estimator is available
        if param_estimator:
            print("Estimating parameters for noise generation...")
            params = param_estimator(self.sampled_point_clouds, self.sampled_weights)
            for key, value in params.items():
                setattr(self.noise_config, key, value)

        print(f"Using {self.noise_type} noise for {self.geom} geometry.")
        if self.noise_config.__dict__:
            print("Noise parameters:")
            for key, value in self.noise_config.__dict__.items():
                print(f"  {key}: {value}")

        if self.num_sinkhorn_iters == -1:
            print("Finding optimal number of Sinkhorn iterations...")
            key = random.key(0)
            
            noise_samples = self.noise_func(size=self.sampled_point_clouds.shape, 
                                            noise_config=self.noise_config,
                                            key=key)
            if len(noise_samples) == 2:
                noise_samples, noise_weights = noise_samples
            else:
                noise_weights = self.sampled_weights
            
            noise_samples = self.project_to_geometry(noise_samples)
            
            pc_list = [np.array(pc) for pc in self.sampled_point_clouds]
            w_list = [np.array(w) for w in self.sampled_weights]
            noise_list = [np.array(n) for n in noise_samples]
            nw_list = [np.array(nw) for nw in noise_weights]

            self.num_sinkhorn_iters = utils_OT.auto_find_num_iter(
                point_clouds=pc_list,
                weights=w_list,
                eps=self.config.wasserstein_eps,
                lse_mode=self.config.wasserstein_lse,
                distance_matrix_func=self.geom_utils.distance_matrix,
                noise_point_clouds=noise_list,
                noise_weights=nw_list
            )
            self.config.num_sinkhorn_iters = self.num_sinkhorn_iters
            print(f"Auto-selected {self.num_sinkhorn_iters} Sinkhorn iterations.")

        print(f"Using {self.monge_map} map with {self.num_sinkhorn_iters} iterations and {self.config.wasserstein_eps} epsilon")

        self.transport_plan_jit = jax.vmap(partial(utils_OT.transport_plan, 
                                            distance_matrix_func =  self.geom_utils.distance_matrix,
                                            eps = self.config.wasserstein_eps, 
                                            lse_mode = self.config.wasserstein_lse, 
                                            num_iteration = self.config.num_sinkhorn_iters),
                                            (0, 0), 0)
        
        if(self.monge_map == 'rounded_matching'):
            self.sample_map_jit = jax.vmap(lambda P, pc_y: utils_OT.get_assignments_rounding(P, pc_y)[0], (0, 0), 0)
        elif(self.monge_map == 'sample'):
            self.sample_map_jit = jax.vmap(lambda P, pc_y, key: utils_OT.get_assignments_sampling(P, key, pc_y)[0], (0, 0, 0), 0)
        elif(self.monge_map == 'entropic'):
            # Entropic assignment uses geometry-specific weighted mean
            self.sample_map_jit = jax.vmap(lambda P, pc_y: partial(utils_OT.get_assignments_entropic, weighted_mean_func=self.geom_utils.weighted_mean)(P, pc_y)[0], (0, 0), 0)
        else:
            # Default to rounded matching
            self.sample_map_jit = jax.vmap(lambda P, pc_y: utils_OT.get_assignments_rounding(P, pc_y)[0], (0, 0), 0)
        

        self.mini_batch_ot_mode = self.config.mini_batch_ot_mode


        if(conditioning is not None):
            self.conditioning = jnp.array(conditioning)
            if self.conditioning.ndim == 1:
                self.conditioning = self.conditioning[:, None]
            self.conditioning_dim = self.conditioning.shape[-1]
            self.config.conditioning_dim = self.conditioning_dim
            self.mini_batch_ot_mode = False

            self.cfg = self.config.cfg
            if self.cfg:
                self.p_cfg_null = self.config.p_cfg_null
                self.w_cfg = self.config.w_cfg
                print(f"Using CFG with null conditioning probability {self.p_cfg_null}, and weight {self.w_cfg}")

        else:
            self.conditioning = None
            self.conditioning_dim = -1
            self.cfg = False  # No CFG when there's no conditioning

        if(self.mini_batch_ot_mode):
            self.mini_batch_ot_solver = self.config.mini_batch_ot_solver
            if(self.mini_batch_ot_solver == 'entropic'):
                print("Entropic Mini-Batch")
                self.ot_mat_jit = jax.vmap(partial(utils_OT.entropic_ot_distance, 
                                                   eps = self.config.minibatch_ot_eps,
                                                   lse_mode = self.config.minibatch_ot_lse), (0, 0), 0)
            elif(self.mini_batch_ot_solver == 'chamfer'):
                print("Chamfer Mini-Batch")
                self.ot_mat_jit = jax.vmap(partial(utils_OT.chamfer_distance, 
                                            distance_matrix_func = self.geom_utils.distance_matrix), (0, 0), 0)
            elif(self.mini_batch_ot_solver == 'euclidean'):
                print("Euclidean Mini-Batch")
                self.ot_mat_jit = jax.vmap(utils_OT.euclidean_distance, (0, 0), 0)
            else:
                print("Frechet Mini-Batch")
                self.ot_mat_jit = jax.vmap(utils_OT.frechet_distance, (0, 0), 0)

            self.mini_batch_ot_num_iter = self.config.mini_batch_ot_num_iter
            if self.mini_batch_ot_num_iter == -1:
                print("Finding optimal number of Sinkhorn iterations for Mini-Batch OT...")
                
                key = random.key(0)
                noise_samples = self.noise_func(size=self.sampled_point_clouds.shape, 
                                                noise_config=self.noise_config,
                                                key=key)
                if len(noise_samples) == 2:
                    noise_samples, noise_weights = noise_samples
                else:
                    noise_weights = self.sampled_weights
                
                noise_samples = self.project_to_geometry(noise_samples)

                self.mini_batch_ot_num_iter = utils_OT.auto_find_num_iter_minibatch(
                    point_clouds=self.sampled_point_clouds,
                    weights=self.sampled_weights,
                    noise_point_clouds=noise_samples,
                    noise_weights=noise_weights,
                    ot_mat_jit=self.ot_mat_jit,
                    eps = self.config.minibatch_ot_eps,
                    lse_mode = self.config.minibatch_ot_lse,
                    sample_size=2048
                )
                print(f"Auto-selected {self.mini_batch_ot_num_iter} Sinkhorn iterations for Mini-Batch OT.")
        
        self.FlowMatchingModel = AttentionNN(config = self.config)


    def train(
        self,
        training_steps=32000,
        batch_size=16,
        verbose=8,
        learning_rate = 2e-4, 
        decay_steps = 1000,
        shape_sample = None,
        source_sample = None,
        saved_state = None,
        key=random.key(0),
        use_wandb=False,
        wandb_config=None,
    ):
        """
        Set up optimization parameters and train the model


        :param training_steps: (int) number of gradient descent steps to train (default 32000)
        :param batch_size: (int) size of train-set point clouds sampled for each training step  (default 16)
        :param verbose: (int) amount of steps between each loss print statement (default 8)
        :param learning_rate: (float) learning rate for ADAM optimizer with exponential decay (default 2e-4)
        :param decay_steps: (int) learning rate decay steps (default 1000)
        :param shape_sample: (int) number of points to sample from each point cloud (default None)
        :param source_sample: (int) number of points to sample from source (default None)
        :param saved_state: (TrainState) saved training state to resume from (default None)
        :param key: (jax.random.key) random seed (default jax.random.key(0))
        :param use_wandb: (bool) whether to log to Weights & Biases (default False)
        :param wandb_config: (dict) W&B config with keys: project, entity, name, config (default None)

        :return: nothing
        """
        
        if use_wandb and not WANDB_AVAILABLE:
            print("Warning: wandb not installed. Install with: pip install wandb")
            use_wandb = False
        
        if use_wandb and wandb_config is not None:
            if not wandb.run:
                wandb.init(**wandb_config)

        subkey, key = random.split(key)

        if saved_state is None:
            self.state = self.create_train_state(
                model=self.FlowMatchingModel,
                learning_rate = learning_rate, 
                decay_steps = decay_steps, 
                key=subkey
            )
        else:
            self.state = saved_state
            print(f"Resuming training from step {int(self.state.step)}")


        if(shape_sample is not None):
            print(f'Sampling {shape_sample} points from each point cloud')
            sample_points = jax.vmap(self.sample_single_batch, in_axes=(0, 0, 0, None))

        tq = trange(training_steps - self.state.step, leave=True, desc="")
        self.losses = []
        try:
            for training_step in tq:

                subkey, key = random.split(key, 2)
                batch_ind = random.choice(
                    key=subkey,
                    a = len(self.pc_idx_dict),
                    shape=[batch_size])

                t = time.time()

                pcs = [self.point_clouds[self.pc_idx_dict[int(bx)][0]:self.pc_idx_dict[int(bx)][1]] for bx in batch_ind]
                wcs = [self.weights[self.pc_idx_dict[int(bx)][0]:self.pc_idx_dict[int(bx)][1]] for bx in batch_ind]
                print("Data loading time:", time.time() - t)
                
                t = time.time()
                point_clouds_batch, weights_batch = pad_pointclouds(pcs, wcs)
                print("Padding time:", time.time() - t)
         

                if(self.matched_noise):
                    noise_samples, noise_weights = self.noise_point_clouds[batch_ind], self.noise_weights[batch_ind]
                    if(source_sample is not None):
                        keys = jax.random.split(subkey, batch_size)
                        noise_samples, noise_weights = sample_points(noise_samples, noise_weights, keys, source_sample)
                else:
                    noise_samples, noise_weights = None, None

                
                if(shape_sample is not None):
                    keys = jax.random.split(subkey, batch_size)
                    point_clouds_batch, weights_batch = sample_points(point_clouds_batch, weights_batch, keys, shape_sample)
                    
                if(self.conditioning is not None):
                    conditioning_batch = self.conditioning[batch_ind]
                    if self.cfg:
                        subkey, key = random.split(key)
                        is_null_conditioning = random.bernoulli(subkey, p=self.p_cfg_null, shape=(batch_size,))
                    else:
                        is_null_conditioning = None
                    
                else:
                    conditioning_batch = None
                    is_null_conditioning = None

                subkey, key = random.split(key, 2)
    
                self.state, loss = self.train_step(self.state, point_clouds_batch, weights_batch, conditioning_batch, noise_samples, noise_weights, is_null_conditioning, key = subkey)

                self.params = self.state.params
                loss_value = float(loss)
                self.losses.append(loss_value)
                
                if use_wandb:
                    wandb.log({
                        "train/loss": loss_value,
                        "train/step": training_step + self.state.step,
                        "train/logloss": np.log(loss_value + 1e-10),
                    })


                if(training_step % verbose == 0):
                    tq.set_description(": {:.3e}".format(loss_value))
        except KeyboardInterrupt:
            print("\nTraining interrupted by user. The model state has been saved.")

        # Ensure state is unreplicated when returning, for compatibility with other methods
        self.state = jax_utils.unreplicate(self.state)
        self.params = self.state.params