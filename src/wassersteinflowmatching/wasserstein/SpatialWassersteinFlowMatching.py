from functools import partial
import types
from typing import Optional

import jax # type: ignore
import jax.numpy as jnp # type: ignore
import numpy as np  # type: ignore
import optax # type: ignore
from jax import jit, random, lax# type: ignore
from tqdm import trange, tqdm # type: ignore
from flax.training import train_state # type: ignore
import pickle # type: ignore

# Imports needed for the new functionality
import pandas as pd # type: ignore
import anndata # type: ignore
import sklearn.neighbors # type: ignore
import scipy.sparse # type: ignore

import wassersteinflowmatching.wasserstein.utils_OT as utils_OT # type: ignore
import wassersteinflowmatching.wasserstein.utils_Noise as utils_Noise # type: ignore
from wassersteinflowmatching.wasserstein._utils_Transformer import AttentionNN # type: ignore
from wassersteinflowmatching.wasserstein.DefaultConfig import DefaultConfig, SpatialDefaultConfig # type: ignore
from wassersteinflowmatching.wasserstein._utils_Processing import pad_pointclouds # type: ignore



class SpatialWassersteinFlowMatching:
    """
    Initializes a spatial Wasserstein Flow Matching model.

    This class learns a generative model of local cellular "niches" (a cell and its
    spatial neighbors) from spatial transcriptomics data.

    :param adata: (anndata.AnnData) The annotated data object, must contain spatial coordinates.
    :param k_neighbours: (int) Number of nearest neighbors to define a niche.
    :param conditioning_obs: (list[str], optional) List of keys in `adata.obs` for conditioning.
    :param conditioning_obsm: (list[str], optional) List of keys in `adata.obsm` for conditioning.
    :param config: (dataclass, optional) Configuration object. Defaults to `SpatialDefaultConfig`.
    :param **kwargs: Additional configuration parameters to override defaults.
    """

    def __init__(
        self,
        adata,
        k_neighbours = 8,
        conditioning_obs: Optional[list] = None,
        conditioning_obsm: Optional[list] = None,
        config = SpatialDefaultConfig,
        **kwargs,
    ):
        print("Initializing Wasserstein Flow Matching")

        if not isinstance(adata, anndata.AnnData):
            raise TypeError("Input 'adata' must be an anndata.AnnData object.")
        if 'spatial' not in adata.obsm:
             raise ValueError("Input 'adata' must have spatial coordinates in .obsm['spatial'].")

        # --- Configuration Setup ---
        if config is None:
            config = SpatialDefaultConfig()
        if kwargs:
            config = config.replace(**kwargs)
        self.config = config
        self.adata = adata

        # --- Data Pre-processing and Caching ---
        print("Pre-computing neighbor indices and caching expression data...")
        self.exp_data_train = self._get_exp_data(self.adata, self.config.rep)
        self.niche_indices_train = self._get_niche_indices(
            self.adata, k_neighbours, self.config.spatial_key, self.config.batch_key
        )
        
        # NOTE: Define space_dim, inp_dim, and max_niche_size correctly.
        self.space_dim = self.exp_data_train.shape[1]
        self.inp_dim = self.space_dim # Use a consistent name for input feature dimension.
        self.max_niche_size = max(len(n) for n in self.niche_indices_train) if self.niche_indices_train else 0
        print(f"INFO: Input feature dimension is {self.space_dim}.")
        print(f"INFO: Maximum niche size is {self.max_niche_size}.")

        self.noise_config = types.SimpleNamespace()
        self.noise_type = self.config.noise_type

        # make sure noise_type is normal if not, return an error

        if self.noise_type not in ['normal', 'uniform']:
            raise ValueError(f"Unsupported noise type: {self.noise_type}. Supported types are 'normal', 'uniform'")

        self.noise_config.minval = self.exp_data_train.min()
        self.noise_config.maxval = self.exp_data_train.max()
        self.noise_func = getattr(utils_Noise, self.noise_type)
        
        # --- Optimal Transport (Monge Map) Setup ---
        self.monge_map = self.config.monge_map
        self.num_sinkhorn_iters = self.config.num_sinkhorn_iters

        if self.num_sinkhorn_iters == -1:
            print("NOTE: auto_find_num_iter is not supported in this version. Using a default value.")
            self.num_sinkhorn_iters = 100 
            print("Setting num_sinkhorn_iter to default value:", self.num_sinkhorn_iters)
        else:
            print("Using num_sinkhorn_iter =", self.num_sinkhorn_iters) 

        # NOTE: Partial function application for different OT methods
        if self.monge_map == 'entropic':
            print(f"Using entropic map with {self.num_sinkhorn_iters} iterations and {self.config.wasserstein_eps} epsilon")
            self.transport_plan_jit = jax.vmap(partial(utils_OT.transport_plan_entropic, 
                                                       eps = self.config.wasserstein_eps, 
                                                       lse_mode = self.config.wasserstein_lse, 
                                                       num_iteration = self.num_sinkhorn_iters),
                                                       (0, 0), 0)
        elif self.monge_map == 'rounded_matching':
            print(f"Using rounded_matching map with {self.num_sinkhorn_iters} iterations and {self.config.wasserstein_eps} epsilon")
            self.transport_plan_jit = jax.vmap(partial(utils_OT.transport_plan_rounded, 
                                            eps = self.config.wasserstein_eps, 
                                            lse_mode = self.config.wasserstein_lse, 
                                            num_iteration = self.num_sinkhorn_iters),
                                            (0, 0), 0)
        elif self.monge_map == 'sample':
            print(f"Using sampled map with {self.num_sinkhorn_iters} iterations and {self.config.wasserstein_eps} epsilon")
            self.transport_plan_jit = jax.vmap(partial(utils_OT.transport_plan_sample, 
                                eps = self.config.wasserstein_eps, 
                                lse_mode = self.config.wasserstein_lse, 
                                num_iteration = self.num_sinkhorn_iters),
                                (0, 0), 0)
            self.sample_map_jit = jax.vmap(utils_OT.sample_ot_matrix, (0, 0, 0, 0), 0)
        elif self.monge_map == 'euclidean':
            print("Using euclidean Monge map")
            self.transport_plan_jit = jax.vmap(utils_OT.transport_plan_euclidean, (0, 0), 0)
        else:
            print(f"Using argmax map with {self.num_sinkhorn_iters} iterations and {self.config.wasserstein_eps} epsilon")
            self.transport_plan_jit = jax.vmap(partial(utils_OT.transport_plan_argmax, 
                    eps = self.config.wasserstein_eps, 
                    lse_mode = self.config.wasserstein_lse, 
                    num_iteration = self.num_sinkhorn_iters),
                    (0, 0), 0)
            
        # --- Conditioning and Guidance Setup ---
        # NOTE: This logic correctly handles conditional vs. unconditional model setup.
        self.mini_batch_ot_mode = self.config.mini_batch_ot_mode
        if conditioning_obs or conditioning_obsm:
            self.conditioning_vectors = self._get_conditioning_data(self.adata, 
                                                                    conditioning_obs = conditioning_obs, 
                                                                    conditioning_obsm = conditioning_obsm)
            self.guidance_gamma = self.config.guidance_gamma
            self.p_uncond = self.config.p_uncond if self.guidance_gamma > 1 else 0.0
            self.mini_batch_ot_mode = False
            
            if self.guidance_gamma > 1:
                print(f"Using conditioning with guidance (gamma={self.guidance_gamma}, p_uncond={self.p_uncond})")
            else:
                print("Using conditioning without guidance.")
        else:
            self.conditioning_vectors = None
            self.guidance_gamma = 0
            self.p_uncond = 0.0
            print("No conditioning provided, training an unconditional model.")

        # --- Mini-batch OT Setup ---
        if self.mini_batch_ot_mode:
            self.mini_batch_ot_solver = self.config.mini_batch_ot_solver
            if self.mini_batch_ot_solver == 'entropic':
                print("Entropic Mini-Batch")
                self.ot_mat_jit = jax.vmap(partial(utils_OT.entropic_ot_distance, 
                                                   eps = self.config.minibatch_ot_eps,
                                                   lse_mode = self.config.minibatch_ot_lse), (0, 0), 0)
            elif self.mini_batch_ot_solver == 'chamfer':
                print("Chamfer Mini-Batch")
                self.ot_mat_jit = jax.vmap(utils_OT.chamfer_distance, (0, 0), 0)
            elif self.mini_batch_ot_solver == 'euclidean':
                print("Euclidean Mini-Batch")
                self.ot_mat_jit = jax.vmap(utils_OT.euclidean_distance, (0, 0), 0)
            else:
                print("Frechet Mini-Batch")
                self.ot_mat_jit = jax.vmap(utils_OT.frechet_distance, (0, 0), 0)
        
        # --- Model Initialization ---
        self.FlowMatchingModel = AttentionNN(config = self.config)
        self.state = None # Initialize state to None before training.

    @staticmethod
    def _get_exp_data(adata, rep):
        """Extracts the expression matrix as a dense numpy array for fast indexing."""
        if rep is None:
            return adata.X.toarray().astype('float32') if scipy.sparse.issparse(adata.X) else np.asarray(adata.X).astype('float32')
        else:
            return np.asarray(adata.obsm[rep]).astype('float32')

    @staticmethod
    def _get_conditioning_data(adata: anndata.AnnData, 
                               conditioning_obs: list[str] = None, 
                               conditioning_obsm: list[str] = None) -> np.ndarray:
        """
        Extracts and combines conditioning data from an AnnData object.
        """
        conditioning_vectors = []
        conditioning_obs = conditioning_obs or []
        conditioning_obsm = conditioning_obsm or []

        for key in conditioning_obs:
            if key not in adata.obs:
                raise KeyError(f"Conditioning key '{key}' not found in adata.obs.")
            obs_data = adata.obs[key]
            if obs_data.dtype.name in ['category', 'object']:
                print(f"INFO: One-hot encoding for .obs key: '{key}'")
                one_hot_encoded = pd.get_dummies(obs_data, prefix=key, dtype=float)
                conditioning_vectors.append(one_hot_encoded.values)
            elif np.issubdtype(obs_data.dtype, np.number):
                print(f"INFO: Using numerical data from .obs key: '{key}'")
                conditioning_vectors.append(obs_data.values.reshape(-1, 1))
            else:
                raise TypeError(f"Unsupported dtype '{obs_data.dtype}' for .obs key '{key}'.")

        for key in conditioning_obsm:
            if key not in adata.obsm:
                raise KeyError(f"Conditioning key '{key}' not found in adata.obsm.")
            obsm_data = adata.obsm[key]
            if scipy.sparse.issparse(obsm_data):
                obsm_data = obsm_data.toarray()
            if obsm_data.ndim != 2:
                raise ValueError(f".obsm['{key}'] must be a 2D array.")
            print(f"INFO: Using embedding from .obsm key: '{key}'")
            conditioning_vectors.append(obsm_data)

        if not conditioning_vectors:
            return np.empty((adata.n_obs, 0))

        final_conditioning_data = np.concatenate(conditioning_vectors, axis=1)
        print(f"INFO: Final conditioning matrix shape: {final_conditioning_data.shape}")
        return final_conditioning_data
        

    @staticmethod
    def _get_niche_indices(spatial_data, k, spatial_key, batch_key):
        """
        Computes the k-NN graph and returns a list of neighbor index arrays for each cell.
        :meta private:
        """
        if batch_key == -1 or batch_key not in spatial_data.obs.columns:
            kNNGraph = sklearn.neighbors.kneighbors_graph(spatial_data.obsm[spatial_key], n_neighbors=k, mode='connectivity', n_jobs=-1).tocsr()
            return np.split(kNNGraph.indices, kNNGraph.indptr[1:-1])
        else:
            batch = spatial_data.obs[batch_key]
            niche_indices_list = [np.array([], dtype=int)] * spatial_data.shape[0]
            for val in np.unique(batch):
                val_mask = (batch == val)
                original_indices = np.where(val_mask)[0]
                data_batch = spatial_data[val_mask]
                batch_k = min(k, data_batch.shape[0] - 1)
                if batch_k < 1: continue
                
                batch_knn = sklearn.neighbors.kneighbors_graph(data_batch.obsm[spatial_key], n_neighbors=batch_k, mode='connectivity', n_jobs=-1).tocsr()
                split_indices = np.split(batch_knn.indices, batch_knn.indptr[1:-1])
                for i, local_indices in enumerate(split_indices):
                    niche_indices_list[original_indices[i]] = original_indices[local_indices]
            return niche_indices_list

    def _assemble_and_pad_batch(self, cell_indices, is_test=False):
        """
        Assembles point clouds for a batch of cells and pads them to a uniform size.
        :meta private:
        """
        if is_test:
            raise NotImplementedError("Test mode is not supported as test data sources are not initialized.")

        exp_data = self.exp_data_train
        niche_indices_source = self.niche_indices_train
        
        batch_size = len(cell_indices)
        unpadded_indices = [niche_indices_source[i] for i in cell_indices]
        
        padded_pcs = np.zeros((batch_size, self.max_niche_size, self.inp_dim), dtype=np.float32)
        padded_weights = np.zeros((batch_size, self.max_niche_size), dtype=np.float32)

        for i, neighbor_idx_array in enumerate(unpadded_indices):
            num_neighbors = len(neighbor_idx_array)
            if num_neighbors == 0:
                continue
            
            pc = exp_data[neighbor_idx_array]
            
            padded_pcs[i, :num_neighbors, :] = pc
            padded_weights[i, :num_neighbors] = 1.0 / num_neighbors

        return jnp.asarray(padded_pcs), jnp.asarray(padded_weights)
    
    def minibatch_ot(self, point_clouds, point_cloud_weights, noise, noise_weights):

        """
        :meta private:
        """
            
        matrix_ind = jnp.array(jnp.meshgrid(jnp.arange(point_clouds.shape[0]), jnp.arange(noise.shape[0]))).T.reshape(-1, 2)


        # compute pairwise ot between point clouds and noise:
        
        if(self.mini_batch_ot_solver == 'frechet'):
            mean_x, cov_x = utils_OT.weighted_mean_and_covariance(point_clouds, point_cloud_weights)
            mean_y, cov_y = utils_OT.weighted_mean_and_covariance(noise, noise_weights)
            ot_matrix = self.ot_mat_jit([mean_x[matrix_ind[:, 0]], cov_x[matrix_ind[:, 0]]], 
                                        [mean_y[matrix_ind[:, 1]], cov_y[matrix_ind[:, 1]]]).reshape(point_clouds.shape[0], noise.shape[0])
        else:
            ot_matrix = self.ot_mat_jit([point_clouds[matrix_ind[:, 0]], point_cloud_weights[matrix_ind[:, 0]]],
                                        [noise[matrix_ind[:, 1]], noise_weights[matrix_ind[:, 1]]]).reshape(point_clouds.shape[0], noise.shape[0])

        noise_ind = utils_OT.ot_mat_from_distance(ot_matrix, 0.002, True)
        return(noise_ind)


    def create_train_state(self, model, learning_rate, decay_steps, key=random.key(0)):
        """Creates the initial training state for the model."""
        subkey, key = random.split(key)
        
        dummy_noise = self.noise_func(
            size=[10, min(self.max_niche_size, 32), self.space_dim],
            noise_config=self.noise_config,
            key=subkey
        )

        subkey, key = random.split(key)
        
        if self.conditioning_vectors is not None:
            dummy_labels = self.conditioning_vectors[np.random.choice(self.conditioning_vectors.shape[0], dummy_noise.shape[0])]
            params = model.init(
                rngs={"params": subkey},
                point_cloud=dummy_noise,
                t=jnp.ones((dummy_noise.shape[0])),
                masks=jnp.ones((dummy_noise.shape[0], dummy_noise.shape[1])),
                labels=dummy_labels,
                deterministic=True
            )['params']
        else:
            params = model.init(
                rngs={"params": subkey},
                point_cloud=dummy_noise,
                t=jnp.ones((dummy_noise.shape[0])),
                masks=jnp.ones((dummy_noise.shape[0], dummy_noise.shape[1])),
                deterministic=True
            )['params']
            
        lr_sched = optax.exponential_decay(learning_rate, decay_steps, 0.998, staircase=False)
        tx = optax.adam(lr_sched)

        return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)
    
    @partial(jit, static_argnums=(0,))
    def train_step(self, state, point_clouds_batch, weights_batch, conditioning_vectors_batch=None, key=random.key(0)):
        """JIT-compiled training step."""
        subkey_t, subkey_noise, key = random.split(key, 3)
        noise_samples = self.noise_func(size=point_clouds_batch.shape, noise_config=self.noise_config, key=subkey_noise)
        noise_weights = weights_batch

        if self.mini_batch_ot_mode:
            noise_ind = self.minibatch_ot(point_clouds_batch, weights_batch, noise_samples, noise_weights)
            noise_samples = noise_samples[noise_ind]
            if(self.monge_map == 'entropic'):
                noise_weights = noise_weights[noise_ind]

        interpolates_time = random.uniform(subkey_t, (point_clouds_batch.shape[0],), minval=0.0, maxval=1.0)
        optimal_flow = self.transport_plan_jit([noise_samples, noise_weights], [point_clouds_batch, weights_batch])[0]

        if self.monge_map == 'sample':
            optimal_flow = self.sample_map_jit(noise_samples, point_clouds_batch, optimal_flow, random.split(key, point_clouds_batch.shape[0]))

        point_cloud_interpolates = noise_samples + (1 - interpolates_time[:, None, None]) * optimal_flow

        if(conditioning_vectors_batch is not None):
            key_uncond, key = random.split(key)
            is_uncond = random.uniform(key_uncond, (point_clouds_batch.shape[0],)) < self.p_uncond
            is_uncond = is_uncond[:, None] if conditioning_vectors_batch.ndim == 2 else is_uncond
            conditioning_vectors_batch = jnp.where(is_uncond, jnp.zeros_like(conditioning_vectors_batch), conditioning_vectors_batch)
        
        optimal_flow = -optimal_flow
        
        subkey, key = random.split(key)
        def loss_fn(params):
            predicted_flow = state.apply_fn({"params": params}, point_cloud=point_cloud_interpolates, t=interpolates_time, masks=noise_weights > 0, labels=conditioning_vectors_batch, deterministic=False, dropout_rng=subkey)
            error = jnp.square(predicted_flow - optimal_flow) * noise_weights[:, :, None]
            loss = jnp.mean(jnp.sum(error, axis=1))
            return loss

        loss, grads = jax.value_and_grad(loss_fn)(state.params)
        state = state.apply_gradients(grads=grads)
        return state, loss

    def train(
        self,
        training_steps=32000,
        batch_size=16,
        verbose=8,
        learning_rate = 3e-4, 
        decay_steps = 1000,
        shape_sample: Optional[int] = None,
        saved_state: Optional[train_state.TrainState] = None,
        key=random.key(0),
    ):
        """Sets up and runs the training loop for the model."""
        subkey, key = random.split(key)

        if saved_state is None:
            print("Creating a new training state.")
            self.state = self.create_train_state(model=self.FlowMatchingModel, learning_rate=learning_rate, decay_steps=decay_steps, key=subkey)
        else:
            print(f"Resuming training from provided state at step {int(saved_state.step)}.")
            self.state = saved_state

        start_step = int(self.state.step)
        
        if shape_sample is not None:
            print(f'Sampling {shape_sample} points from each point cloud per step.')
            sample_points = jax.vmap(self.sample_single_batch, in_axes=(0, 0, 0, None))
            
        tq = trange(start_step, training_steps, leave=True, desc="")
        self.losses = []
        for step in tq:
            subkey, key = random.split(key, 2)
            batch_cell_ind = random.choice(key=subkey, a=self.adata.shape[0], shape=[batch_size])
            point_clouds_batch, weights_batch = self._assemble_and_pad_batch(batch_cell_ind)
            
            if shape_sample is not None:
                keys = jax.random.split(subkey, batch_size)
                point_clouds_batch, weights_batch = sample_points(point_clouds_batch, weights_batch, keys, shape_sample)
            
            if self.conditioning_vectors is not None:
                conditioning_vectors_batch = self.conditioning_vectors[batch_cell_ind]
            else:
                conditioning_vectors_batch = None

            subkey, key = random.split(key, 2)
            self.state, loss = self.train_step(self.state, point_clouds_batch, weights_batch, conditioning_vectors_batch, key=subkey)
            self.params = self.state.params
            self.losses.append(loss)

            if step % verbose == 0:
                tq.set_description(f"Step {self.state.step}/{training_steps} | Loss: {loss:.4e}")

    def load_train_model(self, path):
        """
        Load a pre-trained train state into the model.
        :param path: Path to the saved model parameters.
        """ 
        print("Loading pre-trained model from", path)

        self.FlowMatchingModel = AttentionNN(config = self.config)
        with open(path, 'rb') as f:
            self.params = pickle.load(f)

        self.state = self.create_train_state(
                model=self.FlowMatchingModel,
                learning_rate = 1, 
                decay_steps = 1, 
                key=random.key(0)
            )
        self.state = self.state.replace(params=self.params)

    @partial(jit, static_argnums=(0,))
    def get_flow(self, params, point_clouds, weights, t, labels = None):
        """Applies the model to get the flow vector."""
        if(point_clouds.ndim == 2):
            squeeze = True
            point_clouds = point_clouds[None,:, :]
            weights = weights[None, :]
        else:
            squeeze = False
        
        flow = self.FlowMatchingModel.apply({"params": params}, point_cloud = point_clouds, t = t * jnp.ones(point_clouds.shape[0]), masks = weights>0, labels = labels, deterministic = True)
        
        if(squeeze):
            flow = jnp.squeeze(flow, axis = 0)
        return(flow)
    
    def generate_niches(self, 
                        num_samples: int = 10, 
                        timesteps: int = 100, 
                        conditioning_info: Optional[jnp.ndarray] = None, 
                        init_noise: Optional[jnp.ndarray] = None, 
                        key=random.key(0)):
        """
        Generates new cellular niches from the learned flow using classifier-free guidance.

        :param num_samples: (int) Number of niches to generate.
        :param timesteps: (int) Number of integration steps for the ODE solver.
        :param conditioning_info: (jnp.ndarray, optional) The conditioning vectors to generate for.
                                  Shape should be (num_samples, num_features) or (1, num_features).
                                  If None, conditions are sampled randomly from training data.
        :param init_noise: (jnp.ndarray, optional) An initial noise array to start generation from.
        :param key: JAX random key.
        :return: A tuple of (generated_niches_trajectory, final_conditioning_info).
        """
        # --- 0. Print Generation Information ---
        print("--- Starting Niche Generation ---")
        print(f"  - Niches to Generate: {num_samples}")
        print(f"  - Integration Timesteps: {timesteps}")
        
        is_conditional = self.conditioning_vectors is not None
        if is_conditional:
            if self.guidance_gamma > 1:
                print(f"  - Guidance: Classifier-free guidance active (gamma={self.guidance_gamma})")
            elif self.guidance_gamma == 1.0:
                 print(f"  - Guidance: Standard conditional generation (gamma=1.0)")
            else:
                 print("  - Guidance: Unconditional generation from a conditional model (gamma=0.0)")
        else:
            print("  - Guidance: Model is unconditional.")
        print("------------------------------------")

        key, subkey = random.split(key)

        # --- 1. Setup Particle Weights ---
        # For niche generation, all particles in the cloud are weighted equally.
        particle_weights = jnp.ones([num_samples, self.max_niche_size]) / self.max_niche_size

        # --- 2. Setup Conditional and Unconditional Labels ---
        generate_labels = None
        null_labels = None
        if is_conditional:
            if conditioning_info is None:
                # If no condition is provided, sample random ones from the training data.
                print("INFO: No `conditioning_info` provided, sampling conditions from training data.")
                indices = np.random.choice(self.conditioning_vectors.shape[0], num_samples, replace=True)
                generate_labels = self.conditioning_vectors[indices]
            else:
                # Use the provided conditioning info.
                if conditioning_info.ndim == 1:
                    # If one condition is given, repeat it for all samples.
                    generate_labels = jnp.tile(conditioning_info[None, :], [num_samples, 1])
                elif conditioning_info.shape[0] != num_samples:
                    raise ValueError(f"Shape mismatch: `conditioning_info` has {conditioning_info.shape[0]} samples, but `num_samples` is {num_samples}.")
                else:
                    generate_labels = conditioning_info
            
            # The "null" label for unconditional guidance is a zero vector.
            null_labels = jnp.zeros_like(generate_labels)

        # --- 3. Initialize Noise ---
        key, subkey = random.split(key)
        if init_noise is not None:
            if init_noise.ndim == 2:
                init_noise = init_noise[None, :, :] # Add batch dimension if missing
            generated_samples = [init_noise]
        else:
            # Generate initial noise from the base distribution.
            noise = self.noise_func(size=[num_samples, self.max_niche_size, self.space_dim], 
                                    noise_config=self.noise_config,
                                    key=subkey)
            generated_samples = [noise]

        # --- 4. Guided Integration Loop (RK2 Midpoint Method) ---
        print(f"INFO: Initial sample shape: {generated_samples[0].shape}")
        
        dt = 1.0 / timesteps
        for t_val in tqdm(jnp.linspace(1.0, dt, timesteps), desc="Generating Niches"):
            xt = generated_samples[-1]
            t_curr = jnp.full((num_samples,), t_val)
            t_mid = jnp.full((num_samples,), t_val - 0.5 * dt)

            # --- First RK2 step (calculating midpoint) ---
            if not is_conditional or self.guidance_gamma == 0.0:
                # Unconditional generation
                vt = self.get_flow(self.params, xt, particle_weights, t_curr, None)
            elif self.guidance_gamma == 1.0:
                # Standard conditional generation
                vt = self.get_flow(self.params, xt, particle_weights, t_curr, generate_labels)
            else:
                # Classifier-free guidance
                v_c = self.get_flow(self.params, xt, particle_weights, t_curr, generate_labels)
                v_u = self.get_flow(self.params, xt, particle_weights, t_curr, null_labels)
                vt = v_u + self.guidance_gamma * (v_c - v_u)

            x_mid = xt - 0.5 * dt * vt
            
            # --- Second RK2 step (final update) ---
            if not is_conditional or self.guidance_gamma == 0.0:
                v_mid = self.get_flow(self.params, x_mid, particle_weights, t_mid, None)
            elif self.guidance_gamma == 1.0:
                v_mid = self.get_flow(self.params, x_mid, particle_weights, t_mid, generate_labels)
            else:
                v_c_mid = self.get_flow(self.params, x_mid, particle_weights, t_mid, generate_labels)
                v_u_mid = self.get_flow(self.params, x_mid, particle_weights, t_mid, null_labels)
                v_mid = v_u_mid + self.guidance_gamma * (v_c_mid - v_u_mid)

            x_t_minus_dt = xt - dt * v_mid
            generated_samples.append(x_t_minus_dt)

        # --- 5. Return Results ---
        print("--- Generation Complete ---")
        return generated_samples, generate_labels
