from functools import partial
import types
import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax import jit, random
from tqdm import tqdm
from flax.training import train_state
import pickle

import wassersteinflowmatching.riemannian_wasserstein.utils_OT as utils_OT
import wassersteinflowmatching.riemannian_wasserstein.utils_Geom as utils_Geom
import wassersteinflowmatching.riemannian_wasserstein.utils_Noise as utils_Noise
from wassersteinflowmatching.riemannian_wasserstein._utils_Transformer import AttentionNN
from wassersteinflowmatching.riemannian_wasserstein.DefaultConfig import DefaultConfig
from wassersteinflowmatching.riemannian_wasserstein._utils_Processing import pad_pointclouds, pad_pointclouds_features_and_masks
from wassersteinflowmatching.riemannian_wasserstein.RiemannianWassersteinFlowMatching import RiemannianWassersteinFlowMatching

class SE3WassersteinFlowMatching(RiemannianWassersteinFlowMatching):
    def __init__(
        self,
        point_clouds,
        conditioning = None,
        config = DefaultConfig,
        **kwargs,
    ):
        print("Initializing SE3WassersteinFlowMatching")

        self.config = config

        for key, value in kwargs.items():
            setattr(self.config, key, value)
        
        self.geom = self.config.geom
        
        # Handle masks
        if 'masks' in kwargs:
            self.masks = kwargs['masks']
        else:
            self.masks = [np.ones((pc.shape[0], pc.shape[1] // 7)) for pc in point_clouds]

        self.monge_map = self.config.monge_map
        self.num_sinkhorn_iters = self.config.num_sinkhorn_iters

        print(f'Using {self.geom} geometry')
        print("Projecting point clouds to geometry (with cpu)...")
        
        # Project with masks - handling variable dimensions
        projected_pcs = []
        for i, pc in enumerate(tqdm(point_clouds)):
            n_res = pc.shape[-1] // 7
            # Create temp geometry for projection
            temp_geom = utils_Geom.SE3_n(n=n_res)
            
            # Project
            proj_pc = np.asarray(temp_geom.project_to_geometry(pc, use_cpu=True))
            projected_pcs.append(proj_pc)
        
        self.point_clouds = projected_pcs

        self.weights = [
            np.ones(pc.shape[0]) / pc.shape[0] for pc in self.point_clouds
        ]

        # Pad point clouds, weights AND masks
        self.point_clouds, self.weights, self.masks = pad_pointclouds_features_and_masks(
            self.point_clouds, self.weights, self.masks
        )

        print(f"Padded point clouds to shape: {self.point_clouds.shape}")
        print(f"Padded weights to shape: {self.weights.shape}")
        print(f"Padded masks to shape: {self.masks.shape}")

        self.space_dim = self.point_clouds.shape[-1]
        
        # Initialize main geometry with max_n (from padded data)
        max_dim = self.point_clouds.shape[-1]
        n_features = max_dim // 7

        self.geom_utils = getattr(utils_Geom, self.geom)(n = n_features)
        self.project_to_geometry = self.geom_utils.project_to_geometry

        # Setup vmaps
        self.interpolant_vmap = jax.vmap(jax.vmap(self.geom_utils.interpolant, in_axes=(0, 0, None), out_axes=0), in_axes=(0, 0, 0), out_axes=0)
        self.interpolant_velocity_vmap = jax.vmap(jax.vmap(self.geom_utils.velocity, in_axes=(0, 0, None), out_axes=0), in_axes=(0, 0, 0), out_axes=0)
        self.exponential_map_vmap = jax.vmap(jax.vmap(self.geom_utils.exponential_map, in_axes=(0, 0, None), out_axes=0), in_axes=(0, 0, None), out_axes=0)
        self.loss_func_vmap = jax.vmap(jax.vmap(self.geom_utils.tangent_norm, in_axes=(0, 0, 0, 0), out_axes=0), in_axes=(0, 0, 0, 0), out_axes=0)

        self.noise_config = types.SimpleNamespace()
        self.noise_type = self.config.noise_type
        
        self.noise_geom = self.config.noise_geom
        if(self.noise_geom != self.geom):
            print(f"Using {self.noise_geom} geometry for noise instead of {self.geom}")
            self.noise_geom = self.config.noise_geom
            self.noise_proj_to_geometry = getattr(utils_Geom, self.noise_geom)(n = n_features).project_to_geometry 
        else:
            print(f"Using {self.noise_geom} geometry for noise")
            self.noise_proj_to_geometry = self.project_to_geometry  
        
        # Get noise functions from the factory
        self.noise_func, param_estimator = utils_Noise.get_noise_functions(self.noise_type, self.noise_proj_to_geometry)

        self.matched_noise = False 

        # Estimate parameters if an estimator is available
        if param_estimator:
            params = param_estimator(self.point_clouds, self.weights)
            for key, value in params.items():
                setattr(self.noise_config, key, value)

        print(f"Using {self.noise_type} noise for {self.geom} geometry.")
        if self.noise_config.__dict__:
            print("Noise parameters:")
            for key, value in self.noise_config.__dict__.items():
                print(f"  {key}: {value}")

        if self.num_sinkhorn_iters == -1 and self.monge_map != 'random' and self.monge_map != 'matched':
            print("Finding optimal number of Sinkhorn iterations...")
            key = random.key(0)
            
            N, K, D = self.point_clouds.shape
            sample_N = min(N, 100)
            sample_K = min(K, 512)
            
            print(f"Sampling {sample_N} point clouds with {sample_K} points each for Sinkhorn iteration estimation.")

            pc_indices = np.random.choice(N, sample_N, replace=False)
            pc_subset = []
            w_subset = []
            m_subset = []
            for idx in pc_indices:
                w = self.weights[idx]
                p = w / np.sum(w)
                point_indices = np.random.choice(K, sample_K, replace=True, p=p)
                pc_subset.append(self.point_clouds[idx][point_indices])
                w_subset.append(np.ones(sample_K) / sample_K)
                m_subset.append(self.masks[idx][point_indices])

            pc_subset = np.array(pc_subset)
            w_subset = np.array(w_subset)
            m_subset = np.array(m_subset)

            sample_shape = (sample_N, sample_K, D)

            noise_samples = self.noise_func(size=sample_shape, 
                                            noise_config=self.noise_config,
                                            key=key)
            noise_weights = w_subset
            noise_m = m_subset

            noise_samples = self.project_to_geometry(noise_samples)
            
            pc_list = [np.array(pc) for pc in pc_subset]
            w_list = [np.array(w) for w in w_subset]
            m_list = [np.array(m) for m in m_subset]

            noise_list = [np.array(n) for n in noise_samples]
            nw_list = [np.array(nw) for nw in noise_weights]
            nm_list = [np.array(nm) for nm in noise_m]

            print(f"pc shapes for auto sinkhorn: {pc_list[0].shape}, noise shapes: {noise_list[0].shape}")
            print(f"weight shapes for auto sinkhorn: {w_list[0].shape}, noise weight shapes: {nw_list[0].shape}")
            print(f"mask shapes for auto sinkhorn: {m_list[0].shape}, noise mask shapes: {nm_list[0].shape}")
            self.num_sinkhorn_iters = utils_OT.auto_find_num_iter(
                point_clouds=pc_list,
                weights=w_list,
                eps=self.config.wasserstein_eps,
                lse_mode=self.config.wasserstein_lse,
                distance_matrix_func=self.geom_utils.distance_matrix,
                noise_point_clouds=noise_list,
                noise_weights=nw_list,
                feature_masks = m_list,
                noise_feature_masks = nm_list,
            )
            self.config.num_sinkhorn_iters = self.num_sinkhorn_iters
            print(f"Auto-selected {self.num_sinkhorn_iters} Sinkhorn iterations.")

        print(f"Using {self.monge_map} map with {self.num_sinkhorn_iters} iterations and {self.config.wasserstein_eps} epsilon")
        
        if(self.monge_map == 'random'):
            self.transport_plan_jit = jax.vmap(utils_OT.transport_plan_random,
                                    (0, 0), 0)
        elif(self.monge_map == 'matched'):
            self.transport_plan_jit = jax.vmap(utils_OT.transport_plan_matched,
                        (0, 0), 0)
        else:
            self.transport_plan_jit = jax.vmap(partial(utils_OT.transport_plan, 
                                    distance_matrix_func = self.geom_utils.distance_matrix,
                                    eps = self.config.wasserstein_eps, 
                                    lse_mode = self.config.wasserstein_lse, 
                                    num_iteration = self.config.num_sinkhorn_iters),
                                    (0, 0), 0)
        if(self.monge_map == 'rounded_matching'):
            self.sample_map_jit = jax.vmap(lambda P, pc_y: utils_OT.get_assignments_rounding(P, pc_y)[0], (0, 0), 0)
        elif(self.monge_map == 'sample' or self.monge_map == 'random' or self.monge_map == 'matched'):
            self.sample_map_jit = jax.vmap(lambda P, pc_y, key: utils_OT.get_assignments_sampling(P, key, pc_y)[0], (0, 0, 0), 0)
        elif(self.monge_map == 'barycentric'):
            self.sample_map_jit = jax.vmap(lambda P, pc_y: partial(utils_OT.get_assignments_barycentric, weighted_mean_func=self.geom_utils.weighted_mean)(P, pc_y)[0], (0, 0), 0)
        elif(self.monge_map == 'entropic'):
            self.sample_map_jit = jax.vmap(lambda P, pc_x, pc_y: utils_OT.get_assignments_entropic(P, pc_x, pc_y, self.geom_utils.velocity, self.geom_utils.exponential_map)[0], (0, 0, 0), 0)
        else:
            raise ValueError(f"Unknown monge_map: {self.monge_map}")
        

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
            self.cfg = False
            self.p_cfg_null = 0.0
            self.w_cfg = 1.0
            self.conditioning = None
            self.conditioning_dim = -1
 
    
        if(self.mini_batch_ot_mode):
            self.mini_batch_ot_solver = self.config.mini_batch_ot_solver

            if(self.mini_batch_ot_solver == 'entropic'):
                print("Entropic Mini-Batch")
                self.ot_mat_jit = jax.vmap(partial(utils_OT.entropic_ot_distance, 
                                                   eps = self.config.minibatch_ot_eps,
                                                   lse_mode = self.config.minibatch_ot_lse,
                                                   num_iteration = self.num_sinkhorn_iters), (0, 0), 0)
            elif(self.mini_batch_ot_solver == 'matched'):
                print("Matched Mini-Batch")
                self.ot_mat_jit = jax.vmap(partial(utils_OT.matched_distance, 
                            distance_func = self.geom_utils.distance), (0, 0), 0)
            elif(self.mini_batch_ot_solver == 'random'):
                print("Random Mini-Batch")
                self.ot_mat_jit = jax.vmap(partial(utils_OT.random_distance, 
                            distance_matrix_func = self.geom_utils.distance_matrix), (0, 0), 0)
            elif(self.mini_batch_ot_solver == 'chamfer'):
                print("Chamfer Mini-Batch")
                self.ot_mat_jit = jax.vmap(partial(utils_OT.chamfer_distance, 
                                            distance_matrix_func = self.geom_utils.distance_matrix), (0, 0), 0)
            elif(self.mini_batch_ot_solver == 'euclidean'):
                print("Euclidean Mini-Batch")
                self.ot_mat_jit = jax.vmap(utils_OT.euclidean_distance, (0, 0), 0)
            else:
                raise ValueError(f"Unknown mini_batch_ot_solver: {self.mini_batch_ot_solver}")

            self.mini_batch_ot_num_iter = self.config.mini_batch_ot_num_iter
            if self.mini_batch_ot_num_iter == -1:
                print("Finding optimal number of Sinkhorn iterations for Mini-Batch OT...")
                
                key = random.key(0)
                
                N, K, D = self.point_clouds.shape
                sample_N = min(N, 100)
                sample_K = min(K, 512)
                
                pc_indices = np.random.choice(N, sample_N, replace=False)
                pc_subset = []
                w_subset = []
                m_subset = []
                for idx in pc_indices:
                    w = self.weights[idx]
                    p = w / np.sum(w)
                    point_indices = np.random.choice(K, sample_K, replace=True, p=p)
                    pc_subset.append(self.point_clouds[idx][point_indices])
                    w_subset.append(np.ones(sample_K) / sample_K)
                    m_subset.append(self.masks[idx])

                pc_subset = np.array(pc_subset)
                w_subset = np.array(w_subset)
                m_subset = np.array(m_subset)

                sample_shape = (sample_N, sample_K, D)

                noise_samples = self.noise_func(size=sample_shape, 
                                                noise_config=self.noise_config,
                                                key=key)
                noise_weights = w_subset
                noise_m = m_subset  
                
                noise_samples = self.project_to_geometry(noise_samples)

                self.mini_batch_ot_num_iter = utils_OT.auto_find_num_iter_minibatch(
                    point_clouds=pc_subset,
                    weights=w_subset,
                    noise_point_clouds=noise_samples,
                    noise_weights=noise_weights,
                    feature_masks = m_subset,
                    noise_feature_masks = noise_m,
                    ot_mat_jit=self.ot_mat_jit,
                    eps = self.config.minibatch_ot_eps,
                    lse_mode = self.config.minibatch_ot_lse,
                    sample_size=2048
                )
                print(f"Auto-selected {self.mini_batch_ot_num_iter} Sinkhorn iterations for Mini-Batch OT.")
        
        self.FlowMatchingModel = AttentionNN(config = self.config)

    def minibatch_ot(self, point_clouds, point_cloud_weights, point_cloud_masks, noise, noise_weights, noise_masks, key = random.key(0)):

        """
        :meta private:
        """
            
        matrix_ind = jnp.array(jnp.meshgrid(jnp.arange(point_clouds.shape[0]), jnp.arange(noise.shape[0]))).T.reshape(-1, 2)


        # compute pairwise ot between point clouds and noise:
        

        ot_matrix = self.ot_mat_jit([point_clouds[matrix_ind[:, 0]], 
                                     point_cloud_weights[matrix_ind[:, 0]], 
                                     point_cloud_masks[matrix_ind[:, 0]]],
                                    [noise[matrix_ind[:, 1]], 
                                     noise_weights[matrix_ind[:, 1]], 
                                     noise_masks[matrix_ind[:, 1]]]).reshape(point_clouds.shape[0], noise.shape[0])

        noise_ind, ot_solve = utils_OT.ot_mat_from_distance(ot_matrix, 0.01, True, num_iteration=self.mini_batch_ot_num_iter)
        return(noise_ind, ot_solve)
    
    @partial(jit, static_argnums=(0,))
    def train_step(self, state, point_clouds_batch, weights_batch, masks_batch, conditioning_batch=None, is_null_conditioning=None, key=random.key(0)):
        """
        JIT-compiled training step with internal function timing.
        """

        # Time random.split operation
        subkey_t, subkey_noise, key = random.split(key, 3)

        noise_samples = self.noise_func(size=point_clouds_batch.shape, 
                                        noise_config=self.noise_config,
                                        key=subkey_noise)
        noise_weights = weights_batch
        noise_masks = masks_batch
        # noise_samples = self.project_to_geometry(noise_samples)

        if self.mini_batch_ot_mode:
            # Time minibatch_ot operation
            minibatch_key, key = random.split(key)
            noise_ind = self.minibatch_ot(point_clouds_batch, weights_batch, masks_batch, noise_samples, noise_weights, noise_masks, key=minibatch_key)[0]
            noise_samples = noise_samples[noise_ind]
            if(self.monge_map != 'rounded_matching' and self.monge_map != 'matched'):
                noise_weights = noise_weights[noise_ind]
            noise_masks = noise_masks[noise_ind]

        # set noise samples where noise_masks == 0 to zero

        # noise is n_x, n_points, n_residues*7
        # noise_masks is n_x, n_points, n_residues
        noise_samples = noise_samples * jnp.repeat(noise_masks, 7, axis=-1)

        # Time random.uniform for interpolates_time
        interpolates_time = random.uniform(subkey_t, (point_clouds_batch.shape[0],), minval=0.0, maxval=1.0)

        # Time transport_plan_jit operation
        ot_matrix = self.transport_plan_jit([noise_samples, noise_weights, noise_masks], 
                                           [point_clouds_batch, weights_batch, masks_batch])[0]

        if self.monge_map == 'sample' or self.monge_map == 'matched' or self.monge_map == 'random':
            # For sampling assignment, we need the random keys
            assigned_points = self.sample_map_jit(ot_matrix, point_clouds_batch, random.split(key, point_clouds_batch.shape[0]))
        elif(self.monge_map == 'entropic'):
            assigned_points = self.sample_map_jit(ot_matrix, noise_samples, point_clouds_batch)
        else:
            # For other methods (rounded_matching and barycentric), we don't need keys
            assigned_points = self.sample_map_jit(ot_matrix, point_clouds_batch)

        point_cloud_interpolates = self.interpolant_vmap(noise_samples, assigned_points, 1-interpolates_time)
        point_cloud_velocity = self.interpolant_velocity_vmap(noise_samples, assigned_points, 1-interpolates_time)
  

        # set point_cloud_interpolates and point_cloud_velocity to where masks_batch == 0 to zero

        point_cloud_interpolates = point_cloud_interpolates * jnp.repeat(masks_batch, 7, axis=-1)
        point_cloud_velocity = point_cloud_velocity * jnp.repeat(masks_batch, 7, axis=-1)
        
        subkey, key = random.split(key)

        def loss_fn(params):
            # Time loss function evaluation
            predicted_flow = state.apply_fn({"params": params},  
                                            point_cloud = point_cloud_interpolates, 
                                            t = interpolates_time, 
                                            masks = noise_weights > 0, 
                                            conditioning = conditioning_batch,
                                            is_null_conditioning = is_null_conditioning,
                                            deterministic = False, 
                                            dropout_rng = subkey)
            # set predicted_flow to zero where masks_batch == 0
            predicted_flow = predicted_flow * jnp.repeat(masks_batch, 7, axis=-1)
            
            error = self.loss_func_vmap(predicted_flow, -point_cloud_velocity, point_cloud_interpolates, masks_batch) * noise_weights

            loss = jnp.mean(jnp.sum(error, axis=1))
            return loss

        # Time backpropagation
        loss, grads = jax.value_and_grad(loss_fn)(state.params)
        state = state.apply_gradients(grads=grads)

        return state, loss


    def sample_single_batch(self, single_batch, single_weights, single_masks, key, n_points):
        num_valid = jnp.sum(single_weights > 0)
        
        def sample_without_replacement(k):
            p = single_weights / jnp.sum(single_weights)
            return jax.random.choice(k, single_batch.shape[0], (n_points,), replace=False, p=p)
            
        def take_all_padded(k):
            return jnp.argsort(single_weights > 0)[::-1][:n_points]

        indices = jax.lax.cond(
            num_valid >= n_points,
            sample_without_replacement,
            take_all_padded,
            key
        )
        
        sampled_pc = jnp.take(single_batch, indices, axis=0)
        sample_weights = jnp.take(single_weights, indices, axis=0)
        sample_masks = jnp.take(single_masks, indices, axis=0)

        total_weight = jnp.sum(sample_weights)
        sample_weights = jnp.where(total_weight > 0, sample_weights / total_weight, sample_weights)
        
        return [sampled_pc, sample_weights, sample_masks]

    def train(
        self,
        training_steps=32000,
        batch_size=16,
        verbose=8,
        learning_rate = 2e-4, 
        decay_steps = 1000,
        shape_sample = None,
        saved_state = None,
        key=random.key(0),
    ):
        """
        Set up optimization parameters and train the ENVI moodel


        :param training_steps: (int) number of gradient descent steps to train ENVI (default 10000)
        :param batch_size: (int) size of train-set point clouds sampled for each training step  (default 16)
        :param verbose: (int) amount of steps between each loss print statement (default 8)
        :param learning_rate: (float) initial learning rate for ADAM optimizer with exponential decay (default 1e-4)
        :param decay_steps: (int) number of times of learning rate decay during training (default 4)
        :param key: (jax.random.key) random seed (default jax.random.key(0))

        :return: nothing
        """

        subkey, key = random.split(key)

                
        if saved_state is None:
            print("Creating new train state")
            self.state = self.create_train_state(
                model=self.FlowMatchingModel,
                learning_rate = learning_rate, 
                decay_steps = decay_steps, 
                key=subkey
            )
            print("Train state created")
        else:
            self.state = saved_state
            print(f"Resuming training from step {int(self.state.step)}")


        if(shape_sample is not None):
            print(f'Sampling {shape_sample} points from each point cloud')
            shape_sample = min(shape_sample, self.point_clouds.shape[1])
            sample_points = jax.vmap(self.sample_single_batch, in_axes=(0, 0, 0, 0, None))

        tq = tqdm(range(training_steps - self.state.step), leave=True, desc="")
        self.losses = []
        try:
            for training_step in tq:

                subkey, key = random.split(key, 2)
                batch_ind = random.choice(
                    key=subkey,
                    a = self.point_clouds.shape[0],
                    shape=[batch_size])
                
                point_clouds_batch = self.point_clouds[batch_ind]
                weights_batch = self.weights[batch_ind]
                masks_batch = self.masks[batch_ind]
                
                if(shape_sample is not None):
                    keys = jax.random.split(subkey, batch_size)
                    point_clouds_batch, weights_batch, masks_batch = sample_points(point_clouds_batch, weights_batch, masks_batch, keys, shape_sample)
                    
                if(self.conditioning is not None):
                    conditioning_batch = self.conditioning[batch_ind].copy()
                    if self.cfg:
                        subkey, key = random.split(key)
                        is_null_conditioning = random.bernoulli(subkey, p=self.p_cfg_null, shape=(batch_size,))

                        # for each row in conditioning_batch where all values are NaN, set is_null_conditioning to True
                        
                        is_null_conditioning = jnp.logical_or(is_null_conditioning, jnp.any(jnp.isnan(conditioning_batch), axis=1)) 

                    else:
                        is_null_conditioning = None
                    
                    conditioning_batch = jnp.where(jnp.isnan(conditioning_batch), 0.0, conditioning_batch)
                    
                else:
                    conditioning_batch = None
                    is_null_conditioning = None

                subkey, key = random.split(key, 2)

                self.state, loss = self.train_step(self.state, point_clouds_batch, weights_batch, masks_batch, conditioning_batch, is_null_conditioning, key = subkey)

                self.params = self.state.params
                self.losses.append(loss) 


                if(training_step % verbose == 0):
                    tq.set_description(": {:.3e}".format(loss))
        except KeyboardInterrupt:
            print("\nTraining interrupted by user. The model state has been saved.")



    @partial(jit, static_argnums=(0,))
    def get_flow(self, params, point_clouds, weights, masks, t, dt, conditioning=None, is_null_conditioning=None):
        if point_clouds.ndim == 2:
            point_clouds = point_clouds[None, :, :]
            weights = weights[None, :]
            masks = masks[None, :, :]

        # set point_clouds where masks == 0 to zero
        point_clouds = point_clouds * jnp.repeat(masks, 7, axis=-1)
        
        if conditioning is not None:
            if is_null_conditioning is None:
                is_null_conditioning = jnp.isnan(conditioning).any(axis=-1)
            
            conditioning = jnp.where(jnp.isnan(conditioning), 0.0, conditioning)

            cond_flow = self.FlowMatchingModel.apply(
                {"params": params},
                point_cloud=point_clouds,
                t=t * jnp.ones(point_clouds.shape[0]),
                masks=weights > 0,
                conditioning=conditioning,
                is_null_conditioning=is_null_conditioning,
                deterministic=True
            )

            if(self.cfg):
                uncond_flow = self.FlowMatchingModel.apply(
                    {"params": params},
                    point_cloud=point_clouds,
                    t=t * jnp.ones(point_clouds.shape[0]),
                    masks=weights > 0,
                    conditioning=conditioning,
                    is_null_conditioning=jnp.ones(point_clouds.shape[0], dtype=bool),
                    deterministic=True
                )


                flow = uncond_flow + self.w_cfg * (cond_flow - uncond_flow)
            else:
                flow = cond_flow
        else:

            flow = self.FlowMatchingModel.apply(
                {"params": params},
                point_cloud=point_clouds,
                t=t * jnp.ones(point_clouds.shape[0]),
                masks=weights > 0,
                deterministic=True
            )

        update = self.exponential_map_vmap(point_clouds, flow, -dt)
        
        # set update to 0 where masks == 0

        update = update * jnp.repeat(masks, 7, axis=-1)

        return update
    
    def generate_samples(self, size=None, num_samples=10, timesteps=100, generate_conditioning=None, init_noise=None, max_size = None, n_residues=None, key=random.key(0)):
        """
        Generate samples from the learned flow

        :param num_samples: (int) number of samples to generate (default 10)
        :param timesteps: (int) number of timesteps to generate samples (default 100)
        :param n_residues: (int) optional number of residues to mask the generated samples

        :return: generated samples, weights, (conditioning), masks
        """
        if size is None:
            size = self.point_clouds.shape[1]
            noise_weights = None
        else:
            noise_weights = jnp.ones([num_samples, size])

        if self.conditioning is None:
            generate_conditioning = None
            if noise_weights is None:
                subkey, key = random.split(key)
                noise_weights = random.choice(subkey, self.weights, shape=(num_samples,))
        else:
            if generate_conditioning is None:
                rand_indices = random.choice(key, self.conditioning.shape[0], shape=(num_samples,), replace=True)
                generate_conditioning = self.conditioning[rand_indices]
                
                if noise_weights is None:
                    noise_weights = self.weights[rand_indices]
            else:
                generate_conditioning = jnp.array(generate_conditioning)
                if generate_conditioning.ndim == 1:
                    generate_conditioning = generate_conditioning[None, :]
                
                if generate_conditioning.shape[0] == 1 and num_samples > 1:
                     generate_conditioning = jnp.repeat(generate_conditioning, num_samples, axis=0)

                if noise_weights is None:
                    subkey, key = random.split(key)
                    noise_weights = random.choice(subkey, self.weights, shape=(num_samples,))

        subkey, key = random.split(key)
        
        # Handle masks
        if n_residues is not None:
            n_features = self.space_dim // 7
            # Create mask for first n_residues
            n_residues = jnp.array(n_residues)
            if n_residues.ndim == 0:
                mask_indices = jnp.arange(n_features) < n_residues
                masks = jnp.tile(mask_indices[None, None, :], (num_samples, size, 1))
            else:
                mask_indices = jnp.arange(n_features)[None, :] < n_residues[:, None]
                masks = jnp.tile(mask_indices[:, None, :], (1, size, 1))
        else:
            subkey, key = random.split(key)
            # Sample random masks from training distribution
            rand_indices = random.choice(subkey, self.masks.shape[0], shape=(num_samples,), replace=True)
            masks = self.masks[rand_indices][:, :size, :]

        if init_noise is not None:
            if init_noise.ndim == 2:
                init_noise = init_noise[None, :, :]
            noise = init_noise
        else:
            noise = self.noise_func(size=[num_samples, size, self.space_dim],
                                    noise_config=self.noise_config,
                                    key=subkey)
            if isinstance(noise, (list, tuple)) and len(noise) == 2:
                noise, _ = noise
        
        # Apply masks to noise (zero out masked regions)
        masks_expanded = jnp.repeat(masks, 7, axis=-1)
        noise = noise * masks_expanded

        # fix size of noise and noise_weights to the max size of noise_weights.sum(axis=1)

        if max_size is None:
            max_size = int(jnp.max(jnp.sum(noise_weights > 0, axis=1)))
        else:
            max_size = min(max_size, noise.shape[1])
        
        # reorder noise and noise_weights to have the valid points first
        def reorder_points(single_noise, single_weights, single_masks):
            sorted_indices = jnp.argsort(single_weights > 0)[::-1]
            reordered_noise = jnp.take(single_noise, sorted_indices, axis=0)
            reordered_weights = jnp.take(single_weights, sorted_indices, axis=0)
            reordered_masks = jnp.take(single_masks, sorted_indices, axis=0)
            return reordered_noise[:max_size, :], reordered_weights[:max_size], reordered_masks[:max_size, :]
        
        noise, noise_weights, masks = jax.vmap(reorder_points, in_axes=(0, 0, 0))(noise, noise_weights, masks)
        dt = 1 / timesteps

        def step_fn(carry, t):
            current_noise = carry
            next_noise = self.get_flow(self.params, current_noise, noise_weights, masks, t, dt, generate_conditioning)
            return next_noise, next_noise

        timesteps_array = jnp.linspace(1, dt, timesteps)
        _, all_noises = jax.lax.scan(step_fn, noise, timesteps_array)

        if generate_conditioning is None:
            return all_noises, noise_weights, masks
        return all_noises, noise_weights, masks, generate_conditioning

