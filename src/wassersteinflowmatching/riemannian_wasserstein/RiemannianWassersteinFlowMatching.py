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

import src.wassersteinflowmatching.riemannian_wasserstein.utils_OT as utils_OT # type: ignore
import src.wassersteinflowmatching.riemannian_wasserstein.utils_Geom as utils_Geom # type: ignore  # noqa: F401
import src.wassersteinflowmatching.riemannian_wasserstein.utils_Noise as utils_Noise # type: ignore
from src.wassersteinflowmatching.riemannian_wasserstein._utils_Transformer import AttentionNN # type: ignore
from src.wassersteinflowmatching.riemannian_wasserstein.DefaultConfig import DefaultConfig # type: ignore
from src.wassersteinflowmatching.riemannian_wasserstein._utils_Processing import pad_pointclouds # type: ignore


class RiemannianWassersteinFlowMatching:
    """
    Initializes WFM model and processes input point clouds


    :param point_clouds: (list of np.array) list of train-set point clouds to flow match
    :param config: (flax struct.dataclass) object with parameters

    :return: initialized WassersteinFlowMatching model
    """

    def __init__(
        self,
        point_clouds,
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

        print("Projecting point clouds to geometry (with cpu)...")
        

        self.point_clouds = [np.asarray(self.project_to_geometry(pc, use_cpu = True)) for pc in tqdm(point_clouds)]


        self.weights = [
            np.ones(pc.shape[0]) / pc.shape[0] for pc in self.point_clouds
        ]

        self.point_clouds, self.weights = pad_pointclouds(
            self.point_clouds, self.weights
        )

        self.space_dim = self.point_clouds.shape[-1]

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
            params = param_estimator(self.point_clouds, self.weights)
            for key, value in params.items():
                setattr(self.noise_config, key, value)

        print(f"Using {self.noise_type} noise for {self.geom} geometry.")
        if self.noise_config.__dict__:
            print("Noise parameters:")
            for key, value in self.noise_config.__dict__.items():
                print(f"  {key}: {value}")

        if self.num_sinkhorn_iters == -1 and self.monge_map != 'random':
            print("Finding optimal number of Sinkhorn iterations...")
            key = random.key(0)
            noise_samples = self.noise_func(size=self.point_clouds.shape, 
                                            noise_config=self.noise_config,
                                            key=key)
            if len(noise_samples) == 2:
                noise_samples, noise_weights = noise_samples
            else:
                noise_weights = self.weights
            
            noise_samples = self.project_to_geometry(noise_samples)
            
            pc_list = [np.array(pc) for pc in self.point_clouds]
            w_list = [np.array(w) for w in self.weights]
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
        
        if(self.monge_map == 'random'):
            self.transport_plan_jit = jax.vmap(utils_OT.transport_plan_random,
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
        elif(self.monge_map == 'sample' or self.monge_map == 'random'):
            self.sample_map_jit = jax.vmap(lambda P, pc_y, key: utils_OT.get_assignments_sampling(P, key, pc_y)[0], (0, 0, 0), 0)
        elif(self.monge_map == 'barycentric'):
            # Entropic assignment uses geometry-specific weighted mean
            self.sample_map_jit = jax.vmap(lambda P, pc_y: partial(utils_OT.get_assignments_barycentric, weighted_mean_func=self.geom_utils.weighted_mean)(P, pc_y)[0], (0, 0), 0)
        elif(self.monge_map == 'entropic'):
            self.sample_map_jit = jax.vmap(lambda P, pc_x, pc_y: utils_OT.get_assignments_entropic(P, pc_x, pc_y, self.geom_utils.velocity, self.geom_utils.exponential_map)[0], (0, 0, 0), 0)
        else:
            # raise error for unknown monge_map
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
                print("Frechet Mini-Batch")
                self.ot_mat_jit = jax.vmap(utils_OT.frechet_distance, (0, 0), 0)

            self.mini_batch_ot_num_iter = self.config.mini_batch_ot_num_iter
            if self.mini_batch_ot_num_iter == -1:
                print("Finding optimal number of Sinkhorn iterations for Mini-Batch OT...")
                
                key = random.key(0)
                noise_samples = self.noise_func(size=self.point_clouds.shape, 
                                                noise_config=self.noise_config,
                                                key=key)
                if len(noise_samples) == 2:
                    noise_samples, noise_weights = noise_samples
                else:
                    noise_weights = self.weights
                
                noise_samples = self.project_to_geometry(noise_samples)

                self.mini_batch_ot_num_iter = utils_OT.auto_find_num_iter_minibatch(
                    point_clouds=self.point_clouds,
                    weights=self.weights,
                    noise_point_clouds=noise_samples,
                    noise_weights=noise_weights,
                    ot_mat_jit=self.ot_mat_jit,
                    eps = self.config.minibatch_ot_eps,
                    lse_mode = self.config.minibatch_ot_lse,
                    sample_size=2048
                )
                print(f"Auto-selected {self.mini_batch_ot_num_iter} Sinkhorn iterations for Mini-Batch OT.")
        
        self.FlowMatchingModel = AttentionNN(config = self.config)

    

    def create_train_state(self, model, learning_rate, decay_steps, key = random.key(0)):
        """
        :meta private:
        """

        subkey, key = random.split(key)
        attn_inputs =  self.noise_func(size = [10, min(self.point_clouds[0].shape[0], 32), self.space_dim], 
                                       noise_config = self.noise_config,
                                       key = subkey)
    

        if(len(attn_inputs) == 2):
            attn_inputs = attn_inputs[0]
        subkey, key = random.split(key)

        if(self.conditioning is not None):
            params = model.init(rngs={"params": subkey}, 
                                point_cloud = attn_inputs, 
                                t = jnp.ones((attn_inputs.shape[0])), 
                                masks = jnp.ones((attn_inputs.shape[0], attn_inputs.shape[1])),
                                conditioning =  jnp.ones((attn_inputs.shape[0], self.conditioning_dim)),
                                deterministic = True)['params']
        else:
            params = model.init(rngs={"params": subkey}, 
                    point_cloud = attn_inputs, 
                    t = jnp.ones((attn_inputs.shape[0])), 
                    masks = jnp.ones((attn_inputs.shape[0], attn_inputs.shape[1])),
                    deterministic = True)['params']



        lr_sched = optax.exponential_decay(
            learning_rate, decay_steps, 0.99, staircase = False,
        )

        tx = optax.adam(lr_sched)  #

        return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)


    def minibatch_ot(self, point_clouds, point_cloud_weights, noise, noise_weights, key = random.key(0)):

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

        noise_ind, ot_solve = utils_OT.ot_mat_from_distance(ot_matrix, 0.01, True, num_iteration=self.mini_batch_ot_num_iter)
        return(noise_ind, ot_solve)

    @partial(jit, static_argnums=(0,))
    def train_step(self, state, point_clouds_batch, weights_batch, conditioning_batch=None, noise_samples=None, noise_weights=None, is_null_conditioning=None, key=random.key(0)):
        """
        JIT-compiled training step with internal function timing.
        """

        # Time random.split operation
        subkey_t, subkey_noise, key = random.split(key, 3)

        if noise_samples is None:

            noise_samples = self.noise_func(size=point_clouds_batch.shape, 
                                            noise_config=self.noise_config,
                                            key=subkey_noise)
            if len(noise_samples) == 2:
                noise_samples, noise_weights = noise_samples
            else:
                noise_weights = weights_batch
            
            noise_samples = self.project_to_geometry(noise_samples)

        if self.mini_batch_ot_mode:
            # Time minibatch_ot operation
            minibatch_key, key = random.split(key)
            noise_ind = self.minibatch_ot(point_clouds_batch, weights_batch, noise_samples, noise_weights, key=minibatch_key)[0]
            noise_samples = noise_samples[noise_ind]
            if(self.monge_map != 'rounded_matching'):
                noise_weights = noise_weights[noise_ind]

        # Time random.uniform for interpolates_time
        interpolates_time = random.uniform(subkey_t, (point_clouds_batch.shape[0],), minval=0.0, maxval=1.0)

        # Time transport_plan_jit operation
        ot_matrix = self.transport_plan_jit([noise_samples, noise_weights], 
                                           [point_clouds_batch, weights_batch])[0]

        if self.monge_map == 'sample':
            # For sampling assignment, we need the random keys
            assigned_points = self.sample_map_jit(ot_matrix, point_clouds_batch, random.split(key, point_clouds_batch.shape[0]))
        elif self.monge_map == 'random':
            assigned_points = self.sample_map_jit(ot_matrix, point_clouds_batch, random.split(key, point_clouds_batch.shape[0]))
        elif(self.monge_map == 'entropic'):
            assigned_points = self.sample_map_jit(ot_matrix, noise_samples, point_clouds_batch)
        else:
            # For other methods (rounded_matching and barycentric), we don't need keys
            assigned_points = self.sample_map_jit(ot_matrix, point_clouds_batch)

        point_cloud_interpolates = self.interpolant_vmap(noise_samples, assigned_points, 1-interpolates_time)
        point_cloud_velocity = self.interpolant_velocity_vmap(noise_samples, assigned_points, 1-interpolates_time)
        # Time interpolation computation

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
            error = self.loss_func_vmap(predicted_flow, -point_cloud_velocity, point_cloud_interpolates) * noise_weights
            loss = jnp.mean(jnp.sum(error, axis=1))
            return loss

        # Time backpropagation
        loss, grads = jax.value_and_grad(loss_fn)(state.params)
        state = state.apply_gradients(grads=grads)

        return state, loss#, point_cloud_velocity, point_cloud_interpolates



    def sample_single_batch(self, single_batch, single_weights, key, n_points):
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
        
        total_weight = jnp.sum(sample_weights)
        sample_weights = jnp.where(total_weight > 0, sample_weights / total_weight, sample_weights)
        
        return [sampled_pc, sample_weights]

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
    ):
        """
        Set up optimization parameters and train the ENVI moodel


        :param training_steps: (int) number of gradient descent steps to train ENVI (default 10000)
        :param batch_size: (int) size of train-set point clouds sampled for each training step  (default 16)
        :param verbose: (int) amount of steps between each loss print statement (default 8)
        :param init_lr: (float) initial learning rate for ADAM optimizer with exponential decay (default 1e-4)
        :param decay_num: (int) number of times of learning rate decay during training (default 4)
        :param key: (jax.random.key) random seed (default jax.random.key(0))

        :return: nothing
        """



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
                    a = self.point_clouds.shape[0],
                    shape=[batch_size])
                
                point_clouds_batch, weights_batch = self.point_clouds[batch_ind],  self.weights[batch_ind]
                
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
                self.losses.append(loss) 


                if(training_step % verbose == 0):
                    tq.set_description(": {:.3e}".format(loss))
        except KeyboardInterrupt:
            print("\nTraining interrupted by user. The model state has been saved.")


    
    def load_train_model(self, path):
        """
        Load a pre-trained train state into the model


        :param path to params

        :return: nothing
        """ 

        self.FlowMatchingModel = AttentionNN(config = self.config)
        with open(path, 'rb') as f:
            self.params = pickle.load(f)

    @partial(jit, static_argnums=(0,))
    def get_flow(self, params, point_clouds, weights, t, dt, conditioning=None):
        if point_clouds.ndim == 2:
            point_clouds = point_clouds[None, :, :]
            weights = weights[None, :]

        if conditioning is not None:
            cond_flow = jnp.squeeze(self.FlowMatchingModel.apply(
                {"params": params},
                point_cloud=point_clouds,
                t=t * jnp.ones(point_clouds.shape[0]),
                masks=weights > 0,
                conditioning=conditioning,
                is_null_conditioning=jnp.zeros(point_clouds.shape[0], dtype=bool),
                deterministic=True
            ))

            if(self.cfg):
                uncond_flow = jnp.squeeze(self.FlowMatchingModel.apply(
                    {"params": params},
                    point_cloud=point_clouds,
                    t=t * jnp.ones(point_clouds.shape[0]),
                    masks=weights > 0,
                    conditioning=conditioning,
                    is_null_conditioning=jnp.ones(point_clouds.shape[0], dtype=bool),
                    deterministic=True
                ))


                flow = uncond_flow + self.w_cfg * (cond_flow - uncond_flow)
            else:
                flow = cond_flow
        else:

            flow = jnp.squeeze(self.FlowMatchingModel.apply(
                {"params": params},
                point_cloud=point_clouds,
                t=t * jnp.ones(point_clouds.shape[0]),
                masks=weights > 0,
                deterministic=True
            ))

        update = self.exponential_map_vmap(point_clouds, flow, -dt)
        return update


    def generate_samples(self, size=None, num_samples=10, timesteps=100, generate_conditioning=None, init_noise=None, key=random.key(0)):
        """
        Generate samples from the learned flow

        :param num_samples: (int) number of samples to generate (default 10)
        :param timesteps: (int) number of timesteps to generate samples (default 100)

        :return: generated samples
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
                    generate_conditioning = generate_conditioning[:, None]
                
                if generate_conditioning.shape[0] == 1 and num_samples > 1:
                     generate_conditioning = jnp.repeat(generate_conditioning, num_samples, axis=0)

                if noise_weights is None:
                    subkey, key = random.split(key)
                    noise_weights = random.choice(subkey, self.weights, shape=(num_samples,))

        subkey, key = random.split(key)

        if init_noise is not None:
            if init_noise.ndim == 2:
                init_noise = init_noise[None, :, :]
            noise = init_noise
        else:
            noise = self.noise_func(size=[num_samples, size, self.space_dim],
                                    noise_config=self.noise_config,
                                    key=subkey)
            if len(noise) == 2:
                noise, noise_weights = noise
        
        # fix size of noise and noise_weights to the max size of noise_weights.sum(axis=1)

        max_size = int(jnp.max(jnp.sum(noise_weights > 0, axis=1)))
        
        # reorder noise and noise_weights to have the valid points first
        def reorder_points(single_noise, single_weights):
            sorted_indices = jnp.argsort(single_weights > 0)[::-1]
            reordered_noise = jnp.take(single_noise, sorted_indices, axis=0)
            reordered_weights = jnp.take(single_weights, sorted_indices, axis=0)
            return reordered_noise[:max_size, :], reordered_weights[:max_size]
        
        noise, noise_weights = jax.vmap(reorder_points, in_axes=(0, 0))(noise, noise_weights)

        dt = 1 / timesteps

        def step_fn(carry, t):
            current_noise = carry
            next_noise = self.get_flow(self.params, current_noise, noise_weights, t, dt, generate_conditioning)
            return next_noise, next_noise

        timesteps_array = jnp.linspace(1, dt, timesteps)
        _, all_noises = jax.lax.scan(step_fn, noise, timesteps_array)

        if generate_conditioning is None:
            return all_noises, noise_weights
        return all_noises, noise_weights, generate_conditioning