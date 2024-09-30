from functools import partial
import types
import time

import jax # type: ignore
import jax.numpy as jnp # type: ignore
import numpy as np  # type: ignore
import optax # type: ignore
from jax import jit, random# type: ignore
from tqdm import trange, tqdm # type: ignore
from flax.training import train_state # type: ignore
import pickle # type: ignore

import wassersteinflowmatching.utils_OT as utils_OT # type: ignore
import wassersteinflowmatching.utils_Noise as utils_Noise # type: ignore
from wassersteinflowmatching._utils_Transformer import AttentionNN # type: ignore
from wassersteinflowmatching.DefaultConfig import DefaultConfig # type: ignore
from wassersteinflowmatching._utils_Processing import pad_pointclouds # type: ignore

class WassersteinFlowMatching:
    """
    Initializes Wormhole model and processes input point clouds


    :param point_clouds: (list of np.array) list of train-set point clouds to flow match
    :param config: (flax struct.dataclass) object with parameters

    :return: initialized WassersteinFlowMatching model
    """

    def __init__(
        self,
        point_clouds,
        labels = None,
        noise_point_clouds = None,
        matched_noise = False,
        config = DefaultConfig,
    ):


        
    
        
        self.config = config
        self.scaling = self.config.scaling
        self.factor = self.config.factor

        self.point_clouds = self.scale_func(point_clouds)


        self.weights = [
            np.ones(pc.shape[0]) / pc.shape[0] for pc in self.point_clouds
        ]

        self.point_clouds, self.weights = pad_pointclouds(
            self.point_clouds, self.weights
        )

        self.space_dim = self.point_clouds.shape[-1]
        self.monge_map = self.config.monge_map
        self.num_sinkhorn_iters = self.config.num_sinkhorn_iters

        if(self.monge_map == 'entropic'):
            print(f"Using entropic map with {self.num_sinkhorn_iters} iterations and {self.config.wasserstein_eps} epsilon")
            self.transport_plan_jit = jax.jit(
                jax.vmap(utils_OT.transport_plan_entropic, (0, 0, None, None, None), 0),
                static_argnums=[2, 3, 4],
            )
        elif(self.monge_map == 'row_iter'):
            print(f"Using row_iter map with {self.num_sinkhorn_iters} iterations and {self.config.wasserstein_eps} epsilon")
            self.transport_plan_jit = jax.jit(
                jax.vmap(utils_OT.transport_plan_rowiter, (0, 0, None, None, None), 0),
                static_argnums=[2, 3, 4],
            )
        elif(self.monge_map == 'sample'):
            print(f"Using sampled map with {self.num_sinkhorn_iters} iterations and {self.config.wasserstein_eps} epsilon")
            self.transport_plan_jit = jax.jit(
                jax.vmap(utils_OT.transport_plan_sample, (0, 0, None, None, None), 0),
                static_argnums=[2, 3, 4],
            )
            self.sample_map_jit = jax.jit(jax.vmap(utils_OT.sample_ot_matrix, (0, 0, 0, 0), 0))
        elif(self.monge_map == 'euclidean'):
            print("Using euclidean Monge map")
            self.transport_plan_jit = jax.jit(
                jax.vmap(utils_OT.transport_plan_euclidean, (0, 0, None, None, None), 0),
                static_argnums=[2, 3, 4],
            )
        else:
            print(f"Using argmax map with {self.num_sinkhorn_iters} iterations and {self.config.wasserstein_eps} epsilon")
            self.transport_plan_jit = jax.jit(
                jax.vmap(utils_OT.transport_plan_argmax, (0, 0, None, None, None), 0),
                static_argnums=[2, 3, 4],
            )

        self.noise_config = types.SimpleNamespace()
        if(noise_point_clouds is not None):
            self.noise_point_clouds = self.scale_func(noise_point_clouds)
            self.noise_weights = [
                np.ones(pc.shape[0]) / pc.shape[0] for pc in self.noise_point_clouds
            ]
            self.noise_point_clouds, self.noise_weights = pad_pointclouds(
                self.noise_point_clouds, self.noise_weights
            )

            self.noise_config.noise_point_clouds = self.noise_point_clouds
            self.noise_config.noise_weights = self.noise_weights
            self.matched_noise = matched_noise
            self.noise_func = utils_Noise.random_pointclouds
            self.config.mini_batch_ot_mode = not self.matched_noise

        else:

            self.min_val = self.point_clouds[self.weights > 0].min()
            self.max_val = self.point_clouds[self.weights > 0].max()

            self.noise_config.maxval = self.point_clouds[self.weights > 0].max()
            self.noise_config.minval = self.point_clouds[self.weights > 0].min()

            self.noise_type = self.config.noise_type
            self.noise_func = getattr(utils_Noise, self.noise_type)
            self.matched_noise = False 

            if(self.noise_type == 'meta_normal'):
                self.point_clouds_mean, self.point_clouds_cov = utils_OT.weighted_mean_and_covariance(self.point_clouds, self.weights)
                self.covariance_barycenter = utils_OT.covariance_barycenter(self.point_clouds_cov, max_iter = 100, tol = 1e-6)
                self.noise_config.covariance_barycenter_chol = jnp.linalg.cholesky(self.covariance_barycenter)
                self.noise_config.noise_df_scale = self.config.noise_df_scale
            if(self.noise_type == 'chol_normal'):
                self.point_clouds_mean, self.point_clouds_cov = utils_OT.weighted_mean_and_covariance(self.point_clouds, self.weights)
                self.cov_chol = jax.vmap(jnp.linalg.cholesky)(self.point_clouds_cov)
                self.noise_config.cov_chol_mean = jnp.mean(self.cov_chol, axis = 0)
                self.noise_config.cov_chol_std = jnp.std(self.cov_chol, axis = 0)
                self.noise_config.noise_df_scale = self.config.noise_df_scale
 

        self.mini_batch_ot_mode = self.config.mini_batch_ot_mode


        if(labels is not None):
            self.label_to_num = {label: i for i, label in enumerate(np.unique(labels))}
            self.num_to_label = {i: label for i, label in enumerate(np.unique(labels))}
            self.labels = jnp.array([self.label_to_num[label] for label in labels])
            self.label_dim = len(np.unique(labels))
            self.config.label_dim = self.label_dim 
            self.mini_batch_ot_mode = False
        else:
            self.labels = None
            self.label_dim = -1


        if(self.mini_batch_ot_mode):
            self.mini_batch_ot_solver = self.config.mini_batch_ot_solver
            if(self.mini_batch_ot_solver == 'entropic'):
                print("Entropic Mini-Batch")
                self.ot_mat_jit = jax.jit(
                    jax.vmap(utils_OT.entropic_ot_distance, (0, 0, None, None), 0),
                    static_argnums=[2, 3],
                )
            elif(self.mini_batch_ot_solver == 'euclidean'):
                print("Euclidean Mini-Batch")
                self.ot_mat_jit = jax.jit(
                    jax.vmap(utils_OT.euclidean_distance, (0, 0, None, None), 0),
                    static_argnums=[2, 3],
                )
            else:
                print("Frechet Mini-Batch")
                self.ot_mat_jit = jax.jit(
                    jax.vmap(utils_OT.frechet_distance, (0, 0, None, None), 0),
                    static_argnums=[2, 3],
                )
            self.minibatch_ot_eps = self.config.minibatch_ot_eps
            self.minibatch_ot_lse = self.config.minibatch_ot_lse

        self.FlowMatchingModel = AttentionNN(config = self.config)

    def scale_func(self, point_clouds):
        """
        :meta private:
        """

        if self.scaling == "min_max_total":
            if(hasattr(self, 'max_val_scale')):
                return [self.factor * (2 * ((pc - self.min_val_scale) / (self.max_val_scale - self.min_val_scale)) - 1) for pc in point_clouds]
            else:
                self.max_val_scale = np.max([np.max(pc) for pc in point_clouds])
                self.min_val_scale = np.min([np.min(pc) for pc in point_clouds])
                return [self.factor * (2 * ((pc - self.min_val_scale) / (self.max_val_scale - self.min_val_scale)) - 1) for pc in point_clouds]
        if self.scaling == "min_max_each":
            point_clouds = [self.factor * (2 * (pc - pc.min(keepdims=True)) / (pc.max(keepdims=True) - pc.min(keepdims=True)) - 1) for pc in point_clouds]
        return point_clouds
    

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

        if(self.labels is not None):
            params = model.init(rngs={"params": subkey}, 
                                point_cloud = attn_inputs, 
                                t = jnp.ones((attn_inputs.shape[0])), 
                                masks = jnp.ones((attn_inputs.shape[0], attn_inputs.shape[1])),
                                labels =  jnp.ones((attn_inputs.shape[0])),
                                deterministic = True)['params']
        else:
            params = model.init(rngs={"params": subkey}, 
                    point_cloud = attn_inputs, 
                    t = jnp.ones((attn_inputs.shape[0])), 
                    masks = jnp.ones((attn_inputs.shape[0], attn_inputs.shape[1])),
                    deterministic = True)['params']

    

        lr_sched = optax.exponential_decay(
            learning_rate, decay_steps, 0.97, staircase = False,
        )

        tx = optax.adam(lr_sched)  #

        return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)


    @partial(jit, static_argnums=(0,))
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
                                        [mean_y[matrix_ind[:, 1]], cov_y[matrix_ind[:, 1]]], 
                                        self.minibatch_ot_eps, self.minibatch_ot_lse).reshape(point_clouds.shape[0], noise.shape[0])
        else:
            ot_matrix = self.ot_mat_jit([point_clouds[matrix_ind[:, 0]], point_cloud_weights[matrix_ind[:, 0]]],
                                        [noise[matrix_ind[:, 1]], noise_weights[matrix_ind[:, 1]]],
                                        self.minibatch_ot_eps,
                                        self.minibatch_ot_lse).reshape(point_clouds.shape[0], noise.shape[0])

        noise_ind = utils_OT.ot_mat_from_distance(ot_matrix, 0.002, True)
        return(noise_ind)

    
    @partial(jit, static_argnums=(0,))
    def train_step(self, state, point_clouds_batch, weights_batch, labels_batch = None, noise_samples = None, noise_weights = None, key=random.key(0)):
        """
        :meta private:
        """
        subkey_t, subkey_noise, key = random.split(key, 3)
        
        if(noise_samples is None):
            noise_samples = self.noise_func(size = point_clouds_batch.shape, 
                                            noise_config = self.noise_config,
                                            key = subkey_noise)
            if(len(noise_samples) == 2):
                noise_samples, noise_weights = noise_samples
            else:
                noise_weights = weights_batch
        
        
        if(self.mini_batch_ot_mode):
            minibatch_key, key = random.split(key)
            noise_ind = self.minibatch_ot(point_clouds_batch, weights_batch, noise_samples, noise_weights, key = minibatch_key)
            noise_samples = noise_samples[noise_ind]
            noise_weights = noise_weights[noise_ind]

        interpolates_time = random.uniform(subkey_t, (point_clouds_batch.shape[0],), minval=0.0, maxval=1.0)
        
        optimal_flow = jnp.nan_to_num(self.transport_plan_jit([noise_samples, noise_weights], 
                                                              [point_clouds_batch, weights_batch], 
                                                              self.config.wasserstein_eps, 
                                                              self.config.wasserstein_lse,
                                                              self.num_sinkhorn_iters), neginf=0)

        if(self.monge_map == 'sample'):
            optimal_flow = self.sample_map_jit(noise_samples, point_clouds_batch, optimal_flow, random.split(key, point_clouds_batch.shape[0]))
        
        point_cloud_interpolates = noise_samples + (1-interpolates_time[:, None, None]) * optimal_flow

        
        subkey, key = random.split(key)
        def loss_fn(params):       
            predicted_flow = state.apply_fn({"params": params},  
                                            point_cloud = point_cloud_interpolates, 
                                            t = interpolates_time, 
                                            masks = noise_weights>0, 
                                            labels = labels_batch,
                                            deterministic = False, 
                                            dropout_rng = subkey)
            error = jnp.square(predicted_flow - optimal_flow) * noise_weights[:, :, None]
            loss = jnp.mean(jnp.sum(error, axis = 1))
            return loss
        
        loss, grads = jax.value_and_grad(loss_fn)(state.params)
        state = state.apply_gradients(grads=grads)
        return (state, loss)

    def sample_single_batch(self, single_batch, single_weights, key, n_points):
        indices = jax.random.choice(key, single_batch.shape[0], (n_points,), replace=False)
        sampled_pc = jnp.take(single_batch, indices, axis=0)
        sample_weights = jnp.take(single_weights, indices, axis=0)
        sample_weights = sample_weights / jnp.sum(sample_weights)
        
        return [sampled_pc, sample_weights]

    def train(
        self,
        training_steps=32000,
        batch_size=16,
        verbose=8,
        learning_rate = 3e-4,
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
        :param init_lr: (float) initial learning rate for ADAM optimizer with exponential decay (default 1e-4)
        :param decay_num: (int) number of times of learning rate decay during training (default 4)
        :param key: (jax.random.key) random seed (default jax.random.key(0))

        :return: nothing
        """



        subkey, key = random.split(key)

                
        if saved_state is None:
            self.state = self.create_train_state(
                model=self.FlowMatchingModel,
                learning_rate=learning_rate,
                decay_steps=decay_steps,
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
        for training_step in tq:

            subkey, key = random.split(key, 2)
            batch_ind = random.choice(
                key=subkey,
                a = self.point_clouds.shape[0],
                shape=[batch_size])
            
            point_clouds_batch, weights_batch = self.point_clouds[batch_ind],  self.weights[batch_ind]
            
            if(self.matched_noise):
                noise_samples, noise_weights = self.noise_point_clouds[batch_ind], self.noise_weights[batch_ind]
            else:
                noise_samples, noise_weights = None, None

            
            if(shape_sample is not None):
                keys = jax.random.split(subkey, batch_size)
                point_clouds_batch, weights_batch = sample_points(point_clouds_batch, weights_batch, keys, shape_sample)
                
            if(self.labels is not None):
                labels_batch = self.labels[batch_ind]
                
            else:
                labels_batch = None

            subkey, key = random.split(key, 2)

            self.state, loss = self.train_step(self.state, point_clouds_batch, weights_batch, labels_batch, noise_samples, noise_weights, key = subkey)

            self.params = self.state.params
            self.losses.append(loss) 

            if(training_step % verbose == 0):
                tq.set_description(": {:.3e}".format(loss))

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
    def get_flow(self, params, point_clouds, weights, t, labels = None):

        if(point_clouds.ndim == 2):
            point_clouds = point_clouds[None,:, :]
            weights = weights[None, :]

        flow = jnp.squeeze(self.FlowMatchingModel.apply({"params": params},
                    point_cloud = point_clouds, 
                    t = t * jnp.ones(point_clouds.shape[0]), 
                    masks = weights>0, 
                    labels = labels,
                    deterministic = True))
        return(flow)
        

    def generate_samples(self, size = None, num_samples = 10, timesteps = 100, generate_labels = None, init_noise = None, key = random.key(0)): 
        """
        Generate samples from the learned flow


        :param num_samples: (int) number of samples to generate (default 10)
        :param timesteps: (int) number of timesteps to generate samples (default 100)

        :return: generated samples
        """ 
        if(size is None):
            size = self.point_clouds.shape[1]
            noise_weights = None
        else:
            noise_weights = jnp.ones([num_samples, size])

        if(self.labels is None):
            generate_labels = None
            if(noise_weights is None):
                subkey, key = random.split(key)
                noise_weights = random.choice(subkey, self.weights, [num_samples])
        else:
            if(generate_labels is None):
                generate_labels = random.choice(key, self.label_dim, [num_samples], replace = True)
            elif(isinstance(generate_labels, (str, int))):
                generate_labels = jnp.array([self.label_to_num[generate_labels]] * num_samples)
            else:
                generate_labels = jnp.array([self.label_to_num[label] for label in generate_labels])
            
            if(noise_weights is None):
                noise_weights = []
                for label in generate_labels:
                    subkey, key = random.split(key)
                    noise_weights.append(random.choice(subkey, self.weights[self.labels == label]))
                noise_weights = jnp.vstack(noise_weights)
        subkey, key = random.split(key)

        if(init_noise is not None):
            if(init_noise.ndim == 2):
                init_noise = init_noise[None, :, :]
            noise = [init_noise]
        else:

            # noise = self.noise_func(size =[num_samples, size, self.space_dim], 
            #             minval = self.min_val, 
            #             maxval = self.max_val, key = subkey)


            noise = self.noise_func(size = [num_samples, size, self.space_dim], 
                                      noise_config = self.noise_config,
                                      key = subkey)
            if(len(noise) == 2):
                noise, noise_weights = noise
            noise =  [noise]


        dt = 1/timesteps

        for t in tqdm(jnp.linspace(1, 0, timesteps)):
            grad_fn = self.get_flow(self.params, noise[-1], noise_weights, t, generate_labels)
            noise.append(noise[-1] + dt * grad_fn)
        if(generate_labels is None):
            return noise, noise_weights
        return noise, noise_weights, [self.num_to_label[l] for l in np.array(generate_labels)]