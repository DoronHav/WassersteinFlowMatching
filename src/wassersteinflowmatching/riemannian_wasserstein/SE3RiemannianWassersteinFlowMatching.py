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
from src.wassersteinflowmatching.riemannian_wasserstein._utils_GeomTransformer import SE3AttentionNN # type: ignore
from src.wassersteinflowmatching.riemannian_wasserstein.DefaultConfig import DefaultConfig # type: ignore
from src.wassersteinflowmatching.riemannian_wasserstein._utils_Processing import pad_pointclouds # type: ignore
from src.wassersteinflowmatching.riemannian_wasserstein.RiemannianWassersteinFlowMatching import RiemannianWassersteinFlowMatching # type: ignore

class SE3RiemannianWassersteinFlowMatching(RiemannianWassersteinFlowMatching):
    def __init__(self, point_clouds, conditioning=None, config=DefaultConfig, **kwargs):
        super().__init__(point_clouds=point_clouds, conditioning=conditioning, config=config, **kwargs)
        # Additional initialization specific to SE(3) can be added here
        self.geom = 'se3'

    @partial(jit, static_argnums=(0,))
    def train_step(self, state, point_clouds_batch, weights_batch, res_mask_batch, conditioning_batch=None, noise_samples=None, noise_weights=None, is_null_conditioning=None, key=random.key(0)):
        subkey_t, subkey_noise, key = random.split(key, 3)

        if noise_samples is None:
            noise_samples = self.noise_func(size=point_clouds_batch.shape, 
                                            noise_config=self.noise_config,
                                            key=subkey_noise)
            if len(noise_samples) == 3:
                noise_samples, noise_weights, noise_mask = noise_samples
            else:
                noise_weights = weights_batch
                noise_res_mask = res_mask_batch
            
            noise_samples = self.project_to_geometry(noise_samples)

        if self.mini_batch_ot_mode:
            # Time minibatch_ot operation
            minibatch_key, key = random.split(key)
            noise_ind = self.minibatch_ot(point_clouds_batch, weights_batch, noise_samples, noise_weights, key=minibatch_key)[0]
            noise_samples = noise_samples[noise_ind]
            
            if (self.monge_map != 'rounded_matching'):
                noise_weights = noise_weights[noise_ind]

        # Time random.uniform for interpolates_time
        interpolates_time = random.uniform(subkey_t, (point_clouds_batch.shape[0],), minval=0.0, maxval=1.0)

        # Time transport_plan_jit operation
        # TODO: clarify if res_mask is needed here
        ot_matrix = self.transport_plan_jit([noise_samples, noise_weights], 
                                           [point_clouds_batch, weights_batch])[0]

        # TODO: clarify if res_mask is needed here
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
            
            error = self.loss_func_vmap(predicted_flow, -point_cloud_velocity, point_cloud_interpolates) # B, N_timesteps, N_res_max
            error = error * noise_weights[:, :, None] * res_mask_batch[:, None, :] # # B, N_timesteps, N_res_max
            loss = jnp.mean(jnp.sum(error, axis=1))
            return loss

        # Time backpropagation
        loss, grads = jax.value_and_grad(loss_fn)(state.params)
        state = state.apply_gradients(grads=grads)

        return state, loss
    
    def sample_single_batch(self, single_batch, single_weights, single_res_masks, key, n_points):
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
        sample_res_masks = jnp.take(single_res_masks, indices, axis=0)
        
        total_weight = jnp.sum(sample_weights)
        sample_weights = jnp.where(total_weight > 0, sample_weights / total_weight, sample_weights)
        
        return [sampled_pc, sample_weights, sample_res_masks]
    
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
                
                point_clouds_batch, weights_batch, res_mask_batch = self.point_clouds[batch_ind],  self.weights[batch_ind], self.res_masks[batch_ind]
                
                if (self.matched_noise):
                    noise_samples, noise_weights, noise_res_mask = self.noise_point_clouds[batch_ind], self.noise_weights[batch_ind], self.res_masks[batch_ind]
                    
                    if(source_sample is not None):
                        keys = jax.random.split(subkey, batch_size)
                        noise_samples, noise_weights = sample_points(noise_samples, noise_weights, noise_res_mask, keys, source_sample)
                else:
                    noise_samples, noise_weights = None, None
                
                if(shape_sample is not None):
                    keys = jax.random.split(subkey, batch_size)
                    point_clouds_batch, weights_batch, res_mask_batch = sample_points(point_clouds_batch, weights_batch, res_mask_batch, keys, shape_sample)
                    
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

                self.state, loss = self.train_step(self.state, point_clouds_batch, weights_batch, res_mask_batch, conditioning_batch, noise_samples, noise_weights, is_null_conditioning, key = subkey)

                self.params = self.state.params
                self.losses.append(loss) 

                if (training_step % verbose == 0):
                    tq.set_description(": {:.3e}".format(loss))
        
        except KeyboardInterrupt:
            print("\nTraining interrupted by user. The model state has been saved.")