from functools import partial

import jax # type: ignore
import jax.numpy as jnp # type: ignore
import numpy as np  # type: ignore
import optax # type: ignore
from jax import jit, random# type: ignore
from tqdm import trange, tqdm # type: ignore
from flax.training import train_state # type: ignore

import wassersteinflowmatching.utils_OT as utils_OT # type: ignore
import wassersteinflowmatching.utils_Noise as utils_Noise # type: ignore
from wassersteinflowmatching._utils_Transformer import AttentionNN # type: ignore
from wassersteinflowmatching.DefaultConfig import DefaultConfig # type: ignore


def lower_tri_to_square(v, n):
    """
    :meta private:
    """
    
    # # Initialize the square matrix with zeros
    # mat = jnp.zeros((n, n))
    
    # # Fill the lower triangular part of the matrix (including the diagonal) with the vector elements
    # mat[jnp.tril_indices(n)] = v
    
    # # Since it's symmetric, copy the lower triangular part to the upper triangular part
    # mat = mat + mat.T - jnp.diag(jnp.diag(mat))
    
    # Create an empty lower triangular matrix

    idx = np.tril_indices(n)
    mat = jnp.zeros((n, n), dtype=v.dtype).at[idx].set(v)
    mat = mat + mat.T - jnp.diag(jnp.diag(mat))
    return mat



def pad_pointclouds(point_clouds, weights, max_shape=-1):
    """
    :meta private:
    """

    if max_shape == -1:
        max_shape = np.max([pc.shape[0] for pc in point_clouds]) + 1
    else:
        max_shape = max_shape + 1


    weights_pad = np.asarray(
        [
            np.concatenate((weight, np.zeros(max_shape - pc.shape[0])), axis=0)
            for pc, weight in zip(point_clouds, weights)
        ]
    )
    point_clouds_pad = np.asarray(
        [
            np.concatenate(
                [pc, np.zeros([max_shape - pc.shape[0], pc.shape[-1]])], axis=0
            )
            for pc in point_clouds
        ]
    )

    weights_pad = weights_pad / weights_pad.sum(axis=1, keepdims=True)

    return (
        point_clouds_pad[:, :-1].astype("float32"),
        weights_pad[:, :-1].astype("float32"),
    )



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
        noise_type = 'normal',
        config = DefaultConfig,
    ):


        self.config = config
        self.point_clouds = point_clouds


        self.weights = [
            np.ones(pc.shape[0]) / pc.shape[0] for pc in self.point_clouds
        ]


        self.point_clouds, self.weights = pad_pointclouds(
            self.point_clouds, self.weights
        )

        self.space_dim = self.point_clouds.shape[-1]


        self.transport_plan_jit = jax.jit(
            jax.vmap(utils_OT.transport_plan, (0, 0, None, None), 0),
            static_argnums=[2, 3],
        )



        self.scaling = config.scaling
        self.factor = config.factor
        self.point_clouds = self.scale_func(self.point_clouds) * self.factor
        self.max_val, self.min_val = self.point_clouds.max(), self.point_clouds.min()

        self.noise_type = noise_type
        self.noise_func = getattr(utils_Noise, self.noise_type)

        self.mini_batch_ot_mode = config.mini_batch_ot_mode

        if(self.mini_batch_ot_mode):
            self.ot_mat_jit = jax.jit(
                jax.vmap(utils_OT.ot_mat, (0, 0, None, None), 0),
                static_argnums=[2, 3],
            )
            self.minibatch_ot_eps = config.minibatch_ot_eps
            self.minibatch_ot_lse = config.minibatch_ot_lse

        if(labels is not None):

            self.label_to_num = {label: i for i, label in enumerate(np.unique(labels))}
            self.num_to_label = {i: label for i, label in enumerate(np.unique(labels))}
            self.labels = jnp.array([self.label_to_num[label] for label in labels])
            self.label_dim = len(np.unique(labels))
            self.config.label_dim = self.label_dim 
        else:
            self.labels = None
            self.label_dim = -1

    def scale_func(self, point_clouds):
        """
        :meta private:
        """

        if self.scaling == "min_max_total":
            if not hasattr(self, "max_val"):
                self.max_val_scale = self.point_clouds.max(keepdims=True)
                self.min_val_scale = self.point_clouds.min(keepdims=True)
            else:
                print("Using Calculated Min Max Scaling Values")
            return 2 * (point_clouds - self.min_val_scale) / (self.max_val_scale - self.min_val_scale) - 1
        return point_clouds


    def create_train_state(self, model, learning_rate, decay_steps = 10000, key = random.key(0)):
        """
        :meta private:
        """

        subkey, key = random.split(key)
        attn_inputs =  self.noise_func(size = [10, self.point_clouds[0].shape[0], self.space_dim], 
                                       minval = self.min_val, 
                                       maxval = self.max_val, key = subkey)
        
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
            learning_rate, decay_steps, 0.6, staircase = True,
        )
        tx = optax.adam(lr_sched)  #

        return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)


    @partial(jit, static_argnums=(0,))
    def minibatch_ot(self, point_clouds, point_cloud_weights, noise, noise_weights, key = random.key(0)):

        """
        :meta private:
        """
            
        tri_u_ind = jnp.stack(jnp.triu_indices(point_clouds.shape[0]), axis=1)

        # compute pairwise ot between point clouds and noise:

        ot_matrix = lower_tri_to_square(self.ot_mat_jit(
                    [point_clouds[tri_u_ind[:, 0]], point_cloud_weights[tri_u_ind[:, 0]]],
                    [noise[tri_u_ind[:, 1]], noise_weights[tri_u_ind[:, 1]]],
                    self.minibatch_ot_eps,
                    self.minibatch_ot_lse,
                ), n = point_clouds.shape[0])

        pairing_matrix = utils_OT.ot_mat_from_distance(ot_matrix, self.minibatch_ot_eps, self.minibatch_ot_lse)
        
        subkey, key = random.split(key)
        noise_ind = random.categorical(subkey, logits = jnp.log(pairing_matrix + 0.000001))
        return(noise_ind)

    @partial(jit, static_argnums=(0,))
    def train_step(self, state, point_clouds_batch, weights_batch, labels_batch = None, key=random.key(0)):
        """
        :meta private:
        """
        subkey_t, subkey_noise, key = random.split(key, 3)
        
        noise_samples =  self.noise_func(size = point_clouds_batch.shape, 
                                minval = self.min_val, 
                                maxval = self.max_val, key = subkey_noise)
        noise_weights = weights_batch

        if(self.mini_batch_ot_mode):
            subkey_resample, key = random.split(key)
            noise_ind = self.minibatch_ot(point_clouds_batch, weights_batch, noise_samples, noise_weights, key = subkey_resample)
            noise_samples = noise_samples[noise_ind]
            noise_weights = noise_weights[noise_ind]

        interpolates_time = random.uniform(subkey_t, (point_clouds_batch.shape[0],), minval=0.0, maxval=1.0)
        
        optimal_flow = jnp.nan_to_num(self.transport_plan_jit([noise_samples, noise_weights], 
                                                              [point_clouds_batch, weights_batch], 
                                                              self.config.wasserstein_eps, 
                                                              self.config.wasserstein_lse), neginf=0)
        
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

    def train(
        self,
        training_steps=32000,
        batch_size=16,
        verbose=8,
        init_lr=0.0001,
        decay_num=4,
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

        self.FlowMatchingModel = AttentionNN(config = self.config)
        self.state = self.create_train_state(self.FlowMatchingModel, 
                                             learning_rate=init_lr, 
                                             decay_steps = int(training_steps / decay_num), 
                                             key = subkey)


        tq = trange(training_steps, leave=True, desc="")
        losses = []
        for training_step in tq:

            subkey, key = random.split(key, 2)
            batch_ind = random.choice(
                key=subkey,
                a = self.point_clouds.shape[0],
                shape=[batch_size])
            

            point_clouds_batch, weights_batch = self.point_clouds[batch_ind],  self.weights[batch_ind]

            subkey, key = random.split(key, 2)
            if(self.labels is not None):
                labels_batch = self.labels[batch_ind]
            else:
                labels_batch = None
            self.state, loss = self.train_step(self.state, point_clouds_batch, weights_batch, labels_batch, key = subkey)
            losses.append(loss) 

            if(training_step % verbose == 0):
                tq.set_description(": {:.3e}".format(loss))

    @partial(jit, static_argnums=(0,))
    def get_flow(self, point_clouds, weights, t, labels = None):

        if(point_clouds.ndim == 2):
            point_clouds = point_clouds[None,:, :]

        flow = jnp.squeeze(self.FlowMatchingModel.apply({"params": self.state.params},
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
            noise =  [self.noise_func(size = [num_samples, size, self.space_dim], 
                                    minval = self.min_val, 
                                    maxval = self.max_val, key = subkey)]


        dt = 1/timesteps

        for t in tqdm(jnp.linspace(1, 0, timesteps)):
            grad_fn = self.get_flow(noise[-1], noise_weights, t, generate_labels)
            noise.append(noise[-1] + dt * grad_fn)
        if(generate_labels is None):
            return noise, noise_weights
        return noise, noise_weights, [self.num_to_label[l] for l in np.array(generate_labels)]