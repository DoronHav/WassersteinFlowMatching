from functools import partial
import types
import pickle

import jax  # type: ignore
import jax.numpy as jnp  # type: ignore
import numpy as np  # type: ignore
import optax  # type: ignore
from jax import jit, random  # type: ignore
from tqdm import trange, tqdm  # type: ignore
from flax.training import train_state  # type: ignore

import wassersteinflowmatching.wasserstein.utils_OT as utils_OT  # type: ignore
import wassersteinflowmatching.wasserstein.utils_Noise as utils_Noise  # type: ignore
from wassersteinflowmatching.wasserstein._utils_Processing import pad_pointclouds  # type: ignore
from wassersteinflowmatching.autoregressivewasserstein._utils_Networks import ARFlowModel, CausalContextEncoder, FlowMLP
from wassersteinflowmatching.autoregressivewasserstein.DefaultConfig import ARWFMConfig


class AutoRegressiveWFM:
    """Auto-Regressive Wasserstein Flow Matching.

    Generates point clouds one point at a time. At step k the model integrates
    noise z^k → x^k using a flow conditioned on all previously generated points
    C = {x^1, ..., x^{k-1}}.

    OT coupling is computed at the cloud level (pairing the full noise cloud to
    the target cloud) before the per-point AR training selection takes place.

    :param point_clouds: list of np.array, each of shape (n_i, d)
    :param config: ARWFMConfig (or subclass thereof)
    :param key: JAX random key
    """

    def __init__(
        self,
        point_clouds,
        config=ARWFMConfig,
        key=random.key(0),
        **kwargs,
    ):
        print(config)
        print("Initializing Auto-Regressive Wasserstein Flow Matching")

        if isinstance(config, type):
            config = config()
        if config is None:
            config = ARWFMConfig()
        if kwargs:
            config = config.replace(**kwargs)

        self.config = config

        self.point_clouds = point_clouds
        self.weights = [
            np.ones(pc.shape[0]) / pc.shape[0] for pc in self.point_clouds
        ]
        self.point_clouds, self.weights = pad_pointclouds(self.point_clouds, self.weights)

        self.space_dim = self.point_clouds.shape[-1]
        self.cloud_size = self.point_clouds.shape[1]  # N (after padding)

        self._setup_noise(key)
        self._setup_transport_plan()
        self._setup_minibatch_ot()

        self.FlowModel = ARFlowModel(config=self.config, space_dim=self.space_dim)

    # ------------------------------------------------------------------
    # Setup helpers
    # ------------------------------------------------------------------

    def _setup_noise(self, key):
        self.noise_config = types.SimpleNamespace()
        self.noise_type = self.config.noise_type
        self.noise_func = getattr(utils_Noise, self.noise_type)

        self.noise_config.maxval = self.point_clouds[self.weights > 0].max()
        self.noise_config.minval = self.point_clouds[self.weights > 0].min()

        if self.noise_type in ('chol_normal', 'meta_normal'):
            means, covs = utils_OT.weighted_mean_and_covariance(
                self.point_clouds, self.weights)
            chol = jax.vmap(jnp.linalg.cholesky)(covs)
            self.noise_config.cov_chol_mean = jnp.mean(chol, axis=0)
            self.noise_config.cov_chol_std = jnp.std(chol, axis=0)
            self.noise_config.cov_mean = np.mean(covs, axis=0)
            self.noise_config.cov_std = np.std(covs, axis=0)
            self.noise_config.noise_df_scale = self.config.noise_df_scale

    def _setup_transport_plan(self):
        monge_map = self.config.monge_map
        eps = self.config.wasserstein_eps
        lse = self.config.wasserstein_lse
        n_iter = self.config.num_sinkhorn_iters

        if monge_map == 'random':
            self.transport_plan_jit = None
        elif monge_map == 'entropic':
            self.transport_plan_jit = jax.vmap(
                partial(utils_OT.transport_plan_entropic,
                        eps=eps, lse_mode=lse, num_iteration=n_iter),
                (0, 0), 0)
        elif monge_map == 'rounded_matching':
            self.transport_plan_jit = jax.vmap(
                partial(utils_OT.transport_plan_rounded,
                        eps=eps, lse_mode=lse, num_iteration=n_iter),
                (0, 0), 0)
        elif monge_map == 'euclidean':
            self.transport_plan_jit = jax.vmap(
                utils_OT.transport_plan_euclidean, (0, 0), 0)
        else:  # argmax (default fallback)
            self.transport_plan_jit = jax.vmap(
                partial(utils_OT.transport_plan_argmax,
                        eps=eps, lse_mode=lse, num_iteration=n_iter),
                (0, 0), 0)

        print(f"Using '{monge_map}' OT coupling")

    def _setup_minibatch_ot(self):
        if not self.config.mini_batch_ot_mode:
            print("Mini-batch OT disabled")
            return

        solver = self.config.mini_batch_ot_solver
        if solver == 'entropic':
            print("Mini-batch OT: entropic")
            self.ot_mat_jit = jax.vmap(
                partial(utils_OT.entropic_ot_distance,
                        eps=self.config.minibatch_ot_eps,
                        lse_mode=self.config.minibatch_ot_lse),
                (0, 0), 0)
        elif solver == 'euclidean':
            print("Mini-batch OT: euclidean")
            self.ot_mat_jit = jax.vmap(utils_OT.euclidean_distance, (0, 0), 0)
        elif solver == 'frechet':
            print("Mini-batch OT: frechet")
            self.ot_mat_jit = jax.vmap(utils_OT.frechet_distance, (0, 0), 0)
        else:
            print("Mini-batch OT: chamfer")
            self.ot_mat_jit = jax.vmap(utils_OT.chamfer_distance, (0, 0), 0)

    def minibatch_ot(self, point_clouds, point_cloud_weights, noise, noise_weights):
        matrix_ind = jnp.array(
            jnp.meshgrid(jnp.arange(point_clouds.shape[0]), jnp.arange(noise.shape[0]))
        ).T.reshape(-1, 2)

        if self.config.mini_batch_ot_solver == 'frechet':
            mean_x, cov_x = utils_OT.weighted_mean_and_covariance(point_clouds, point_cloud_weights)
            mean_y, cov_y = utils_OT.weighted_mean_and_covariance(noise, noise_weights)
            ot_matrix = self.ot_mat_jit(
                [mean_x[matrix_ind[:, 0]], cov_x[matrix_ind[:, 0]]],
                [mean_y[matrix_ind[:, 1]], cov_y[matrix_ind[:, 1]]],
            ).reshape(point_clouds.shape[0], noise.shape[0])
        else:
            ot_matrix = self.ot_mat_jit(
                [point_clouds[matrix_ind[:, 0]], point_cloud_weights[matrix_ind[:, 0]]],
                [noise[matrix_ind[:, 1]], noise_weights[matrix_ind[:, 1]]],
            ).reshape(point_clouds.shape[0], noise.shape[0])

        noise_ind = utils_OT.ot_mat_from_distance(
            ot_matrix, self.config.minibatch_ot_eps,
            self.config.minibatch_ot_lse, self.config.num_sinkhorn_iters)
        return noise_ind

    def create_train_state(self, model, learning_rate, decay_steps, key=random.key(0)):
        subkey_params, subkey_dropout = random.split(key)

        # Use a small N for init — the model is shape-agnostic so we don't need
        # cloud_size here, and using it (e.g. 15 000) would build a huge causal
        # attention matrix just to initialise parameters.
        B, N, d = 4, min(self.cloud_size, 64), self.space_dim
        dummy_x_t      = jnp.zeros([B, N, d])
        dummy_t        = jnp.zeros([B, N])
        dummy_targets  = jnp.zeros([B, N, d])
        dummy_padding  = jnp.ones([B, N], dtype=bool)

        params = model.init(
            {"params": subkey_params, "dropout": subkey_dropout},
            x_t=dummy_x_t,
            t=dummy_t,
            target_points=dummy_targets,
            padding_mask=dummy_padding,
            deterministic=True,
        )['params']

        lr_sched = optax.exponential_decay(learning_rate, decay_steps, 0.998, staircase=False)
        tx = optax.adam(lr_sched)
        return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)

    @partial(jit, static_argnums=(0,))
    def train_step(self, state, point_clouds_batch, weights_batch, key=random.key(0)):
        """One JIT-compiled AR-WFM training step.

        Trains on all N points in a cloud simultaneously using causal attention,
        analogously to LLM training.  For each cloud in the batch:
          1. Sample a full noise cloud and compute the OT coupling.
          2. Draw a random permutation defining the AR generation order.
          3. Reorder everything by the permutation so position k always refers
             to the k-th point to be generated.
          4. Interpolate all N noise→target pairs and predict all N velocities
             in one forward pass.  The causal context encoder ensures position k
             only attends to positions 0..k-1, so no target leaks into its own
             context.
        """
        B, N, d = point_clouds_batch.shape

        key, k_noise, k_t, k_perm, k_dropout, k_ot = random.split(key, 6)

        # --- 1. Sample noise clouds ---
        noise_samples = self.noise_func(
            size=[B, N, d],
            noise_config=self.noise_config,
            key=k_noise,
        )
        if isinstance(noise_samples, tuple):
            noise_samples, noise_weights = noise_samples
        else:
            noise_weights = weights_batch

        # --- 2. Mini-batch OT: reassign noise clouds to target clouds ---
        if self.config.mini_batch_ot_mode:
            noise_ind = self.minibatch_ot(point_clouds_batch, weights_batch, noise_samples, noise_weights)
            noise_samples = noise_samples[noise_ind]
            noise_weights = noise_weights[noise_ind]

        # --- 3. Per-point coupling: flow[b, i] = target[b, match(i)] - noise[b, i] ---
        if self.config.monge_map == 'random':
            # Random permutation pairing — no OT solve
            b_idx_ot = jnp.arange(B)
            perms_ot = jax.vmap(lambda k: random.permutation(k, N))(
                random.split(k_ot, B))  # [B, N]
            target_matched = point_clouds_batch[b_idx_ot[:, None], perms_ot]  # [B, N, d]
            optimal_flow = target_matched - noise_samples                      # [B, N, d]
            batch_iters = jnp.array(0.0)
        else:
            ot_result = self.transport_plan_jit(
                [noise_samples, noise_weights],
                [point_clouds_batch, weights_batch],
            )
            optimal_flow = ot_result[0]  # [B, N, d]
            if self.config.num_sinkhorn_iters is None:
                ot_solve = ot_result[1]
                batch_iters = jnp.max(
                    jnp.sum(ot_solve.errors > -1, axis=-1) * jnp.mean(ot_solve.inner_iterations))
            else:
                batch_iters = jnp.array(float(self.config.num_sinkhorn_iters))
            target_matched = noise_samples + optimal_flow  # [B, N, d]

        # --- 4. Random permutation — defines the AR generation order for this step ---
        perms = jax.vmap(lambda k: random.permutation(k, N))(
            random.split(k_perm, B))  # [B, N]

        # Reorder all per-point arrays so that position k = k-th point to generate
        b_idx = jnp.arange(B)
        target_ordered  = target_matched[b_idx[:, None], perms]   # [B, N, d]
        noise_ordered   = noise_samples[b_idx[:, None], perms]    # [B, N, d]
        flow_ordered    = optimal_flow[b_idx[:, None], perms]     # [B, N, d]
        weights_ordered = weights_batch[b_idx[:, None], perms]    # [B, N]

        # Valid mask first — needed to zero out padded positions below
        valid        = weights_ordered > 0                                     # [B, N]
        padding_mask = valid

        # Zero out padded positions. OT produces undefined flow vectors for
        # weight-0 points (possibly NaN), and LayerNorm propagates NaN across
        # the entire sequence. Zeroing here makes padded positions inert.
        target_ordered = jnp.where(padding_mask[:, :, None], target_ordered, 0.0)
        noise_ordered  = jnp.where(padding_mask[:, :, None], noise_ordered,  0.0)
        flow_ordered   = jnp.where(padding_mask[:, :, None], flow_ordered,   0.0)

        # --- 5. Interpolation across all N positions ---
        t = random.uniform(k_t, [B, N])                                       # [B, N]
        x_t = noise_ordered + (1.0 - t[:, :, None]) * flow_ordered            # [B, N, d]
        u   = -flow_ordered                                                    # [B, N, d]

        # --- 6. Loss over all valid (non-padded) positions ---
        def loss_fn(params):
            v_pred = state.apply_fn(
                {"params": params},
                x_t=x_t,
                t=t,
                target_points=target_ordered,
                padding_mask=padding_mask,
                deterministic=False,
                rngs={"dropout": k_dropout},
            )  # [B, N, d]
            per_point_loss = jnp.mean(jnp.square(v_pred - u), axis=-1)  # [B, N]
            # jnp.where instead of multiplication: NaN * 0 = NaN in JAX's autodiff
            return jnp.mean(jnp.where(valid, per_point_loss, 0.0))

        loss, grads = jax.value_and_grad(loss_fn)(state.params)
        state = state.apply_gradients(grads=grads)
        return state, loss, batch_iters

    @partial(jit, static_argnums=(0,))
    def train_step_completion(
        self, state, point_clouds_batch, weights_batch, context_mask_batch, key=random.key(0)
    ):
        """AR-WFM training step for inpainting / shape completion.

        Identical to train_step except the AR generation order is structured:
        context points (context_mask_batch == True) always appear first in the
        sequence (in a random internal order), target points after.  This means
        every target-position prediction is conditioned on the full context,
        exactly matching the inference setup in generate_completion.

        Loss is computed only over target positions (valid & not context).

        :param context_mask_batch: bool [B, N], True = context (known) point.
        """
        B, N, d = point_clouds_batch.shape

        key, k_noise, k_t, k_perm, k_dropout, k_ot = random.split(key, 6)

        # --- 1. Sample noise clouds ---
        noise_samples = self.noise_func(
            size=[B, N, d],
            noise_config=self.noise_config,
            key=k_noise,
        )
        if isinstance(noise_samples, tuple):
            noise_samples, noise_weights = noise_samples
        else:
            noise_weights = weights_batch

        # --- 2. Mini-batch OT: reassign noise clouds to target clouds ---
        if self.config.mini_batch_ot_mode:
            noise_ind = self.minibatch_ot(point_clouds_batch, weights_batch, noise_samples, noise_weights)
            noise_samples = noise_samples[noise_ind]
            noise_weights = noise_weights[noise_ind]

        # --- 3. Per-point coupling ---
        if self.config.monge_map == 'random':
            b_idx_ot = jnp.arange(B)
            perms_ot = jax.vmap(lambda k: random.permutation(k, N))(
                random.split(k_ot, B))
            target_matched = point_clouds_batch[b_idx_ot[:, None], perms_ot]
            optimal_flow = target_matched - noise_samples
            batch_iters = jnp.array(0.0)
        else:
            ot_result = self.transport_plan_jit(
                [noise_samples, noise_weights],
                [point_clouds_batch, weights_batch],
            )
            optimal_flow = ot_result[0]
            if self.config.num_sinkhorn_iters is None:
                ot_solve = ot_result[1]
                batch_iters = jnp.max(
                    jnp.sum(ot_solve.errors > -1, axis=-1) * jnp.mean(ot_solve.inner_iterations))
            else:
                batch_iters = jnp.array(float(self.config.num_sinkhorn_iters))
            target_matched = noise_samples + optimal_flow

        # --- 4. Structured permutation: context first, target second ---
        # Assign priorities in [0, 0.5) for context and [0.5, 1.0) for target.
        # Padded points (weight == 0) get priority 2.0 and land last.
        # argsort produces a permutation with context randomly shuffled among
        # themselves, followed by target randomly shuffled among themselves.
        u = random.uniform(k_perm, [B, N])
        valid = weights_batch > 0  # [B, N]
        priority = jnp.where(context_mask_batch, u * 0.5, 0.5 + u * 0.5)
        priority = jnp.where(valid, priority, 2.0)
        perms = jnp.argsort(priority, axis=1)  # [B, N]

        b_idx = jnp.arange(B)
        target_ordered        = target_matched[b_idx[:, None], perms]         # [B, N, d]
        noise_ordered         = noise_samples[b_idx[:, None], perms]          # [B, N, d]
        flow_ordered          = optimal_flow[b_idx[:, None], perms]           # [B, N, d]
        weights_ordered       = weights_batch[b_idx[:, None], perms]          # [B, N]
        context_mask_ordered  = context_mask_batch[b_idx[:, None], perms]     # [B, N]

        valid        = weights_ordered > 0
        padding_mask = valid

        target_ordered = jnp.where(padding_mask[:, :, None], target_ordered, 0.0)
        noise_ordered  = jnp.where(padding_mask[:, :, None], noise_ordered,  0.0)
        flow_ordered   = jnp.where(padding_mask[:, :, None], flow_ordered,   0.0)

        # --- 5. Interpolation ---
        t   = random.uniform(k_t, [B, N])
        x_t = noise_ordered + (1.0 - t[:, :, None]) * flow_ordered
        u_  = -flow_ordered

        # --- 6. Loss over target positions only ---
        target_valid = valid & ~context_mask_ordered  # [B, N]

        def loss_fn(params):
            v_pred = state.apply_fn(
                {"params": params},
                x_t=x_t,
                t=t,
                target_points=target_ordered,
                padding_mask=padding_mask,
                deterministic=False,
                rngs={"dropout": k_dropout},
            )
            per_point_loss = jnp.mean(jnp.square(v_pred - u_), axis=-1)
            return jnp.mean(jnp.where(target_valid, per_point_loss, 0.0))

        loss, grads = jax.value_and_grad(loss_fn)(state.params)
        state = state.apply_gradients(grads=grads)
        return state, loss, batch_iters

    def train_completion(
        self,
        masks,
        training_steps=32000,
        batch_size=16,
        verbose=8,
        learning_rate=2e-4,
        decay_steps=1000,
        shape_sample=None,
        saved_state=None,
        key=random.key(0),
    ):
        """Fine-tune (or train from scratch) for inpainting / shape completion.

        At each step, a batch of (point_cloud, context_mask) pairs is sampled.
        The AR generation order places context points first so the model learns
        to generate target points conditioned on the full context — exactly the
        inference setup used by generate_completion.

        :param masks: bool array [n_clouds, N], True = context point.
                      Must be aligned with self.point_clouds (same index order).
        :param training_steps: total gradient steps
        :param batch_size: clouds per step
        :param verbose: print loss every this many steps
        :param learning_rate: initial Adam learning rate
        :param decay_steps: exponential decay period
        :param shape_sample: if set, subsample this many points per cloud per step
        :param saved_state: resume from a saved TrainState
        :param key: JAX random key
        """
        masks = jnp.array(masks, dtype=bool)

        subkey, key = random.split(key)
        if saved_state is None:
            self.state = self.create_train_state(
                model=self.FlowModel,
                learning_rate=learning_rate,
                decay_steps=decay_steps,
                key=subkey,
            )
        else:
            self.state = saved_state
            print(f"Resuming from step {int(self.state.step)}")

        if shape_sample is not None:
            print(f"Sampling {shape_sample} points per cloud per step")
            _sample_batch = jax.vmap(self.sample_single_batch_with_mask, in_axes=(0, 0, 0, 0, None))

        tq = trange(training_steps, leave=True, desc="")
        self.losses = []

        for training_step in tq:
            subkey, key = random.split(key)
            batch_ind = random.choice(subkey, self.point_clouds.shape[0], shape=[batch_size])

            point_clouds_batch = self.point_clouds[batch_ind]
            weights_batch      = self.weights[batch_ind]
            context_mask_batch = masks[batch_ind]

            if shape_sample is not None:
                subkey, key = random.split(key)
                keys = jax.random.split(subkey, batch_size)
                point_clouds_batch, weights_batch, context_mask_batch = _sample_batch(
                    point_clouds_batch, weights_batch, context_mask_batch, keys, shape_sample)

            subkey, key = random.split(key)
            self.state, loss, batch_iters = self.train_step_completion(
                self.state, point_clouds_batch, weights_batch, context_mask_batch, key=subkey)

            self.params = self.state.params
            self.losses.append(loss)

            if training_step % verbose == 0:
                desc = ": {:.3e}".format(loss)
                if self.config.num_sinkhorn_iters is None:
                    desc += " | OT iters: {:.0f}".format(float(batch_iters))
                tq.set_description(desc)

    def sample_single_batch(self, single_batch, single_weights, key, n_points):
        indices = jax.random.choice(key, single_batch.shape[0], (n_points,), replace=False)
        sampled_pc = jnp.take(single_batch, indices, axis=0)
        sample_weights = jnp.take(single_weights, indices, axis=0)
        sample_weights = sample_weights / jnp.sum(sample_weights)
        return [sampled_pc, sample_weights]

    def sample_single_batch_with_mask(self, single_batch, single_weights, single_mask, key, n_points):
        indices = jax.random.choice(key, single_batch.shape[0], (n_points,), replace=False)
        sampled_pc = jnp.take(single_batch, indices, axis=0)
        sample_weights = jnp.take(single_weights, indices, axis=0)
        sample_weights = sample_weights / jnp.sum(sample_weights)
        sampled_mask = jnp.take(single_mask, indices, axis=0)
        return sampled_pc, sample_weights, sampled_mask

    def train(
        self,
        training_steps=32000,
        batch_size=16,
        verbose=8,
        learning_rate=2e-4,
        decay_steps=1000,
        shape_sample=None,
        saved_state=None,
        key=random.key(0),
    ):
        """Train the AR flow model.

        :param training_steps: total gradient steps
        :param batch_size: point clouds per step
        :param verbose: print loss every this many steps
        :param learning_rate: initial Adam learning rate
        :param decay_steps: exponential decay period
        :param shape_sample: subsample this many points per cloud per step
        :param saved_state: resume from a saved TrainState
        :param key: JAX random key
        """
        subkey, key = random.split(key)

        if saved_state is None:
            self.state = self.create_train_state(
                model=self.FlowModel,
                learning_rate=learning_rate,
                decay_steps=decay_steps,
                key=subkey,
            )
        else:
            self.state = saved_state
            print(f"Resuming training from step {int(self.state.step)}")

        if shape_sample is not None:
            print(f"Sampling {shape_sample} points per cloud per step")
            _sample_batch = jax.vmap(self.sample_single_batch, in_axes=(0, 0, 0, None))

        tq = trange(training_steps - int(self.state.step), leave=True, desc="")
        self.losses = []

        for training_step in tq:
            subkey, key = random.split(key)
            batch_ind = random.choice(
                key=subkey,
                a=self.point_clouds.shape[0],
                shape=[batch_size],
            )
            point_clouds_batch = self.point_clouds[batch_ind]
            weights_batch = self.weights[batch_ind]

            if shape_sample is not None:
                subkey, key = random.split(key)
                keys = jax.random.split(subkey, batch_size)
                point_clouds_batch, weights_batch = _sample_batch(
                    point_clouds_batch, weights_batch, keys, shape_sample)

            subkey, key = random.split(key)
            self.state, loss, batch_iters = self.train_step(
                self.state, point_clouds_batch, weights_batch, key=subkey)

            self.params = self.state.params
            self.losses.append(loss)

            if training_step % verbose == 0:
                desc = ": {:.3e}".format(loss)
                if self.config.num_sinkhorn_iters is None:
                    desc += " | OT iters: {:.0f}".format(float(batch_iters))
                tq.set_description(desc)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @partial(jit, static_argnums=(0,))
    def get_flow(self, params, x_t, t, target_points, padding_mask):
        """Full forward pass — used during training inspection."""
        return self.FlowModel.apply(
            {"params": params},
            x_t=x_t,
            t=t,
            target_points=target_points,
            padding_mask=padding_mask,
            deterministic=True,
        )

    @partial(jit, static_argnums=(0,))
    def encode_context(self, params, target_points, padding_mask):
        """Run the causal context encoder and return per-position embeddings.

        During generation of point k, pass the full (partially filled) target
        array and take the embedding at index k.  Called once per generated
        point, not once per ODE timestep.
        """
        return CausalContextEncoder(config=self.config).apply(
            {"params": params["CausalContextEncoder_0"]},
            target_points,
            padding_mask,
            deterministic=True,
        )  # [B, N, context_embedding_dim]

    @partial(jit, static_argnums=(0,))
    def flow_from_embedding(self, params, x_t, t, context_emb):
        """Evaluate the flow MLP given a precomputed context embedding."""
        return FlowMLP(config=self.config, output_dim=self.space_dim).apply(
            {"params": params["FlowMLP_0"]},
            x_t,
            t,
            context_emb,
            deterministic=True,
        )

    def generate_completion(
        self,
        context_points,
        n_complete,
        num_samples=10,
        timesteps=100,
        key=random.key(0),
    ):
        """Complete a partial point cloud auto-regressively.

        Places the observed points in the first n_obs positions of the AR
        sequence (as fixed context) and generates n_complete additional points
        starting at position n_obs, each conditioned on everything before it.

        :param context_points: observed points, shape (n_obs, d) or
                               (num_samples, n_obs, d).  If 2-D, tiled across
                               all samples.
        :param n_complete: number of new points to generate
        :param num_samples: number of completions to produce
        :param timesteps: ODE integration steps per generated point
        :param key: JAX random key
        :return: generated points only, shape [num_samples, n_complete, d]
        """
        context_points = jnp.array(context_points)
        if context_points.ndim == 2:
            context_points = jnp.broadcast_to(
                context_points[None], (num_samples,) + context_points.shape)
        # context_points: [num_samples, n_obs, d]
        n_obs = context_points.shape[1]
        N = n_obs + n_complete
        d = self.space_dim

        print(f"Completing {num_samples} point clouds: "
              f"{n_obs} context + {n_complete} generated ({timesteps} steps/point)...")

        key, subkey = random.split(key)
        noise = self.noise_func(
            size=[num_samples, n_complete, d],
            noise_config=self.noise_config,
            key=subkey,
        )
        if isinstance(noise, tuple):
            noise = noise[0]

        # Seed the AR buffer with the observed points at positions 0..n_obs-1
        target_so_far  = jnp.zeros([num_samples, N, d])
        padding_so_far = jnp.zeros([num_samples, N], dtype=bool)
        target_so_far  = target_so_far.at[:, :n_obs, :].set(context_points)
        padding_so_far = padding_so_far.at[:, :n_obs].set(True)

        dt = 1.0 / timesteps
        t_schedule = jnp.linspace(1.0, dt, timesteps)

        generated = []
        for i in tqdm(range(n_complete), desc="Completing points"):
            k = n_obs + i
            z_k = noise[:, i, :]  # [num_samples, d]
            x = z_k

            all_embs = self.encode_context(self.params, target_so_far, padding_so_far)
            ctx_k = all_embs[:, k, :]  # [num_samples, context_embedding_dim]

            for t_val in t_schedule:
                t_curr = jnp.full([num_samples], t_val)
                t_mid  = jnp.full([num_samples], t_val - 0.5 * dt)
                v1     = self.flow_from_embedding(self.params, x,     t_curr, ctx_k)
                x_mid  = x - 0.5 * dt * v1
                v2     = self.flow_from_embedding(self.params, x_mid, t_mid,  ctx_k)
                x      = x - dt * v2

            generated.append(x)
            target_so_far  = target_so_far.at[:, k, :].set(x)
            padding_so_far = padding_so_far.at[:, k].set(True)

        return jnp.stack(generated, axis=1)  # [num_samples, n_complete, d]

    def generate_samples(self, num_samples=10, timesteps=100, size = None, key=random.key(0)):
        """Generate point clouds auto-regressively, one point at a time.

        At step k the previously generated points {x^1, ..., x^{k-1}} are used
        as context when integrating the k-th noise point z^k to x^k via RK2.

        :param num_samples: number of point clouds to generate
        :param timesteps: ODE integration steps per point
        :param key: JAX random key
        :return: generated point clouds, shape [num_samples, N, d]
        """

        if size is not None:
            N = size
        else:
            N = self.cloud_size
        d = self.space_dim

        print(f"Generating {num_samples} point clouds "
              f"({N} points each, {timesteps} steps/point)...")

        key, subkey = random.split(key)
        noise = self.noise_func(
            size=[num_samples, N, d],
            noise_config=self.noise_config,
            key=subkey,
        )
        if isinstance(noise, tuple):
            noise = noise[0]
        # noise: [num_samples, N, d]



        # Holds the generated points in AR order; grows as each point is committed.
        # The causal encoder sees this array and position k's embedding reflects
        # exactly the context {x^0, ..., x^{k-1}} via the internal shift-by-1.
        target_so_far  = jnp.zeros([num_samples, N, d])
        padding_so_far = jnp.zeros([num_samples, N], dtype=bool)

        dt = 1.0 / timesteps
        t_schedule = jnp.linspace(1.0, dt, timesteps)

        generated = []
        for k in tqdm(range(N), desc="Generating points"):
            z_k = noise[:, k, :]  # [num_samples, d]
            x = z_k

            # Encode the full context array once; extract position k's embedding.
            # Due to the causal mask, position k only sees points 0..k-1.
            all_embs = self.encode_context(self.params, target_so_far, padding_so_far)
            ctx_k = all_embs[:, k, :]  # [num_samples, context_embedding_dim]

            # RK2 (midpoint method) integration from t=1 → t=0
            for t_val in t_schedule:
                t_curr = jnp.full([num_samples], t_val)
                t_mid  = jnp.full([num_samples], t_val - 0.5 * dt)

                v1    = self.flow_from_embedding(self.params, x,     t_curr, ctx_k)
                x_mid = x - 0.5 * dt * v1
                v2    = self.flow_from_embedding(self.params, x_mid, t_mid,  ctx_k)
                x     = x - dt * v2

            generated.append(x)

            # Commit x^k so that position k+1's context includes it
            if k < N - 1:
                target_so_far  = target_so_far.at[:, k, :].set(x)
                padding_so_far = padding_so_far.at[:, k].set(True)

        return jnp.stack(generated, axis=1)  # [num_samples, N, d]

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_model(self, path):
        """Save trained parameters to disk."""
        with open(path, 'wb') as f:
            pickle.dump(self.params, f)
        print(f"Model saved to {path}")

    def load_model(self, path):
        """Load parameters from disk and rebuild train state."""
        print(f"Loading model from {path}")
        self.FlowModel = ARFlowModel(config=self.config, space_dim=self.space_dim)
        with open(path, 'rb') as f:
            self.params = pickle.load(f)
        self.state = self.create_train_state(
            model=self.FlowModel, learning_rate=1, decay_steps=1, key=random.key(0))
        self.state = self.state.replace(params=self.params)
