from functools import partial
import io
import contextlib
import pickle

import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax import jit, random
from tqdm import tqdm, trange
from flax.training import train_state

from wassersteinflowmatching.autoregressivewasserstein.AutoRegressiveWFM import AutoRegressiveWFM
from wassersteinflowmatching.autoregressivewasserstein._utils_Networks import (
    SpatialARFlowModel, SpatialCausalContextEncoder, FlowMLP,
)
from wassersteinflowmatching.autoregressivewasserstein.DefaultConfig import ARWFMConfig


class SpatialAutoRegressiveWassersteinFM(AutoRegressiveWFM):
    """AR-WFM with isotropic ALiBi spatial relative positional embeddings.

    Positions fix the identity of each point, so OT matching is unnecessary —
    noise point i flows directly to target point i.  Pairwise Euclidean distances
    between spatial positions are used as ALiBi biases in the causal context encoder.

    :param point_clouds: list of np.array, each (n_i, d)  — point features
    :param positions:    list of np.array, each (n_i, s), OR np.array [N, n, s]
                         — spatial coordinates for RPE (s may differ from d)
    :param config: ARWFMConfig
    :param key: JAX random key
    """

    def __init__(
        self,
        point_clouds,
        positions,
        config=ARWFMConfig,
        key=random.key(0),
        **kwargs,
    ):
        with contextlib.redirect_stdout(io.StringIO()):
            super().__init__(point_clouds=point_clouds, config=config, key=key, **kwargs)

        # Pad positions to match self.cloud_size (same convention as point_clouds)
        if isinstance(positions, list):
            positions = self._pad_positions(positions)
        self.positions = jnp.array(positions)

        assert self.positions.shape[0] == self.point_clouds.shape[0], (
            "positions and point_clouds must have the same number of clouds"
        )
        assert self.positions.shape[1] == self.cloud_size, (
            f"positions cloud size {self.positions.shape[1]} must match "
            f"cloud_size {self.cloud_size} (pad before passing)"
        )

        self.space_dim_spatial = int(self.positions.shape[-1])

        # Replace the flow model with the spatial variant
        self.FlowModel = SpatialARFlowModel(config=self.config, space_dim=self.space_dim)

        cfg = self.config
        print(
            f"SpatialAutoRegressiveWassersteinFM | "
            f"noise={cfg.noise_type} | "
            f"context: {cfg.context_num_layers}L × {cfg.context_embedding_dim}d, {cfg.context_num_heads}h | "
            f"flow: {cfg.flow_num_layers}L × {cfg.flow_hidden_dim}d | "
            f"clouds: {self.point_clouds.shape[0]} × {self.cloud_size} pts × {self.space_dim}d "
            f"[pos {self.space_dim_spatial}d]"
        )

    # ------------------------------------------------------------------
    # Disable inherited OT setup — not needed when positions fix identity
    # ------------------------------------------------------------------

    def _setup_transport_plan(self):
        pass

    def _setup_minibatch_ot(self):
        pass

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _pad_positions(self, positions_list):
        s = positions_list[0].shape[-1]
        padded = np.zeros([len(positions_list), self.cloud_size, s])
        for i, pos in enumerate(positions_list):
            n = pos.shape[0]
            padded[i, :n, :] = pos
        return padded

    @staticmethod
    def _compute_permutation(positions_batch, weights_batch, key, order_mode):
        """Return a permutation [B, N] defining the AR generation order.

        order_mode:
          'random'     — uniformly random permutation each step
          'inside_out' — random first point, then sorted by ascending distance
          'mixture'    — random first point, then sampled by 1/distance (Gumbel-max)

        Padded points (weight == 0) are always placed last.
        """
        B, N, _ = positions_batch.shape
        padded_mask = weights_batch == 0          # [B, N]

        if order_mode == 'random':
            return jax.vmap(lambda k: random.permutation(k, N))(random.split(key, B))

        # Inside-out and mixture both need a first point and distances
        k_first, k_gumbel = random.split(key)

        valid_counts = jnp.sum(~padded_mask, axis=1).astype(jnp.int32)   # [B]
        # Pick the first point uniformly from the valid slots
        raw = random.randint(k_first, [B], 0, N)
        first_idx = raw % jnp.maximum(valid_counts, 1)                    # [B]

        first_pos = positions_batch[jnp.arange(B), first_idx, :]          # [B, s]
        dists = jnp.linalg.norm(
            positions_batch - first_pos[:, None, :], axis=-1)             # [B, N]

        if order_mode == 'inside_out':
            # Force the first point to sort first; padded points sort last
            dists = dists.at[jnp.arange(B), first_idx].set(-1.0)
            dists = jnp.where(padded_mask, jnp.inf, dists)
            return jnp.argsort(dists, axis=-1)                            # [B, N]

        if order_mode == 'mixture':
            # Gumbel-max trick: sample without replacement with weight ∝ 1/dist
            inv_dist = 1.0 / (dists + 1e-8)
            inv_dist = jnp.where(padded_mask, 0.0, inv_dist)
            log_w = jnp.log(inv_dist + 1e-30)                            # -inf for padded
            # Force first_idx to position 0; padded already -inf → sorts last
            log_w = log_w.at[jnp.arange(B), first_idx].set(jnp.inf)
            gumbel = -jnp.log(-jnp.log(
                random.uniform(k_gumbel, [B, N]) + 1e-20) + 1e-20)
            # argsort(-(log_w + gumbel)): largest score → smallest negated → first
            return jnp.argsort(-(log_w + gumbel), axis=-1)               # [B, N]

        raise ValueError(f"Unknown order_mode '{order_mode}'; "
                         "choose 'random', 'inside_out', or 'mixture'.")

    @staticmethod
    def _make_positions_shifted(positions_ordered, padding_mask):
        """Shift positions by 1; BOS position = centroid of valid points.

        positions_ordered: [B, N, s]
        padding_mask:      [B, N]   bool
        returns: (positions_shifted [B, N, s], centroid [B, 1, s])
        """
        # valid_count: [B, 1, 1] so division with [B, 1, s] stays [B, 1, s]
        valid_count = jnp.sum(padding_mask, axis=1, keepdims=True).astype(jnp.float32)[:, :, None]
        centroid = (
            jnp.sum(positions_ordered, axis=1, keepdims=True)
            / jnp.maximum(valid_count, 1.0)
        )  # [B, 1, s]
        positions_shifted = jnp.concatenate(
            [centroid, positions_ordered[:, :-1, :]], axis=1)  # [B, N, s]
        return positions_shifted, centroid

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def create_train_state(self, model, learning_rate, decay_steps, key=random.key(0)):
        subkey_params, subkey_dropout = random.split(key)

        B, N, d = 4, min(self.cloud_size, 64), self.space_dim
        s = self.space_dim_spatial
        dummy_x_t        = jnp.zeros([B, N, d])
        dummy_t          = jnp.zeros([B, N])
        dummy_targets    = jnp.zeros([B, N, d])
        dummy_positions  = jnp.zeros([B, N, s])
        dummy_padding    = jnp.ones([B, N], dtype=bool)

        params = model.init(
            {"params": subkey_params, "dropout": subkey_dropout},
            x_t=dummy_x_t,
            t=dummy_t,
            target_points=dummy_targets,
            positions_shifted=dummy_positions,
            current_positions=dummy_positions,
            padding_mask=dummy_padding,
            deterministic=True,
        )['params']

        lr_sched = optax.exponential_decay(learning_rate, decay_steps, 0.998, staircase=False)
        tx = optax.adam(lr_sched)
        return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)

    @partial(jit, static_argnums=(0,))
    def train_step(self, state, point_clouds_batch, weights_batch, positions_batch,
                   key=random.key(0)):
        """One JIT-compiled spatial AR-WFM training step.

        No OT matching — positions fix point identity so noise[i] flows directly
        to target[i].  A random permutation still defines the AR generation order
        so the model trains on all N positions simultaneously via causal attention.
        """
        B, N, d = point_clouds_batch.shape

        key, k_noise, k_t, k_perm, k_dropout = random.split(key, 5)

        # --- 1. Sample noise clouds ---
        noise_samples = self.noise_func(
            size=[B, N, d], noise_config=self.noise_config, key=k_noise)
        if isinstance(noise_samples, tuple):
            noise_samples, _ = noise_samples

        # --- 2. Direct pairing: flow[b, i] = target[b, i] − noise[b, i] ---
        optimal_flow  = point_clouds_batch - noise_samples  # [B, N, d]
        target_matched = point_clouds_batch                  # [B, N, d]

        # --- 3. Permutation — defines AR generation order ---
        perms = self._compute_permutation(
            positions_batch, weights_batch, k_perm, self.config.order_mode)
        b_idx = jnp.arange(B)
        target_ordered    = target_matched[b_idx[:, None], perms]
        noise_ordered     = noise_samples[b_idx[:, None], perms]
        flow_ordered      = optimal_flow[b_idx[:, None], perms]
        weights_ordered   = weights_batch[b_idx[:, None], perms]
        positions_ordered = positions_batch[b_idx[:, None], perms]  # [B, N, s]

        valid        = weights_ordered > 0
        padding_mask = valid

        # Zero out padded slots to prevent NaN propagation through LayerNorm
        target_ordered    = jnp.where(padding_mask[:, :, None], target_ordered,    0.0)
        noise_ordered     = jnp.where(padding_mask[:, :, None], noise_ordered,     0.0)
        flow_ordered      = jnp.where(padding_mask[:, :, None], flow_ordered,      0.0)
        positions_ordered = jnp.where(padding_mask[:, :, None], positions_ordered, 0.0)

        # --- 4. Build shifted positions (BOS = centroid of valid points) ---
        positions_shifted, _ = self._make_positions_shifted(positions_ordered, padding_mask)
        # ALiBi uses pairwise distances, which are already translation-invariant
        current_positions = positions_ordered  # [B, N, s]

        # --- 5. Interpolation ---
        t   = random.uniform(k_t, [B, N])
        x_t = noise_ordered + (1.0 - t[:, :, None]) * flow_ordered
        u   = -flow_ordered

        # --- 6. Loss over all valid positions ---
        def loss_fn(params):
            v_pred = state.apply_fn(
                {"params": params},
                x_t=x_t,
                t=t,
                target_points=target_ordered,
                positions_shifted=positions_shifted,
                current_positions=current_positions,
                padding_mask=padding_mask,
                deterministic=False,
                rngs={"dropout": k_dropout},
            )  # [B, N, d]
            per_point_loss = jnp.mean(jnp.square(v_pred - u), axis=-1)  # [B, N]
            return jnp.mean(jnp.where(valid, per_point_loss, 0.0))

        loss, grads = jax.value_and_grad(loss_fn)(state.params)
        state = state.apply_gradients(grads=grads)
        return state, loss

    def sample_single_batch_spatial(self, single_batch, single_weights, single_positions,
                                    key, n_points):
        indices = jax.random.choice(key, single_batch.shape[0], (n_points,), replace=False)
        sampled_pc        = jnp.take(single_batch,     indices, axis=0)
        sampled_weights   = jnp.take(single_weights,   indices, axis=0)
        sampled_weights   = sampled_weights / jnp.sum(sampled_weights)
        sampled_positions = jnp.take(single_positions, indices, axis=0)
        return [sampled_pc, sampled_weights, sampled_positions]

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
            _sample_batch = jax.vmap(
                self.sample_single_batch_spatial, in_axes=(0, 0, 0, 0, None))

        tq = trange(training_steps - int(self.state.step), leave=True, desc="")
        self.losses = []

        for training_step in tq:
            subkey, key = random.split(key)
            batch_ind = random.choice(
                key=subkey, a=self.point_clouds.shape[0], shape=[batch_size])
            point_clouds_batch = self.point_clouds[batch_ind]
            weights_batch      = self.weights[batch_ind]
            positions_batch    = self.positions[batch_ind]

            if shape_sample is not None:
                subkey, key = random.split(key)
                keys = jax.random.split(subkey, batch_size)
                point_clouds_batch, weights_batch, positions_batch = _sample_batch(
                    point_clouds_batch, weights_batch, positions_batch, keys, shape_sample)

            subkey, key = random.split(key)
            self.state, loss = self.train_step(
                self.state, point_clouds_batch, weights_batch, positions_batch, key=subkey)

            self.params = self.state.params
            self.losses.append(loss)

            if training_step % verbose == 0:
                tq.set_description(": {:.3e}".format(loss))

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @partial(jit, static_argnums=(0,))
    def get_flow(self, params, x_t, t, target_points, positions_shifted, current_positions,
                 padding_mask):
        return self.FlowModel.apply(
            {"params": params},
            x_t=x_t,
            t=t,
            target_points=target_points,
            positions_shifted=positions_shifted,
            current_positions=current_positions,
            padding_mask=padding_mask,
            deterministic=True,
        )

    @partial(jit, static_argnums=(0,))
    def encode_context(self, params, target_points, positions_shifted, current_positions,
                       padding_mask):
        """Run the spatial causal encoder; returns per-position embeddings [B, N, emb_dim]."""
        return SpatialCausalContextEncoder(config=self.config).apply(
            {"params": params["SpatialCausalContextEncoder_0"]},
            target_points,
            positions_shifted,
            current_positions,
            padding_mask,
            deterministic=True,
        )

    def generate_samples(self, positions, num_samples=10, timesteps=100, size=None,
                         order_mode=None, key=random.key(0)):
        """Generate point clouds auto-regressively conditioned on spatial positions.

        :param positions: spatial coordinates [num_samples, N, s] or [N, s]
        :param num_samples: number of clouds to generate
        :param timesteps: ODE integration steps per point
        :param size: expected N — validated against positions.shape; defaults to positions.shape
        :param order_mode: AR generation order — 'random', 'inside_out', or 'mixture'.
                           None falls back to self.config.order_mode.
        :param key: JAX random key
        :return: generated point clouds [num_samples, N, d] in original position order
        """
        if order_mode is None:
            order_mode = self.config.order_mode

        d = self.space_dim

        positions = jnp.array(positions)
        if positions.ndim == 2:  # [N, s] → [num_samples, N, s]
            positions = jnp.broadcast_to(
                positions[None], (num_samples,) + positions.shape)

        N = positions.shape[1]
        if size is not None and size != N:
            raise ValueError(
                f"positions has {N} points but size={size}; pass positions of the "
                f"correct length or omit size"
            )

        print(f"Generating {num_samples} point clouds "
              f"({N} points each, {timesteps} steps/point, order={order_mode})...")

        key, k_perm, k_noise = random.split(key, 3)

        # --- Compute AR generation order ---
        dummy_weights = jnp.ones([num_samples, N])     # all points valid (no padding)
        perms = self._compute_permutation(
            positions, dummy_weights, k_perm, order_mode)  # [num_samples, N]

        b_idx = jnp.arange(num_samples)
        positions_ordered = positions[b_idx[:, None], perms, :]            # [S, N, s]

        # Build shifted positions from the ordered sequence
        centroid = positions_ordered.mean(axis=1, keepdims=True)           # [S, 1, s]
        positions_shifted = jnp.concatenate(
            [centroid, positions_ordered[:, :-1, :]], axis=1)              # [S, N, s]
        current_positions = positions_ordered

        noise = self.noise_func(
            size=[num_samples, N, d], noise_config=self.noise_config, key=k_noise)
        if isinstance(noise, tuple):
            noise = noise[0]

        target_so_far  = jnp.zeros([num_samples, N, d])
        padding_so_far = jnp.zeros([num_samples, N], dtype=bool)

        dt = 1.0 / timesteps
        t_schedule = jnp.linspace(1.0, dt, timesteps)

        generated_ordered = []
        for k in tqdm(range(N), desc="Generating points"):
            z_k = noise[:, k, :]
            x   = z_k

            all_embs = self.encode_context(
                self.params, target_so_far, positions_shifted, current_positions,
                padding_so_far)
            ctx_k = all_embs[:, k, :]  # [num_samples, context_embedding_dim]

            for t_val in t_schedule:
                t_curr = jnp.full([num_samples], t_val)
                t_mid  = jnp.full([num_samples], t_val - 0.5 * dt)

                v1    = self.flow_from_embedding(self.params, x,     t_curr, ctx_k)
                x_mid = x - 0.5 * dt * v1
                v2    = self.flow_from_embedding(self.params, x_mid, t_mid,  ctx_k)
                x     = x - dt * v2

            generated_ordered.append(x)

            if k < N - 1:
                target_so_far  = target_so_far.at[:, k, :].set(x)
                padding_so_far = padding_so_far.at[:, k].set(True)

        # Stack in AR order then inverse-permute so output[b, i] ↔ positions[b, i]
        result_ordered = jnp.stack(generated_ordered, axis=1)     # [S, N, d]
        inv_perms = jnp.argsort(perms, axis=-1)                   # [S, N]
        return result_ordered[b_idx[:, None], inv_perms, :]        # [S, N, d]

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def load_model(self, path):
        print(f"Loading model from {path}")
        self.FlowModel = SpatialARFlowModel(config=self.config, space_dim=self.space_dim)
        with open(path, 'rb') as f:
            self.params = pickle.load(f)
        self.state = self.create_train_state(
            model=self.FlowModel, learning_rate=1, decay_steps=1, key=random.key(0))
        self.state = self.state.replace(params=self.params)
