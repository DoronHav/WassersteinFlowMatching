from functools import partial
import io
import contextlib
import pickle
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np
import scipy.sparse
import sklearn.neighbors
import anndata
import optax
from jax import jit, random
from tqdm import tqdm, trange
from flax.training import train_state

from wassersteinflowmatching.autoregressivewasserstein.SpatialAutoRegressiveWassersteinFM import (
    SpatialAutoRegressiveWassersteinFM,
)
from wassersteinflowmatching.autoregressivewasserstein._utils_Networks import SpatialARFlowModel
from wassersteinflowmatching.autoregressivewasserstein.DefaultConfig import ARWFMConfig


class SpatialAutoRegressiveWassersteinFM_anndata(SpatialAutoRegressiveWassersteinFM):
    """AR-WFM with spatial ALiBi RPE, loading niches directly from AnnData.

    Each cell's "point cloud" is the expression matrix of the cell itself plus its
    (niche_size - 1) nearest spatial neighbors, giving a total of niche_size points
    per niche.  Niche construction is lazy (on-the-fly during training) so only the
    full expression matrix and the k-NN index list are held in memory — matching the
    memory-efficient pattern of SpatialWassersteinFlowMatching.

    :param adata: AnnData with spatial coordinates in obsm[spatial_key].
    :param niche_size: Total points per niche = 1 (self) + (niche_size - 1) neighbors.
    :param config: ARWFMConfig.
    :param key: JAX random key.
    :param spatial_key: Key in adata.obsm for spatial coordinates.
    :param batch_key: Column in adata.obs for batch-aware k-NN, or -1 to ignore batches.
    :param rep: Key in adata.obsm for expression features; None uses adata.X.
    """

    def __init__(
        self,
        adata: anndata.AnnData,
        niche_size: int = 9,
        config=ARWFMConfig,
        key=random.key(0),
        spatial_key: str = 'spatial',
        batch_key=-1,
        rep: Optional[str] = None,
        **kwargs,
    ):
        if not isinstance(adata, anndata.AnnData):
            raise TypeError("Input 'adata' must be an anndata.AnnData object.")
        if spatial_key not in adata.obsm:
            raise ValueError(f"adata.obsm['{spatial_key}'] not found.")
        if niche_size < 2:
            raise ValueError("niche_size must be at least 2 (1 self + 1 neighbor).")

        k_neighbours = niche_size - 1  # excludes the center cell

        print("Pre-computing neighbor indices and caching expression data...")
        exp_data      = self._get_exp_data(adata, rep)
        niche_indices = self._get_niche_indices(adata, k_neighbours, spatial_key, batch_key)
        spatial_coords = np.asarray(adata.obsm[spatial_key]).astype('float32')

        d = exp_data.shape[1]
        s = spatial_coords.shape[1]
        # Each niche = self + neighbors; max is niche_size (self + k_neighbours).
        max_niche_size = niche_size

        # Build a small sample for parent init (architecture + noise stats inferred from it).
        # Use niche_size worth of points so cloud_size is inferred correctly.
        sample_cell_ids = list(range(min(8, adata.n_obs)))
        sample_pcs = []
        sample_pos = []
        for i in sample_cell_ids:
            nbrs = niche_indices[i]
            all_idx = np.concatenate([[i], nbrs])
            sample_pcs.append(exp_data[all_idx])
            sample_pos.append(spatial_coords[all_idx])

        # Call parent init with the sample (architecture + noise stats inferred from it).
        with contextlib.redirect_stdout(io.StringIO()):
            super().__init__(
                point_clouds=sample_pcs,
                positions=sample_pos,
                config=config,
                key=key,
                **kwargs,
            )

        # Override sizes to reflect the full dataset (parent only saw the sample).
        self.cloud_size        = max_niche_size
        self.space_dim_spatial = s

        # Store AnnData-related fields for on-the-fly batch assembly during training.
        self.adata               = adata
        self.exp_data_train      = exp_data
        self.niche_indices_train = niche_indices
        self.spatial_coords_train = spatial_coords
        self.max_niche_size      = max_niche_size

        # Rebuild flow model now that cloud_size is finalised.
        self.FlowModel = SpatialARFlowModel(config=self.config, space_dim=self.space_dim)

        print(
            f"SpatialAutoRegressiveWassersteinFM_anndata | "
            f"cells={adata.n_obs} | niche_size={niche_size} (1 self + {k_neighbours} neighbors) | "
            f"feature_dim={d} | spatial_dim={s}"
        )

    # ------------------------------------------------------------------
    # Data loading helpers (ported from SpatialWassersteinFlowMatching)
    # ------------------------------------------------------------------

    @staticmethod
    def _get_exp_data(adata: anndata.AnnData, rep: Optional[str]) -> np.ndarray:
        if rep is None:
            X = adata.X
            return (
                X.toarray().astype('float32')
                if scipy.sparse.issparse(X)
                else np.asarray(X).astype('float32')
            )
        return np.asarray(adata.obsm[rep]).astype('float32')

    @staticmethod
    def _get_niche_indices(
        adata: anndata.AnnData,
        k: int,
        spatial_key: str,
        batch_key,
    ) -> list:
        """Returns a list of length n_obs; each entry is an int array of the k neighbor indices (self excluded)."""
        if batch_key == -1 or (
            isinstance(batch_key, str) and batch_key not in adata.obs.columns
        ):
            knn = sklearn.neighbors.kneighbors_graph(
                adata.obsm[spatial_key], n_neighbors=k,
                mode='connectivity', n_jobs=-1,
            ).tocsr()
            return np.split(knn.indices, knn.indptr[1:-1])

        batch = adata.obs[batch_key]
        niche_indices = [np.array([], dtype=int)] * adata.n_obs
        for val in np.unique(batch):
            mask = (batch == val).values
            orig_idx = np.where(mask)[0]
            sub = adata[mask]
            batch_k = min(k, sub.n_obs - 1)
            if batch_k < 1:
                continue
            knn = sklearn.neighbors.kneighbors_graph(
                sub.obsm[spatial_key], n_neighbors=batch_k,
                mode='connectivity', n_jobs=-1,
            ).tocsr()
            for i, local_nbrs in enumerate(np.split(knn.indices, knn.indptr[1:-1])):
                niche_indices[orig_idx[i]] = orig_idx[local_nbrs]
        return niche_indices

    # ------------------------------------------------------------------
    # On-the-fly batch assembly
    # ------------------------------------------------------------------

    def _assemble_batch(self, cell_indices: np.ndarray):
        """Assemble padded niches and spatial positions for a batch of cells.

        Returns:
            point_clouds_batch : [B, max_niche_size, d]
            positions_batch    : [B, max_niche_size, s]
            weights_batch      : [B, max_niche_size]
        """
        B = len(cell_indices)
        d = self.exp_data_train.shape[1]
        s = self.spatial_coords_train.shape[1]

        pcs = np.zeros((B, self.max_niche_size, d), dtype='float32')
        pos = np.zeros((B, self.max_niche_size, s), dtype='float32')
        wts = np.zeros((B, self.max_niche_size),    dtype='float32')

        for i, cell_idx in enumerate(cell_indices):
            nbrs = self.niche_indices_train[int(cell_idx)]
            all_idx = np.concatenate([[int(cell_idx)], nbrs])  # self first, then neighbors
            n = len(all_idx)
            pcs[i, :n, :] = self.exp_data_train[all_idx]
            pos[i, :n, :] = self.spatial_coords_train[all_idx]
            wts[i, :n]    = 1.0 / n

        return jnp.asarray(pcs), jnp.asarray(pos), jnp.asarray(wts)

    # ------------------------------------------------------------------
    # Training (overrides parent to use on-the-fly niche assembly)
    # ------------------------------------------------------------------

    def train(
        self,
        training_steps: int = 32000,
        batch_size: int = 16,
        verbose: int = 8,
        learning_rate: float = 2e-4,
        decay_steps: int = 1000,
        shape_sample: Optional[int] = None,
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
            batch_cell_ind = np.asarray(
                random.choice(subkey, self.adata.n_obs, shape=(batch_size,), replace=True))
            pcs_batch, pos_batch, wts_batch = self._assemble_batch(batch_cell_ind)

            if shape_sample is not None:
                subkey, key = random.split(key)
                keys = jax.random.split(subkey, batch_size)
                pcs_batch, wts_batch, pos_batch = _sample_batch(
                    pcs_batch, wts_batch, pos_batch, keys, shape_sample)

            subkey, key = random.split(key)
            self.state, loss = self.train_step(
                self.state, pcs_batch, wts_batch, pos_batch, key=subkey)

            self.params = self.state.params
            self.losses.append(loss)

            if training_step % verbose == 0:
                tq.set_description(": {:.3e}".format(loss))

    # ------------------------------------------------------------------
    # Generation helpers
    # ------------------------------------------------------------------

    def get_niche_positions(self, cell_indices) -> jnp.ndarray:
        """Return padded spatial positions of the niche (self + k-NN neighbors) for the given cells.

        :param cell_indices: 1-D sequence of integer indices into adata.
        :return: [B, max_niche_size, s]
        """
        cell_indices = np.asarray(cell_indices)
        B = len(cell_indices)
        s = self.spatial_coords_train.shape[1]
        pos = np.zeros((B, self.max_niche_size, s), dtype='float32')
        for i, cell_idx in enumerate(cell_indices):
            nbrs = self.niche_indices_train[int(cell_idx)]
            all_idx = np.concatenate([[int(cell_idx)], nbrs])
            n = len(all_idx)
            pos[i, :n, :] = self.spatial_coords_train[all_idx]
        return jnp.asarray(pos)

    def generate_niche_for_cells(
        self,
        cell_indices,
        timesteps: int = 100,
        order_mode=None,
        key=random.key(0),
    ) -> jnp.ndarray:
        """Generate niches conditioned on the observed spatial positions of the given cells.

        Uses the k-NN neighbor coordinates already stored in the model, so no
        additional inputs are required beyond the cell indices.

        :param cell_indices: List or array of integer indices into adata.
        :param timesteps: ODE integration steps per point.
        :param order_mode: AR generation order — 'random', 'inside_out', or 'mixture'.
                           None falls back to self.config.order_mode.
        :param key: JAX random key.
        :return: Generated niches [len(cell_indices), max_niche_size, d].
        """
        cell_indices = np.asarray(cell_indices)
        positions = self.get_niche_positions(cell_indices)
        return self.generate_samples(
            positions=positions,
            num_samples=len(cell_indices),
            timesteps=timesteps,
            order_mode=order_mode,
            key=key,
        )

    def generate_niche_for_position(
        self,
        positions,
        n_generate: int = 1,
        timesteps: int = 100,
        order_mode: str = 'inside_out',
        key=random.key(0),
    ) -> jnp.ndarray:
        """Generate niches conditioned on an arbitrary spatial layout.

        Unlike generate_niche_for_cells, no center cell or k-NN graph is needed —
        any set of spatial coordinates can be used to condition the generation.

        Inside-out order is the default: a random anchor point is chosen and remaining
        positions are presented to the causal encoder in ascending distance order,
        so the model builds context outward from the anchor.

        :param positions: spatial coordinates [N, s] (broadcast to n_generate copies)
                          or [n_generate, N, s] (one layout per sample).
        :param n_generate: number of samples to generate; used only when positions is [N, s].
        :param timesteps: ODE integration steps per point.
        :param order_mode: 'inside_out' (default), 'random', or 'mixture'.
        :param key: JAX random key.
        :return: [n_generate, N, d]
        """
        positions = jnp.array(positions, dtype=jnp.float32)
        if positions.ndim == 2:                           # [N, s] → broadcast
            positions = jnp.broadcast_to(
                positions[None], (n_generate,) + positions.shape)
        elif positions.ndim == 3:
            n_generate = positions.shape[0]
        else:
            raise ValueError(
                f"positions must be 2D [N, s] or 3D [n_generate, N, s], "
                f"got shape {positions.shape}")

        return self.generate_samples(
            positions=positions,
            num_samples=n_generate,
            timesteps=timesteps,
            order_mode=order_mode,
            key=key,
        )

    def seed_niche(
        self,
        positions,
        seed_features,
        n_generate: int = 1,
        timesteps: int = 100,
        key=random.key(0),
    ) -> jnp.ndarray:
        """Generate niche cells conditioned on a set of known seed cells.

        The seed cells occupy the first n_seeds slots in positions and are committed
        directly into the AR context without ODE integration.  The remaining
        N - n_seeds cells are then generated in ascending distance order from the
        nearest seed (inside-out from the seed cluster).

        Typical use: seed_features = expression of the center cell (slot 0), so the
        model generates the rest of the niche conditioned on knowing the center.

        :param positions: full niche spatial layout [N, s] or [n_generate, N, s].
                          Slots 0 … n_seeds-1 must correspond to the seed cells.
        :param seed_features: known expression features [n_seeds, d] or
                              [n_generate, n_seeds, d].
        :param n_generate: samples to draw when inputs are not already batched.
        :param timesteps: ODE integration steps per generated point.
        :param key: JAX random key.
        :return: [n_generate, N, d] — seed slots contain seed_features,
                 remaining slots contain generated features, in original position order.
        """
        positions     = jnp.array(positions,     dtype=jnp.float32)
        seed_features = jnp.array(seed_features, dtype=jnp.float32)

        # --- Normalise to [S, *, *] ---
        if seed_features.ndim == 2:            # [n_seeds, d] → broadcast
            seed_features = jnp.broadcast_to(
                seed_features[None], (n_generate,) + seed_features.shape)
        elif seed_features.ndim == 3:
            n_generate = seed_features.shape[0]

        if positions.ndim == 2:                # [N, s] → broadcast
            positions = jnp.broadcast_to(
                positions[None], (n_generate,) + positions.shape)

        S, N, s = positions.shape
        n_seeds = seed_features.shape[1]
        d       = self.space_dim

        if n_seeds >= N:
            raise ValueError(
                f"n_seeds ({n_seeds}) must be < N ({N}); nothing left to generate.")
        if seed_features.shape[2] != d:
            raise ValueError(
                f"seed_features last dim {seed_features.shape[2]} != model space_dim {d}")

        print(f"Seeded generation: {S} samples | "
              f"{n_seeds} seeds fixed | {N - n_seeds} cells to generate "
              f"({timesteps} steps/point) ...")

        # --- AR order: seeds first (0 … n_seeds-1), then rest by min-dist to seed ---
        seed_pos = positions[:, :n_seeds, :]           # [S, n_seeds, s]
        rest_pos = positions[:, n_seeds:,  :]          # [S, N-n_seeds, s]

        # dist[b, i] = distance from rest_pos[b, i] to nearest seed
        diffs         = rest_pos[:, :, None, :] - seed_pos[:, None, :, :]
        # [S, N-n_seeds, n_seeds, s] → norm → [S, N-n_seeds, n_seeds] → min → [S, N-n_seeds]
        dist_to_seeds = jnp.linalg.norm(diffs, axis=-1).min(axis=-1)
        rest_order    = jnp.argsort(dist_to_seeds, axis=-1) + n_seeds  # global indices

        seed_slots = jnp.broadcast_to(jnp.arange(n_seeds)[None], (S, n_seeds))
        perms      = jnp.concatenate([seed_slots, rest_order], axis=1)  # [S, N]

        b_idx             = jnp.arange(S)
        positions_ordered = positions[b_idx[:, None], perms, :]        # [S, N, s]
        centroid          = positions_ordered.mean(axis=1, keepdims=True)
        positions_shifted = jnp.concatenate(
            [centroid, positions_ordered[:, :-1, :]], axis=1)          # [S, N, s]
        current_positions = positions_ordered

        key, k_noise = random.split(key)
        noise = self.noise_func(
            size=[S, N, d], noise_config=self.noise_config, key=k_noise)
        if isinstance(noise, tuple):
            noise = noise[0]

        # Pre-commit seeds; causal attention will condition generated cells on them
        target_so_far  = jnp.zeros([S, N, d])
        padding_so_far = jnp.zeros([S, N], dtype=bool)
        target_so_far  = target_so_far.at[:,  :n_seeds, :].set(seed_features)
        padding_so_far = padding_so_far.at[:, :n_seeds].set(True)

        dt         = 1.0 / timesteps
        t_schedule = jnp.linspace(1.0, dt, timesteps)

        generated_ordered = []
        for k in tqdm(range(n_seeds, N), desc="Generating points"):
            z_k = noise[:, k, :]
            x   = z_k

            all_embs = self.encode_context(
                self.params, target_so_far, positions_shifted, current_positions,
                padding_so_far)
            ctx_k = all_embs[:, k, :]

            for t_val in t_schedule:
                t_curr = jnp.full([S], t_val)
                t_mid  = jnp.full([S], t_val - 0.5 * dt)
                v1     = self.flow_from_embedding(self.params, x,     t_curr, ctx_k)
                x_mid  = x - 0.5 * dt * v1
                v2     = self.flow_from_embedding(self.params, x_mid, t_mid,  ctx_k)
                x      = x - dt * v2

            generated_ordered.append(x)
            if k < N - 1:
                target_so_far  = target_so_far.at[:, k, :].set(x)
                padding_so_far = padding_so_far.at[:, k].set(True)

        # seeds | generated → [S, N, d] in permuted order; inverse-permute to original
        result_ordered = jnp.concatenate(
            [seed_features, jnp.stack(generated_ordered, axis=1)], axis=1)
        inv_perms = jnp.argsort(perms, axis=-1)
        return result_ordered[b_idx[:, None], inv_perms, :]            # [S, N, d]

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def load_model(self, path: str):
        print(f"Loading model from {path}")
        self.FlowModel = SpatialARFlowModel(config=self.config, space_dim=self.space_dim)
        with open(path, 'rb') as f:
            self.params = pickle.load(f)
        self.state = self.create_train_state(
            model=self.FlowModel, learning_rate=1, decay_steps=1, key=random.key(0))
        self.state = self.state.replace(params=self.params)
