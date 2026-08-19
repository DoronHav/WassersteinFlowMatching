"""
Gromov-Wasserstein Gradient Flow Matching (GW-GFM).

Trains a Set Transformer f_theta(X_t, t) to imitate the velocity field of
the GW gradient flow:

    dX_t/dt = -∇_X GW_ε²(X_t, X_1)

The teacher path is computed per training step via Euler integration with
warm-started GW solves.  At inference time, the learned field f_theta is
integrated with an RK2 midpoint step to generate new point clouds.
"""

from functools import partial
import types
import pickle  # type: ignore

import jax  # type: ignore
import jax.numpy as jnp  # type: ignore
import numpy as np  # type: ignore
import optax  # type: ignore
from jax import jit, random  # type: ignore
from flax.training import train_state  # type: ignore
from tqdm import trange, tqdm  # type: ignore

import wassersteinflowmatching.gromov_wasserstein.utils_OT as utils_OT  # type: ignore
import wassersteinflowmatching.gromov_wasserstein.utils_Noise as utils_Noise  # type: ignore
from wassersteinflowmatching.gromov_wasserstein._utils_Transformer import SetTransformer  # type: ignore
from wassersteinflowmatching.gromov_wasserstein.DefaultConfig import GromovWassersteinFlowMatchingConfig  # type: ignore
from wassersteinflowmatching.gromov_wasserstein._utils_Processing import pad_pointclouds  # type: ignore


class GromovWassersteinFlowMatching:
    """Gromov-Wasserstein Gradient Flow Matching model.

    Learns a Set Transformer velocity field whose teacher path is defined by
    the gradient flow of the regularized GW energy.

    :param point_clouds: List of target (data) point clouds, each ``(n_i, d)``.
    :param noise_point_clouds: Optional list of source (noise) point clouds.
        When ``None``, noise is sampled from a simple analytical distribution.
    :param config: :class:`GromovWassersteinFlowMatchingConfig` instance or
        class.  Extra keyword arguments override config fields.
    :param key: JAX random seed.
    :param kwargs: Override any config field by name.
    """

    def __init__(
        self,
        point_clouds: list,
        noise_point_clouds=None,
        config=GromovWassersteinFlowMatchingConfig,
        key=random.key(0),
        **kwargs,
    ):
        print("Initializing Gromov-Wasserstein Flow Matching")

        # ------------------------------------------------------------------
        # Config
        # ------------------------------------------------------------------
        if isinstance(config, type):
            config = config()
        if config is None:
            config = GromovWassersteinFlowMatchingConfig()
        if kwargs:
            config = config.replace(**kwargs)
        self.config = config

        # ------------------------------------------------------------------
        # Data
        # ------------------------------------------------------------------
        self.point_clouds = list(point_clouds)
        self.weights = [
            np.ones(pc.shape[0]) / pc.shape[0] for pc in self.point_clouds
        ]
        self.point_clouds, self.weights = pad_pointclouds(
            self.point_clouds, self.weights
        )
        self.space_dim = self.point_clouds.shape[-1]
        print(
            f"  Data: {self.point_clouds.shape[0]} point clouds, "
            f"{self.point_clouds.shape[1]} points, {self.space_dim}D"
        )

        # ------------------------------------------------------------------
        # Noise distribution
        # ------------------------------------------------------------------
        self.noise_config = types.SimpleNamespace()
        if noise_point_clouds is not None:
            self.noise_point_clouds = self._scale(noise_point_clouds)
            self.noise_weights = [
                np.ones(pc.shape[0]) / pc.shape[0]
                for pc in self.noise_point_clouds
            ]
            self.noise_point_clouds, self.noise_weights = pad_pointclouds(
                self.noise_point_clouds, self.noise_weights
            )
            self.noise_func = utils_Noise.random_pointclouds
            self.noise_config.noise_point_clouds = self.noise_point_clouds
            self.noise_config.noise_weights = self.noise_weights
            print("  Source: custom noise point clouds")
        else:
            active = self.point_clouds[self.weights > 0]
            self.noise_config.minval = float(active.min())
            self.noise_config.maxval = float(active.max())
            self.noise_func = getattr(utils_Noise, self.config.noise_type)
            print(
                f"  Source: '{self.config.noise_type}' noise in "
                f"[{self.noise_config.minval:.3f}, {self.noise_config.maxval:.3f}]"
            )

        # ------------------------------------------------------------------
        # GW solver
        # ------------------------------------------------------------------
        self.gw_solver = utils_OT.make_gw_solver(
            epsilon=self.config.gw_epsilon,
            lse_mode=self.config.gw_lse_mode,
        )
        print(
            f"  GW solver: epsilon={self.config.gw_epsilon}, "
            f"teacher_dt={self.config.teacher_dt}"
        )

        # ------------------------------------------------------------------
        # Neural network
        # ------------------------------------------------------------------
        self.model = SetTransformer(config=self.config)

    # ------------------------------------------------------------------
    # Initialise Flax train state
    # ------------------------------------------------------------------

    def create_train_state(
        self,
        learning_rate: float = 2e-4,
        decay_steps: int = 1000,
        key=random.key(0),
    ):
        """Create a Flax :class:`TrainState` with Adam + exponential decay.

        :param learning_rate: Initial learning rate.
        :param decay_steps: Steps per LR decay cycle.
        :param key: JAX random key.
        :return: Initialised :class:`TrainState`.
        """
        subkey, key = random.split(key)
        dummy_pc = self.noise_func(
            size=[4, min(self.point_clouds.shape[1], 32), self.space_dim],
            noise_config=self.noise_config,
            key=subkey,
        )
        if isinstance(dummy_pc, tuple):
            dummy_pc = dummy_pc[0]

        subkey, _ = random.split(key)
        params = self.model.init(
            rngs={"params": subkey},
            point_cloud=dummy_pc,
            t=jnp.ones((dummy_pc.shape[0],)),
            masks=jnp.ones((dummy_pc.shape[0], dummy_pc.shape[1])),
            deterministic=True,
        )["params"]

        lr_sched = optax.exponential_decay(
            learning_rate, decay_steps, 0.998, staircase=False
        )
        tx = optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adam(lr_sched),
        )
        return train_state.TrainState.create(
            apply_fn=self.model.apply, params=params, tx=tx
        )

    # ------------------------------------------------------------------
    # Teacher path
    # ------------------------------------------------------------------

    def teacher_step_batch(
        self,
        x0_batch: jnp.ndarray,
        x1_batch: jnp.ndarray,
        t: float,
        key=None,
        x0_weights: jnp.ndarray = None,
        x1_weights: jnp.ndarray = None,
    ) -> tuple:
        """Compute (X_t, v_t) for a whole batch using the GW gradient flow.

        Each sample is processed independently via ``jax.vmap``.

        When ``config.rotation_aug`` is ``True`` a different random orthogonal
        rotation is applied to X0 and X1 independently before computing the
        teacher path.  Because GW is invariant to rotations, the coupling is
        unaffected, but the Set Transformer sees the clouds in diverse
        orientations.

        :param x0_batch: Noise point clouds, shape ``(B, n, d)``.
        :param x1_batch: Data point clouds, shape ``(B, m, d)``.
        :param t: Scalar time in [0, 1].
        :param key: JAX random key (used only when ``rotation_aug=True``).
        :return: ``(x_t_batch, v_t_batch)`` both ``(B, n, d)``.
        """
        B = x0_batch.shape[0]
        d = self.space_dim

        if x0_weights is None:
            x0_weights = jnp.ones((B, x0_batch.shape[1])) / x0_batch.shape[1]
        if x1_weights is None:
            x1_weights = jnp.ones((B, x1_batch.shape[1])) / x1_batch.shape[1]

        # Split into per-sample rotation keys (always split; unused when aug=False)
        if key is None:
            key = random.key(0)
        k0, k1 = random.split(key)
        keys_x0 = random.split(k0, B)   # (B,) of keys for X0 rotations
        keys_x1 = random.split(k1, B)   # (B,) of keys for X1 rotations

        # Capture config as local Python constants so closure is vmap-safe
        do_aug      = self.config.rotation_aug
        scale_cost  = self.config.gw_scale_cost
        solver      = self.gw_solver
        epsilon     = self.config.gw_epsilon
        teacher_dt  = self.config.teacher_dt
        grad_clip   = self.config.teacher_grad_clip

        def _step_one(
            x0_i: jnp.ndarray,
            x1_i: jnp.ndarray,
            rk0,
            rk1,
            w0_i: jnp.ndarray,
            w1_i: jnp.ndarray,
        ) -> tuple:
            if do_aug:
                R0 = jax.random.orthogonal(rk0, d)
                R1 = jax.random.orthogonal(rk1, d)
                x0_i = x0_i @ R0
                x1_i = x1_i @ R1
            return utils_OT.teacher_step_single(
                solver=solver,
                epsilon=epsilon,
                teacher_dt=teacher_dt,
                x0=x0_i,
                x1=x1_i,
                t=t,
                grad_clip=grad_clip,
                scale_cost=scale_cost,
                w0=w0_i,
                w1=w1_i,
            )

        x_t, v_t = jax.vmap(_step_one)(x0_batch, x1_batch, keys_x0, keys_x1, x0_weights, x1_weights)
        return x_t, v_t

    # ------------------------------------------------------------------
    # JIT'd neural-network gradient step
    # ------------------------------------------------------------------

    @partial(jit, static_argnums=(0,))
    def _nn_grad_step(
        self,
        state,
        x_t: jnp.ndarray,
        v_t: jnp.ndarray,
        t_batch: jnp.ndarray,
        key=random.key(0),
    ):
        """JIT-compiled kernel: predict velocity and update parameters.

        :param state: Current :class:`TrainState`.
        :param x_t: Interpolated point clouds, shape ``(B, n, d)``.
        :param v_t: Teacher velocity targets, shape ``(B, n, d)``.
        :param t_batch: Time values, shape ``(B,)``.
        :param key: JAX random key for dropout.
        :return: ``(updated_state, loss_scalar)``.
        """
        subkey, _ = random.split(key)

        def loss_fn(params):
            v_pred = state.apply_fn(
                {"params": params},
                point_cloud=x_t,
                t=t_batch,
                masks=jnp.ones((x_t.shape[0], x_t.shape[1])),
                deterministic=False,
                dropout_rng=subkey,
            )
            # Mean squared error over all elements (batch × points × dim)
            error = jnp.square(v_pred - v_t)
            return jnp.mean(error)

        loss, grads = jax.value_and_grad(loss_fn)(state.params)
        state = state.apply_gradients(grads=grads)
        return state, loss

    def nn_train_step(
        self,
        x0_batch: jnp.ndarray,
        x1_batch: jnp.ndarray,
        t_scalar: float,
        key=random.key(0),
        x1_weights: jnp.ndarray = None,
    ) -> float:
        """Full training step: teacher path then NN parameter update.

        Computes the GW gradient-flow interpolant ``(X_t, v_t)`` from the
        given source and target clouds, then calls the JIT-compiled NN
        gradient step.  ``self.state`` and ``self.params`` are updated
        in-place.

        :param x0_batch: Noise point clouds, shape ``(B, n, d)``.
        :param x1_batch: Data point clouds, shape ``(B, m, d)``.
        :param t_scalar: Time value in (0, 1].
        :param key: JAX random key.
        :return: Scalar loss value.
        """
        aug_key, nn_key = random.split(key)
        x_t_batch, v_t_batch = self.teacher_step_batch(
            x0_batch, x1_batch, t_scalar, key=aug_key, x1_weights=x1_weights
        )
        t_batch = jnp.full((x0_batch.shape[0],), t_scalar)
        self.state, loss = self._nn_grad_step(
            self.state, x_t_batch, v_t_batch, t_batch, key=nn_key
        )
        self.params = self.state.params
        return loss

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def train(
        self,
        training_steps: int = 10000,
        batch_size: int = 8,
        verbose: int = 50,
        learning_rate: float = 2e-4,
        decay_steps: int = 1000,
        saved_state=None,
        key=random.key(0),
    ):
        """Train the GW-GFM model.

        For each step:
          1. Sample a mini-batch of data point clouds X1.
          2. Sample a mini-batch of noise point clouds X0.
          3. Sample a random time t ∈ (0, 1].
          4. Integrate the GW gradient flow to obtain (X_t, v_t).
          5. Update the Set Transformer to minimise ||f_theta(X_t, t) - v_t||².

        :param training_steps: Total number of gradient steps.
        :param batch_size: Number of (X0, X1) pairs per step.
        :param verbose: Print loss every ``verbose`` steps.
        :param learning_rate: Initial Adam learning rate.
        :param decay_steps: Steps per LR decay cycle.
        :param saved_state: Resume from a previously saved :class:`TrainState`.
        :param key: JAX random seed.
        """
        subkey, key = random.split(key)

        if saved_state is None:
            self.state = self.create_train_state(
                learning_rate=learning_rate,
                decay_steps=decay_steps,
                key=subkey,
            )
        else:
            self.state = saved_state
            print(f"Resuming training from step {int(self.state.step)}")

        n_samples = self.point_clouds.shape[0]
        self.losses = []

        start_step = int(self.state.step)
        tq = trange(training_steps - start_step, leave=True, desc="")

        for step in tq:
            subkey, key = random.split(key)

            # Sample a mini-batch of data point clouds (X1)
            data_ind = random.choice(subkey, n_samples, shape=(batch_size,))
            x1_batch = self.point_clouds[data_ind]
            x1_weights_batch = self.weights[data_ind]

            # Sample noise point clouds (X0)
            subkey, key = random.split(key)
            x0_raw = self.noise_func(
                size=[batch_size, x1_batch.shape[1], self.space_dim],
                noise_config=self.noise_config,
                key=subkey,
            )
            if isinstance(x0_raw, tuple):
                x0_batch = x0_raw[0]
            else:
                x0_batch = x0_raw

            # Sample time t ∈ (teacher_dt, 1]
            subkey, key = random.split(key)
            t_scalar = float(
                random.uniform(subkey, (), minval=self.config.teacher_dt, maxval=1.0)
            )

            # Teacher path + NN gradient step
            subkey, key = random.split(key)
            loss = self.nn_train_step(x0_batch, x1_batch, t_scalar, key=subkey, x1_weights=x1_weights_batch)

            self.losses.append(float(loss))

            if step % verbose == 0:
                tq.set_description("{:.3e}".format(loss))

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @partial(jit, static_argnums=(0,))
    def get_flow(
        self,
        params,
        point_cloud: jnp.ndarray,
        t: jnp.ndarray,
    ) -> jnp.ndarray:
        """Evaluate the learned velocity field at (point_cloud, t).

        :param params: Flax parameter pytree.
        :param point_cloud: ``(B, n, d)`` or ``(n, d)`` input.
        :param t: ``(B,)`` or scalar time values.
        :return: Velocity field of the same shape as ``point_cloud``.
        """
        squeeze = point_cloud.ndim == 2
        if squeeze:
            point_cloud = point_cloud[None, :, :]
            t = jnp.atleast_1d(t)

        v = self.model.apply(
            {"params": params},
            point_cloud=point_cloud,
            t=t * jnp.ones(point_cloud.shape[0]),
            masks=jnp.ones((point_cloud.shape[0], point_cloud.shape[1])),
            deterministic=True,
        )
        return jnp.squeeze(v, axis=0) if squeeze else v

    def generate_samples(
        self,
        num_samples: int = 10,
        size: int = None,
        timesteps: int = 100,
        init_noise: jnp.ndarray = None,
        key=random.key(0),
    ) -> tuple:
        """Generate point clouds by integrating f_theta from t=0 to t=1.

        Uses the RK2 midpoint method for improved accuracy.

        :param num_samples: Number of point clouds to generate.
        :param size: Number of points per cloud; defaults to the training data size.
        :param timesteps: Number of integration steps.
        :param init_noise: Optional ``(num_samples, size, d)`` initial noise;
            sampled automatically if ``None``.
        :param key: JAX random key.
        :return: ``(trajectory, weights)`` where *trajectory* is a list of
            ``(num_samples, size, d)`` arrays (one per timestep) and *weights*
            is ``(num_samples, size)``.
        """
        if size is None:
            size = self.point_clouds.shape[1]

        key, subkey = random.split(key)

        # Initial noise
        if init_noise is not None:
            if init_noise.ndim == 2:
                init_noise = init_noise[None, :, :]
            x = init_noise
        else:
            x = self.noise_func(
                size=[num_samples, size, self.space_dim],
                noise_config=self.noise_config,
                key=subkey,
            )
            if isinstance(x, tuple):
                x = x[0]

        weights = jnp.ones([num_samples, size]) / size
        trajectory = [x]

        dt = 1.0 / timesteps
        for t_val in tqdm(
            jnp.linspace(0.0, 1.0 - dt, timesteps), desc="Generating"
        ):
            t_curr = jnp.full((num_samples,), float(t_val))
            t_mid_val = jnp.full((num_samples,), float(t_val) + 0.5 * dt)

            # RK2 midpoint
            vt = self.get_flow(self.params, x, t_curr)
            x_mid = x + 0.5 * dt * vt
            v_mid = self.get_flow(self.params, x_mid, t_mid_val)
            x = x + dt * v_mid
            trajectory.append(x)

        return trajectory, weights

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_model(self, path: str):
        """Serialise the current parameters to a pickle file.

        :param path: Destination file path.
        """
        with open(path, "wb") as f:
            pickle.dump(self.params, f)
        print(f"Model saved to {path}")

    def load_model(self, path: str):
        """Load parameters from a pickle file.

        :param path: Source file path.
        """
        print(f"Loading model from {path}")
        with open(path, "rb") as f:
            self.params = pickle.load(f)

        self.state = self.create_train_state(
            learning_rate=1.0,
            decay_steps=1,
            key=random.key(0),
        )
        self.state = self.state.replace(params=self.params)
