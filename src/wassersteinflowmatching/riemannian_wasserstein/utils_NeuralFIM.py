"""
Neural FIM (Fasina et al., ICML 2023, arXiv:2306.06062) learned Fisher-Rao geometry for RWEFM.

Learns f_psi: R^d -> distribution over m diffusion-landmark anchors (a small softmax MLP), trained
to match a landmark-diffusion sketch of a kNN graph -- the same construction
``utils_PullbackFlow.build_diffusion_sketch`` computes, reused directly here. The Fisher-Rao sphere
embedding s(x) = sqrt(f_psi(x)) turns Euclidean geometry on the sphere into the Fisher-Rao distance
between predicted distributions: d(x, y) = (2*arccos(<s(x), s(y)>))^2.

Because f_psi is a non-invertible softmax map (not a bijection like ``utils_PullbackFlow.PullbackFlow``),
there is no closed-form log/exp on ambient space -- they are derived automatically via
``utils_Metric.generic_riemannian``'s autodiff gradient log-map and multi-step retraction
exponential map, exactly like ``utils_Metric.mystery_sphere``. Unlike the mesh geometry
(``utils_Mesh.TriangleMesh``), whose retraction is expensive because of an O(n_faces)
nearest-triangle search every substep, ``project_to_geometry`` here is identity (ambient X_pca is
unconstrained), so each marching substep costs one MLP forward/backward pass -- architecturally
cheap, but multiplied by ``generic_riemannian``'s substep count (default 1000). Calibrate
``n_interpolation_steps`` empirically before a full run.
"""

import numpy as np  # type: ignore
import jax  # type: ignore
import jax.numpy as jnp  # type: ignore
from jax import random  # type: ignore
from flax import linen as nn  # type: ignore

# Re-export so RiemannianWassersteinFlowMatching's
# ``getattr(self._geom_module, 'generic_riemannian', None)`` subclass check finds it.
from wassersteinflowmatching.riemannian_wasserstein.utils_Metric import generic_riemannian  # type: ignore  # noqa: F401
from wassersteinflowmatching.riemannian_wasserstein.utils_PullbackFlow import build_diffusion_sketch  # type: ignore  # noqa: F401


class NeuralFIMEncoder(nn.Module):
    """Small MLP f_psi: R^d -> logits over m landmarks (softmax applied outside)."""
    hidden_dim: int = 128
    n_landmarks: int = 150

    @nn.compact
    def __call__(self, x):
        h = nn.Dense(self.hidden_dim)(x)
        h = nn.relu(h)
        h = nn.Dense(self.hidden_dim)(h)
        h = nn.relu(h)
        return nn.Dense(self.n_landmarks)(h)


def train_neural_fim(X, R, hidden_dim=128, steps=3000, batch_size=2048, lr=1e-3, key=None,
                     verbose=200):
    """Train f_psi via cross-entropy against a landmark-diffusion sketch R.

    :param X: (n, d) training points (e.g. all-cell ``X_pca``).
    :param R: (n, m) landmark-diffusion sketch from :func:`build_diffusion_sketch` (need not be
        exactly row-stochastic; renormalized internally).
    :returns: (net, params, losses) -- ``net`` is the (stateless) :class:`NeuralFIMEncoder`
        definition, ``params`` the trained parameters, ``losses`` a (steps,) numpy array.
    """
    import optax  # type: ignore
    from tqdm import trange  # type: ignore

    if key is None:
        key = random.key(0)

    X = jnp.asarray(X, dtype=jnp.float32)
    R = jnp.asarray(R, dtype=jnp.float32)
    R = R / jnp.maximum(jnp.sum(R, axis=-1, keepdims=True), 1e-12)
    n, d = X.shape
    m = R.shape[1]
    batch_size = int(min(batch_size, n))

    net = NeuralFIMEncoder(hidden_dim=hidden_dim, n_landmarks=m)
    subkey, key = random.split(key)
    params = net.init(subkey, X[:2])['params']

    tx = optax.adam(lr)
    opt_state = tx.init(params)

    def loss_fn(params, x_batch, r_batch):
        logits = net.apply({'params': params}, x_batch)
        log_probs = jax.nn.log_softmax(logits)
        return -jnp.mean(jnp.sum(r_batch * log_probs, axis=-1))

    @jax.jit
    def step(params, opt_state, x_batch, r_batch):
        loss, grads = jax.value_and_grad(loss_fn)(params, x_batch, r_batch)
        updates, opt_state = tx.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    losses = []
    tq = trange(steps, desc="neural-fim", leave=True)
    for i in tq:
        subkey, key = random.split(key)
        batch_idx = random.choice(subkey, n, shape=(batch_size,), replace=False)
        params, opt_state, loss = step(params, opt_state, X[batch_idx], R[batch_idx])
        losses.append(float(loss))
        if verbose and i % verbose == 0:
            tq.set_description(f"neural-fim: {loss:.3e}")

    return net, params, np.asarray(losses)


class NeuralFIM(generic_riemannian):
    """Fisher-Rao sphere geometry via a learned (non-invertible) softmax embedding.

    Subclasses :class:`generic_riemannian`: log/exp/interpolant/tangent_norm/weighted_mean all
    come from autodiff + multi-step retraction (see module docstring for the cost caveat), since
    f_psi is not invertible -- there is no closed-form log/exp the way there is for
    :class:`utils_PullbackFlow.PullbackFlow`.

    :param net: a :class:`NeuralFIMEncoder` architecture definition (untrained; stateless).
    :param params: trained parameters matching ``net`` (see :func:`train_neural_fim`).
    :param n_interpolation_steps: marching-scan substeps (default 1000, per
        ``generic_riemannian``'s default -- likely worth lowering; see module docstring).
    :param n_exp_steps: substeps for the retraction exponential map (default: same as above).
    :param eps: additive smoothing inside the softmax before the sqrt (numerical safety at the
        simplex boundary).
    """

    def __init__(self, net, params, n_interpolation_steps=1000, n_exp_steps=None, eps=1e-8):
        super().__init__(n_interpolation_steps=n_interpolation_steps, n_exp_steps=n_exp_steps)
        self.net = net
        self.params = params
        self.eps = eps

    def project_to_geometry(self, P, use_cpu=False):
        # Ambient (X_pca) space is unconstrained -- no manifold constraint to project onto.
        return P

    def _embed(self, p):
        """Fisher-Rao sphere embedding s(x) = sqrt(softmax(f_psi(x)) + eps)."""
        logits = self.net.apply({'params': self.params}, p)
        probs = jax.nn.softmax(logits)
        return jnp.sqrt(probs + self.eps)

    def _squared_distance(self, p, q):
        sp, sq = self._embed(p), self._embed(q)
        dot = jnp.clip(jnp.sum(sp * sq), -1.0, 1.0)
        return (2.0 * jnp.arccos(dot)) ** 2

    def distance_matrix(self, P0, P1):
        """O(k) distance matrix: embed each cloud once, then one sphere-distance matrix."""
        E0, E1 = self._embed(P0), self._embed(P1)
        dot = jnp.clip(E0 @ E1.T, -1.0, 1.0)
        return jnp.nan_to_num((2.0 * jnp.arccos(dot)) ** 2, nan=0.0)

    def velocity_at_source(self, P0, P1):
        """velocity(P0, P1, 0) without the marching-scan interpolant call.

        At t=0 the marching interpolant is a (still O(n_interpolation_steps)) no-op scan
        (x_t == P0), so the initial conditional velocity is just the log map. Skipping the scan
        is important for the entropic monge map, which evaluates this for every source/target
        pair -- exactly the shortcut ``utils_Mesh.TriangleMesh`` uses for the same reason.
        """
        return self._log_map(P0, P1)
