"""
Learned pullback-flow geometry for Riemannian Wasserstein Flow Matching.

Implements an invertible (RealNVP-style) flow phi: R^d -> R^d, trained with a triplet/contrastive
metric-learning loss on a kNN graph of the input data: true graph neighbors are pulled together
and random pairs pushed apart in phi's output space (see :func:`train_pullback_flow`). Because
phi is invertible, every RWEFM Riemannian primitive (distance, log/exp map,
geodesic interpolant, tangent norm) is closed-form: straight lines in the flat latent space,
pulled back through phi^-1. No marching-scan retraction is needed (contrast with
``utils_Mesh.TriangleMesh`` / ``utils_Metric.generic_riemannian``, whose premetrics have no known
global isometry to a flat space and so approximate log/exp via multi-step retraction).

This answers the "learned, not analytically prescribed, high-dimensional geometry" regime raised
in the UosR-rebuttal experiment (claude_md/learned_geometry_niche_experiment.md, Sec. 6.2), with
training cost close to the Euclidean baseline's for every model variant -- unlike a non-invertible
learned embedding (e.g. Neural FIM's Fisher-Rao sphere map), whose lack of closed-form log/exp
would force the same expensive scan the mesh geometry needs.
"""

import numpy as np  # type: ignore
import jax  # type: ignore
import jax.numpy as jnp  # type: ignore
from jax import random  # type: ignore
from flax import linen as nn  # type: ignore


# ##################################################################################################
# Diffusion-geometry training target (numpy/scipy, one-time preprocessing)
# ##################################################################################################

def build_diffusion_sketch(X, k_knn=15, n_landmarks=150, diffusion_t=3, seed=0):
    """Landmark-diffusion sketch of a kNN graph on X (a la PHATE / landmark diffusion maps).

    Builds a symmetrized Gaussian-kernel kNN graph, row-normalizes it into a random-walk
    transition matrix T, then propagates one-hot indicators at a set of landmark points through
    T for ``diffusion_t`` steps. Row i of the result approximates the diffusion-t-step transition
    distribution from point i to each landmark; ``||R_i - R_j||`` approximates the diffusion
    distance between points i and j.

    :param X: (n, d) array of points (e.g. all-cell ``X_pca``).
    :param k_knn: number of nearest neighbors per point in the kNN graph.
    :param n_landmarks: number of landmark points (kMeans-selected, snapped to real data points).
    :param diffusion_t: number of random-walk steps to propagate.
    :param seed: random seed for landmark selection.
    :returns: (R (n, n_landmarks) float32, landmark_idx (n_landmarks,) int64 indices into X,
        knn_idx (n, k_knn) int32 indices of each point's k nearest neighbors -- used by
        :func:`train_pullback_flow` to sample genuine local-neighbor pairs into each batch).
    """
    from sklearn.neighbors import NearestNeighbors  # type: ignore
    from sklearn.cluster import MiniBatchKMeans  # type: ignore
    from scipy import sparse  # type: ignore

    X = np.asarray(X, dtype=np.float64)
    n = X.shape[0]
    k_knn = int(min(k_knn, n - 1))

    nbrs = NearestNeighbors(n_neighbors=k_knn + 1).fit(X)
    dist, idx = nbrs.kneighbors(X)
    dist, idx = dist[:, 1:], idx[:, 1:]  # drop self (first neighbor)

    sigma = np.median(dist[:, -1]) + 1e-8
    w = np.exp(-(dist ** 2) / (2 * sigma ** 2))
    rows = np.repeat(np.arange(n), k_knn)
    cols = idx.reshape(-1)
    A = sparse.csr_matrix((w.reshape(-1), (rows, cols)), shape=(n, n))
    A = A.maximum(A.T)  # symmetrize

    deg = np.asarray(A.sum(axis=1)).ravel()
    deg[deg <= 0] = 1.0
    T = sparse.diags(1.0 / deg) @ A  # row-stochastic transition matrix

    n_landmarks = int(min(n_landmarks, n))
    km = MiniBatchKMeans(n_clusters=n_landmarks, random_state=seed, n_init=3).fit(X)
    _, landmark_idx = NearestNeighbors(n_neighbors=1).fit(X).kneighbors(km.cluster_centers_)
    landmark_idx = landmark_idx[:, 0]

    R = np.zeros((n, n_landmarks), dtype=np.float64)
    R[landmark_idx, np.arange(n_landmarks)] = 1.0
    for _ in range(diffusion_t):
        R = T @ R
    return R.astype(np.float32), landmark_idx, idx.astype(np.int32)


# ##################################################################################################
# Invertible coupling flow (RealNVP-style)
# ##################################################################################################

class CouplingLayer(nn.Module):
    """One affine coupling layer with a fixed checkerboard mask.

    The conditioner only ever consumes the masked (passthrough) half of the input, which is
    identical between a point and its forward image, so the exact inverse is recovered
    algebraically from the same weights (standard RealNVP). The final conditioner layer is
    zero-initialized so the flow starts as (approximately) the identity map.
    """
    dim: int
    hidden_dim: int
    parity: int

    @nn.compact
    def __call__(self, x, reverse=False):
        idx = jnp.arange(self.dim)
        mask = (idx % 2 == (self.parity % 2)).astype(x.dtype)

        x_pass = x * mask
        h = nn.Dense(self.hidden_dim)(x_pass)
        h = nn.relu(h)
        h = nn.Dense(self.hidden_dim)(h)
        h = nn.relu(h)
        st = nn.Dense(2 * self.dim, kernel_init=nn.initializers.zeros,
                     bias_init=nn.initializers.zeros)(h)
        log_s, t = jnp.split(st, 2, axis=-1)
        log_s = jnp.tanh(log_s) * (1.0 - mask)  # bounded, zero on the passthrough half
        t = t * (1.0 - mask)

        if not reverse:
            y = x_pass + (1.0 - mask) * (x * jnp.exp(log_s) + t)
            return y, log_s
        x_rec = x_pass + (1.0 - mask) * ((x - t) * jnp.exp(-log_s))
        return x_rec, log_s


class PullbackFlowNet(nn.Module):
    """Stack of alternating-checkerboard-mask coupling layers.

    ``forward``/``inverse`` are exact inverses of each other and share the same parameters
    (submodules are declared once in ``setup``).
    """
    dim: int
    hidden_dim: int = 128
    n_layers: int = 6

    def setup(self):
        self.layers = [
            CouplingLayer(dim=self.dim, hidden_dim=self.hidden_dim, parity=i % 2)
            for i in range(self.n_layers)
        ]

    def forward(self, x):
        """x -> (z, log_det) where log_det is the total log-Jacobian-determinant of the flow."""
        log_det = jnp.zeros(x.shape[:-1])
        for layer in self.layers:
            x, log_s = layer(x, reverse=False)
            log_det = log_det + jnp.sum(log_s, axis=-1)
        return x, log_det

    def inverse(self, z):
        """z -> x, the exact inverse of ``forward`` (log-det discarded)."""
        for layer in reversed(self.layers):
            z, _ = layer(z, reverse=True)
        return z


def train_pullback_flow(X, knn_idx, hidden_dim=128, n_layers=6, steps=3000, batch_size=512,
                        margin_frac=0.5, lambda_iso=0.01, lr=1e-3, key=None, verbose=200):
    """Train phi with a triplet/contrastive metric-learning loss on the data's kNN graph.

    For each of ``batch_size // 2`` random anchor points, draws one true kNN neighbor (positive,
    from ``knn_idx``) and one independent random point (negative -- with n in the hundreds of
    thousands and k_knn << n, the false-negative rate from an occasional true-neighbor draw is
    negligible). Trains phi so ``||phi(anchor)-phi(positive)||`` is smaller than
    ``||phi(anchor)-phi(negative)||`` by a margin, i.e. a standard triplet loss pulling genuine
    local (kNN-graph) neighbors together and pushing random pairs apart in the latent space.

    This directly optimizes local neighborhood preservation -- the property the geometry actually
    needs for RWEFM and the one a global pairwise-distance regression can trade away in favor of
    coarse/large-scale fit. The margin is calibrated per batch (stop-gradient) as a fraction of the
    typical anchor-negative squared distance, so it adapts to the ambient feature scale.

    The isometry/conditioning regularizer penalizes the flow's total log-Jacobian-determinant
    (accumulated in ``PullbackFlowNet.forward``'s ``log_det``), discouraging large volume changes.

    :param X: (n, d) training points (e.g. all-cell ``X_pca``).
    :param knn_idx: (n, k_knn) neighbor indices, e.g. from :func:`build_diffusion_sketch`, same
        row order as X.
    :returns: (net, params, losses) -- ``net`` is the (stateless) :class:`PullbackFlowNet`
        definition, ``params`` the trained parameters, ``losses`` a (steps,) numpy array.
    """
    import optax  # type: ignore
    from tqdm import trange  # type: ignore

    if key is None:
        key = random.key(0)

    X = jnp.asarray(X, dtype=jnp.float32)
    knn_idx = jnp.asarray(knn_idx, dtype=jnp.int32)
    n, d = X.shape
    k_knn = knn_idx.shape[1]
    half = int(min(batch_size, n)) // 2

    net = PullbackFlowNet(dim=d, hidden_dim=hidden_dim, n_layers=n_layers)
    subkey, key = random.split(key)
    params = net.init(subkey, X[:2], method=PullbackFlowNet.forward)['params']

    # Fixed (not per-batch-adaptive) margin, from the *ambient* data before any training: the
    # isometry regularizer only constrains the coupling layers' log-scale (log-det), not their
    # translation shifts, so a margin computed from the network's own (evolving) negative
    # distances creates a runaway feedback loop -- the flow can translate negatives arbitrarily
    # far apart, which inflates the margin, which rewards translating them even farther. Fixing
    # the margin from data once removes that loop; the flow starts near-identity (zero-init
    # coupling layers), so "ambient" is also the initial latent scale.
    subkey, key = random.split(key)
    pair_idx = random.choice(subkey, n, shape=(4096, 2))
    margin = margin_frac * jnp.median(
        jnp.sum((X[pair_idx[:, 0]] - X[pair_idx[:, 1]]) ** 2, axis=-1))

    tx = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(lr))
    opt_state = tx.init(params)

    def loss_fn(params, anchor_x, pos_x, neg_x):
        stacked = jnp.concatenate([anchor_x, pos_x, neg_x], axis=0)
        z, log_det = net.apply({'params': params}, stacked, method=PullbackFlowNet.forward)
        z_anchor, z_pos, z_neg = jnp.split(z, 3, axis=0)

        d_pos_sq = jnp.sum((z_anchor - z_pos) ** 2, axis=-1)
        d_neg_sq = jnp.sum((z_anchor - z_neg) ** 2, axis=-1)
        triplet = jnp.mean(jax.nn.relu(d_pos_sq - d_neg_sq + margin))

        iso = jnp.mean(log_det ** 2)
        return triplet + lambda_iso * iso

    @jax.jit
    def step(params, opt_state, anchor_x, pos_x, neg_x):
        loss, grads = jax.value_and_grad(loss_fn)(params, anchor_x, pos_x, neg_x)
        updates, opt_state = tx.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    losses = []
    tq = trange(steps, desc="pullback-flow", leave=True)
    for i in tq:
        subkey_a, subkey_n, subkey_neg, key = random.split(key, 4)
        anchors = random.choice(subkey_a, n, shape=(half,), replace=False)
        neighbor_choice = random.randint(subkey_n, shape=(half,), minval=0, maxval=k_knn)
        positives = knn_idx[anchors, neighbor_choice]
        negatives = random.choice(subkey_neg, n, shape=(half,), replace=False)
        params, opt_state, loss = step(params, opt_state, X[anchors], X[positives], X[negatives])
        losses.append(float(loss))
        if verbose and i % verbose == 0:
            tq.set_description(f"pullback-flow: {loss:.3e}")

    return net, params, np.asarray(losses)


# ##################################################################################################
# RWEFM geometry primitives (closed-form, pulled back through the invertible flow)
# ##################################################################################################

class PullbackFlow:
    """Riemannian geometry on ambient space, pulled back from a flat latent space through a
    pretrained invertible flow ``net``/``params`` (see :func:`train_pullback_flow`).

    Every primitive reduces to a straight line / Jacobian-vector-product in the latent space,
    mapped back through ``phi^-1`` -- no marching-scan retraction, so this class does not
    subclass :class:`utils_Metric.generic_riemannian` (there is no scan configuration to share).

    :param net: a :class:`PullbackFlowNet` architecture definition (untrained; stateless).
    :param params: trained parameters matching ``net``.
    """

    def __init__(self, net, params):
        self.net = net
        self.params = params

    def _phi(self, p):
        z, _ = self.net.apply({'params': self.params}, p, method=PullbackFlowNet.forward)
        return z

    def _phi_inv(self, z):
        return self.net.apply({'params': self.params}, z, method=PullbackFlowNet.inverse)

    def project_to_geometry(self, P, use_cpu=False):
        # Ambient (X_pca) space is unconstrained -- no manifold constraint to project onto.
        return P

    def distance(self, P0, P1):
        z0, z1 = self._phi(P0), self._phi(P1)
        return jnp.sum((z0 - z1) ** 2)

    def distance_matrix(self, P0, P1):
        """O(k) distance matrix: embed each cloud once, then a single Euclidean distance matrix."""
        E0, E1 = self._phi(P0), self._phi(P1)
        sq0 = jnp.sum(E0 ** 2, axis=-1)
        sq1 = jnp.sum(E1 ** 2, axis=-1)
        d2 = sq0[:, None] + sq1[None, :] - 2.0 * (E0 @ E1.T)
        return jnp.maximum(d2, 0.0)

    def interpolant(self, P0, P1, t):
        """Straight line in latent space -- the exact geodesic of the pulled-back flat metric."""
        z0, z1 = self._phi(P0), self._phi(P1)
        z_t = (1 - t) * z0 + t * z1
        return self._phi_inv(z_t)

    def velocity(self, P0, P1, t):
        z0, z1 = self._phi(P0), self._phi(P1)
        z_t = (1 - t) * z0 + t * z1
        _, v = jax.jvp(self._phi_inv, (z_t,), (z1 - z0,))
        return v

    def exponential_map(self, p, v, delta_t):
        z, dz = jax.jvp(self._phi, (p,), (v,))
        return self._phi_inv(z + dz * delta_t)

    def tangent_norm(self, v, w, p):
        """Pullback-metric squared distance between tangent vectors v, w at p."""
        _, dv = jax.jvp(self._phi, (p,), (v,))
        _, dw = jax.jvp(self._phi, (p,), (w,))
        return jnp.mean(jnp.square(dv - dw))

    def weighted_mean(self, points, weights):
        weights = weights / (jnp.sum(weights) + 1e-9)
        z_mean = jnp.sum(self._phi(points) * weights[:, None], axis=0)
        return self._phi_inv(z_mean)
