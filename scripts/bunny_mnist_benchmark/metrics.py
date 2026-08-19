"""Fast, mask-aware point-cloud metrics on a mesh (spectral) base distance.

All distances between two surface points x, y use the mesh spectral (biharmonic) metric:
    d(x, y)   = || E(x) - E(y) ||          (spectral distance)
    d2(x, y)  = || E(x) - E(y) ||^2        (squared spectral distance)
where E = geom._embed is the k-dim spectral embedding. ``geom._embed`` performs the nearest-
triangle projection internally, so embedding an off-surface point equals projecting it onto the
mesh and then embedding -- i.e. the "project generated clouds onto the mesh" step is implicit.

Cloud-level distances (all support clouds of different sizes via boolean masks):
    - Chamfer distance (CD): squared spectral distance, symmetric two-way nearest-neighbour mean.
    - Earth Mover's distance (EMD): entropic-regularised optimal transport cost with the spectral
      distance as ground cost and uniform marginals over the (masked) points.

Benchmark statistics computed from the pairwise cloud-distance matrices:
    - 1-NNA (1-nearest-neighbour accuracy): leave-one-out; for each cloud, does its nearest cloud
      share its real/fake label? Reported for real clouds, fake clouds, and their average.
    - MMD (minimum matching distance): for each real cloud, distance to its nearest generated cloud,
      averaged. Lower is better.
"""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.special import logsumexp

_BIG = 1e12


# --------------------------------------------------------------------------------------------------
# spectral embedding (chunked -- geom._embed materialises (n_faces, 3) per point)
# --------------------------------------------------------------------------------------------------
def embed_clouds(geom, clouds, chunk=8):
    """Spectral embeddings of a stack/list of equal-size clouds.

    :param clouds: (M, K, 3) array or list of (K, 3) arrays (K may differ per call, not per cloud).
    :returns: (E, mask) with E (M, K, k) jnp array and mask (M, K) bool (all True here).
    """
    C = jnp.asarray(np.stack(clouds) if isinstance(clouds, (list, tuple)) else clouds)
    embed_pc = jax.jit(jax.vmap(jax.vmap(geom._embed)))  # (chunk, K, 3) -> (chunk, K, k)
    outs = []
    for s in range(0, C.shape[0], chunk):
        outs.append(np.asarray(embed_pc(C[s:s + chunk])))
    E = jnp.asarray(np.concatenate(outs, axis=0))
    mask = jnp.ones(E.shape[:2], dtype=bool)
    return E, mask


# --------------------------------------------------------------------------------------------------
# pairwise point cost + single-pair cloud distances
# --------------------------------------------------------------------------------------------------
def _sqdist(Ei, Ej):
    """Pairwise squared spectral distance between point sets Ei (Ki, k) and Ej (Kj, k)."""
    sqi = jnp.sum(Ei ** 2, axis=-1)
    sqj = jnp.sum(Ej ** 2, axis=-1)
    d2 = sqi[:, None] + sqj[None, :] - 2.0 * (Ei @ Ej.T)
    return jnp.maximum(d2, 0.0)


def _chamfer(Ei, mi, Ej, mj):
    """Symmetric masked Chamfer distance (squared spectral) between two clouds."""
    d2 = _sqdist(Ei, Ej)
    d2_cols = d2 + (~mj)[None, :] * _BIG          # ignore padded target points
    min_ij = jnp.min(d2_cols, axis=1)             # nearest j for each i
    cd_i = jnp.sum(jnp.where(mi, min_ij, 0.0)) / jnp.maximum(jnp.sum(mi), 1)
    d2_rows = d2 + (~mi)[:, None] * _BIG          # ignore padded source points
    min_ji = jnp.min(d2_rows, axis=0)             # nearest i for each j
    cd_j = jnp.sum(jnp.where(mj, min_ji, 0.0)) / jnp.maximum(jnp.sum(mj), 1)
    return cd_i + cd_j


def _emd(Ei, mi, Ej, mj, eps, n_iter):
    """Masked entropic-OT (EMD) cost between two clouds; ground cost = spectral distance."""
    C = jnp.sqrt(_sqdist(Ei, Ej) + 1e-12)         # spectral distance ground cost
    a = mi / jnp.maximum(jnp.sum(mi), 1)          # uniform marginals over valid points
    b = mj / jnp.maximum(jnp.sum(mj), 1)
    scale = jnp.max(C) + 1e-12                     # scale-invariant epsilon (like ott max_cost)
    Cn = C / scale
    la = jnp.log(a + 1e-30)                        # padded points -> ~ -69 -> negligible mass
    lb = jnp.log(b + 1e-30)

    def body(fg, _):
        f, g = fg
        g = eps * (lb - logsumexp((f[:, None] - Cn) / eps, axis=0))
        f = eps * (la - logsumexp((g[None, :] - Cn) / eps, axis=1))
        return (f, g), None

    (f, g), _ = jax.lax.scan(
        body, (jnp.zeros(Ei.shape[0]), jnp.zeros(Ej.shape[0])), None, length=n_iter)
    P = jnp.exp((f[:, None] + g[None, :] - Cn) / eps)
    return jnp.sum(P * Cn) * scale


# --------------------------------------------------------------------------------------------------
# blocks of the pairwise cloud-distance matrix (chunked over rows)
# --------------------------------------------------------------------------------------------------
@partial(jax.jit, static_argnums=())
def _cd_block(EA, mA, EB, mB):
    row = lambda ei, mi: jax.vmap(lambda ej, mj: _chamfer(ei, mi, ej, mj))(EB, mB)
    return jax.vmap(row)(EA, mA)


@partial(jax.jit, static_argnums=(4, 5))
def _emd_block(EA, mA, EB, mB, eps, n_iter):
    row = lambda ei, mi: jax.vmap(lambda ej, mj: _emd(ei, mi, ej, mj, eps, n_iter))(EB, mB)
    return jax.vmap(row)(EA, mA)


def block_matrix(EA, mA, EB, mB, kind, chunk, eps=0.05, n_iter=100):
    """Pairwise cloud-distance block between set A (rows) and set B (cols).

    :param kind: 'cd' or 'emd'.
    :returns: (|A|, |B|) numpy array.
    """
    rows = []
    for s in range(0, EA.shape[0], chunk):
        ea, ma = EA[s:s + chunk], mA[s:s + chunk]
        if kind == 'cd':
            blk = _cd_block(ea, ma, EB, mB)
        else:
            blk = _emd_block(ea, ma, EB, mB, eps, n_iter)
        rows.append(np.asarray(blk))
    return np.concatenate(rows, axis=0)


# --------------------------------------------------------------------------------------------------
# benchmark statistics from a full (M, M) cloud-distance matrix
# --------------------------------------------------------------------------------------------------
def one_nna(D, n_real):
    """1-nearest-neighbour accuracy. Labels: first n_real are real (1), rest fake (0).

    :returns: dict with 'real', 'fake', 'avg' accuracies.
    """
    M = D.shape[0]
    Dd = np.array(D, dtype=np.float64)
    np.fill_diagonal(Dd, _BIG)                    # leave-one-out (exclude self)
    nn = np.argmin(Dd, axis=1)
    labels = np.concatenate([np.ones(n_real), np.zeros(M - n_real)]).astype(bool)
    same = labels[nn] == labels
    real_acc = float(np.mean(same[:n_real]))
    fake_acc = float(np.mean(same[n_real:]))
    return {'real': real_acc, 'fake': fake_acc, 'avg': 0.5 * (real_acc + fake_acc)}


def mmd(D_real_fake):
    """Minimum matching distance: mean over real clouds of the nearest generated cloud distance."""
    return float(np.mean(np.min(np.asarray(D_real_fake), axis=1)))


def assemble_full(rr, rf, ff):
    """Assemble the symmetric (M, M) matrix from real-real, real-fake, fake-fake blocks."""
    top = np.concatenate([rr, rf], axis=1)
    bot = np.concatenate([rf.T, ff], axis=1)
    return np.concatenate([top, bot], axis=0)
