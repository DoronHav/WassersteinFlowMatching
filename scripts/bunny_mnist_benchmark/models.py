"""The four benchmark model variants.

+---------+---------------+-----------+----------------+------------------------+
| model   | geometry      | monge_map | mini-batch OT  | project generated?     |
+---------+---------------+-----------+----------------+------------------------+
| rwefm   | TriangleMesh  | entropic  | on  (OT)       | already on mesh        |
| setrfm  | TriangleMesh  | random    | off (random)   | already on mesh        |
| wfm     | euclidean     | entropic  | on  (OT)       | yes (implicit in metric)|
| setfm   | euclidean     | random    | off (random)   | yes (implicit in metric)|
| rwsfm   | TriangleMesh  | sample    | on  (OT)       | already on mesh        |
| wsfm    | euclidean     | sample    | on  (OT)       | yes (implicit in metric)|
+---------+---------------+-----------+----------------+------------------------+

- rwefm  : Riemannian Wasserstein (entropic) Flow Matching -- the notebook model.
- wfm    : plain Wasserstein Flow Matching in ambient 3D (no manifold); generated clouds are
           snapped to the mesh (the snap is implicit in the spectral metric used for scoring).
- setrfm : rwefm with random couplings instead of OT (no inner/outer OT).
- setfm  : wfm with random couplings.
- rwsfm  : rwefm but the inner OT plan is realised by stochastic sampling ('sample' map).
- wsfm   : wfm but with the 'sample' inner map.
"""

import numpy as np

from wassersteinflowmatching.riemannian_wasserstein import (
    MeshWassersteinFlowMatching,
    RiemannianWassersteinFlowMatching,
)

MODEL_NAMES = ('rwefm', 'wfm', 'setrfm', 'setfm', 'rwsfm', 'wsfm')


def build_model(name, train_clouds, mesh, k=100, n_interpolation_steps=100):
    """Construct one benchmark model.

    :returns: (model, project_generated) where project_generated is True for the euclidean models.
    """
    V, F = mesh
    common = dict(noise_type='ambient_gaussian', cpu_projection=False)

    if name == 'rwefm':
        model = MeshWassersteinFlowMatching(
            point_clouds=train_clouds, mesh=(V, F), k=k, normalize=False,
            n_interpolation_steps=n_interpolation_steps, monge_map='entropic',
            mini_batch_ot_mode=True, num_sinkhorn_iters=-1, mini_batch_ot_num_iter=-1, **common)
        return model, False

    if name == 'setrfm':
        model = MeshWassersteinFlowMatching(
            point_clouds=train_clouds, mesh=(V, F), k=k, normalize=False,
            n_interpolation_steps=n_interpolation_steps, monge_map='random',
            mini_batch_ot_mode=False, **common)
        return model, False

    if name == 'wfm':
        model = RiemannianWassersteinFlowMatching(
            point_clouds=train_clouds, geom='euclidean', monge_map='entropic',
            mini_batch_ot_mode=True, num_sinkhorn_iters=-1, mini_batch_ot_num_iter=-1, **common)
        return model, True

    if name == 'setfm':
        model = RiemannianWassersteinFlowMatching(
            point_clouds=train_clouds, geom='euclidean', monge_map='random',
            mini_batch_ot_mode=False, **common)
        return model, True

    if name == 'rwsfm':
        model = MeshWassersteinFlowMatching(
            point_clouds=train_clouds, mesh=(V, F), k=k, normalize=False,
            n_interpolation_steps=n_interpolation_steps, monge_map='sample',
            mini_batch_ot_mode=True, num_sinkhorn_iters=-1, mini_batch_ot_num_iter=-1, **common)
        return model, False

    if name == 'wsfm':
        model = RiemannianWassersteinFlowMatching(
            point_clouds=train_clouds, geom='euclidean', monge_map='sample',
            mini_batch_ot_mode=True, num_sinkhorn_iters=-1, mini_batch_ot_num_iter=-1, **common)
        return model, True

    raise ValueError(f"unknown model {name!r}; choose from {MODEL_NAMES}")


def generate_clouds(model, num, n_pts, key, gen_batch=250):
    """Generate ``num`` clouds of ``n_pts`` points, batching to bound memory.

    :returns: (num, n_pts, 3) numpy array of generated point clouds.
    """
    import jax

    out = []
    got = 0
    b = 0
    while got < num:
        n = int(min(gen_batch, num - got))
        subkey = jax.random.fold_in(key, b)
        samples, weights = model.generate_samples(
            num_samples=n, size=n_pts, timesteps=100, key=subkey)
        final = np.array(samples[-1])                    # (n, n_pts, 3)
        out.append(final)
        got += n
        b += 1
    return np.concatenate(out, axis=0)[:num]
