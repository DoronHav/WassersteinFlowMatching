"""The six benchmark model variants (mirrors bunny_mnist_benchmark/models.py).

+---------+---------------+-----------+----------------+
| model   | geometry      | monge_map | mini-batch OT  |
+---------+---------------+-----------+----------------+
| rwefm   | learned       | entropic  | on  (OT)       |
| setrfm  | learned       | random    | off (random)   |
| wfm     | euclidean     | entropic  | on  (OT)       |
| setfm   | euclidean     | random    | off (random)   |
| rwsfm   | learned       | sample    | on  (OT)       |
| wsfm    | euclidean     | sample    | on  (OT)       |
+---------+---------------+-----------+----------------+

- rwefm  : Riemannian Wasserstein (entropic) Flow Matching on the learned geometry -- the
           experiment's primary method.
- wfm    : plain Wasserstein Flow Matching in ambient X_pca (Euclidean, no learned geometry).
- setrfm : rwefm with random couplings instead of OT (no inner/outer OT) -- isolates the
           geometry's contribution with OT off.
- setfm  : wfm with random couplings -- double baseline (no geometry, no OT).
- rwsfm  : rwefm but the inner OT plan is realised by stochastic sampling ('sample' map).
- wsfm   : wfm but with the 'sample' inner map.

"learned" geometry is one of two interchangeable backends, selected by ``backend``:
  - 'pullback'  : ``utils_PullbackFlow.PullbackFlow`` (invertible flow, closed-form primitives).
  - 'neuralfim' : ``utils_NeuralFIM.NeuralFIM`` (Fasina et al. ICML 2023 Fisher-Rao sphere map;
                  non-invertible, so log/exp are autodiff + marching-scan -- pass
                  ``n_interpolation_steps`` to bound that scan's cost).

Both backends have an identity ``project_to_geometry`` (ambient X_pca is unconstrained, not a
surface), so there is no "project generated clouds" distinction to track, unlike the bunny mesh.
"""

import numpy as np

from wassersteinflowmatching.riemannian_wasserstein import (
    NeuralFIMWassersteinFlowMatching,
    PullbackFlowWassersteinFlowMatching,
    RiemannianWassersteinFlowMatching,
)

MODEL_NAMES = ('rwefm', 'wfm', 'setrfm', 'setfm', 'rwsfm', 'wsfm')
BACKENDS = ('pullback', 'neuralfim')


def _learned_model(backend, monge_map, mini_batch_ot_mode, train_clouds, geom_net, geom_params,
                   n_interpolation_steps, common):
    if backend == 'pullback':
        return PullbackFlowWassersteinFlowMatching(
            point_clouds=train_clouds, flow_net=geom_net, flow_params=geom_params,
            monge_map=monge_map, mini_batch_ot_mode=mini_batch_ot_mode,
            **({'num_sinkhorn_iters': -1, 'mini_batch_ot_num_iter': -1}
               if mini_batch_ot_mode else {}), **common)
    if backend == 'neuralfim':
        return NeuralFIMWassersteinFlowMatching(
            point_clouds=train_clouds, fim_net=geom_net, fim_params=geom_params,
            n_interpolation_steps=n_interpolation_steps,
            monge_map=monge_map, mini_batch_ot_mode=mini_batch_ot_mode,
            **({'num_sinkhorn_iters': -1, 'mini_batch_ot_num_iter': -1}
               if mini_batch_ot_mode else {}), **common)
    raise ValueError(f"unknown backend {backend!r}; choose from {BACKENDS}")


def build_model(name, train_clouds, geom_net, geom_params, backend='pullback',
                n_interpolation_steps=32, conditioning=None):
    """Construct one benchmark model.

    :param geom_net: pretrained geometry architecture (untrained; stateless) -- a
        ``utils_PullbackFlow.PullbackFlowNet`` or ``utils_NeuralFIM.NeuralFIMEncoder`` matching
        ``backend``. Unused for 'wfm'/'setfm'/'wsfm' (plain euclidean, no learned geometry).
    :param geom_params: trained parameters matching ``geom_net``.
    :param backend: 'pullback' or 'neuralfim' (see module docstring).
    :param n_interpolation_steps: marching-scan substeps, only meaningful for backend='neuralfim'.
    :param conditioning: optional (n_train_clouds, cond_dim) array, e.g. each niche's anchor-cell
        embedding, aligned with ``train_clouds``. When given, the library forces
        ``mini_batch_ot_mode=False`` internally regardless of the value requested below (see
        ``RiemannianWassersteinFlowMatching.__init__``) -- conditioning replaces cloud-level OT
        matching as the noise/data correspondence signal. Classifier-free guidance is disabled
        (``cfg=False``): every training step conditions on the real anchor (no null-conditioning
        dropout), and generation uses the conditional flow directly with no cond/uncond
        extrapolation -- see the mode-collapse diagnosis that motivated this (learned-geometry
        conditional generation was pulled toward the distribution centroid under CFG's linear
        cond/uncond extrapolation).
    :returns: model instance.
    """
    common = dict(noise_type='ambient_gaussian', cpu_projection=False)
    if conditioning is not None:
        common['conditioning'] = conditioning
        common['cfg'] = False

    if name == 'rwefm':
        return _learned_model(backend, 'entropic', True, train_clouds, geom_net, geom_params,
                              n_interpolation_steps, common)

    if name == 'setrfm':
        return _learned_model(backend, 'random', False, train_clouds, geom_net, geom_params,
                              n_interpolation_steps, common)

    if name == 'rwsfm':
        return _learned_model(backend, 'sample', True, train_clouds, geom_net, geom_params,
                              n_interpolation_steps, common)

    if name == 'wfm':
        return RiemannianWassersteinFlowMatching(
            point_clouds=train_clouds, geom='euclidean', monge_map='entropic',
            mini_batch_ot_mode=True, num_sinkhorn_iters=-1, mini_batch_ot_num_iter=-1, **common)

    if name == 'setfm':
        return RiemannianWassersteinFlowMatching(
            point_clouds=train_clouds, geom='euclidean', monge_map='random',
            mini_batch_ot_mode=False, **common)

    if name == 'wsfm':
        return RiemannianWassersteinFlowMatching(
            point_clouds=train_clouds, geom='euclidean', monge_map='sample',
            mini_batch_ot_mode=True, num_sinkhorn_iters=-1, mini_batch_ot_num_iter=-1, **common)

    raise ValueError(f"unknown model {name!r}; choose from {MODEL_NAMES}")


def generate_clouds(model, num, n_pts, key, gen_batch=250, timesteps=100, conditioning=None):
    """Generate ``num`` clouds of ``n_pts`` points, batching to bound memory.

    :param timesteps: number of Euler steps integrating the learned flow from noise to data.
    :param conditioning: optional (num, cond_dim) array -- e.g. the real anchor-cell embeddings
        to condition each generated niche on, so generation is tied one-to-one to a specific real
        niche rather than drawn unconditionally. Sliced per batch and passed as
        ``generate_conditioning`` to ``model.generate_samples``.
    :returns: (num, n_pts, d) numpy array of generated point clouds.
    """
    import jax

    out = []
    got = 0
    b = 0
    while got < num:
        n = int(min(gen_batch, num - got))
        subkey = jax.random.fold_in(key, b)
        cond_batch = None if conditioning is None else conditioning[got:got + n]
        # generate_samples returns a 3-tuple (samples, weights, conditioning) when conditioning
        # is active, vs. a 2-tuple otherwise -- only the samples are needed here.
        samples = model.generate_samples(
            num_samples=n, size=n_pts, timesteps=timesteps, key=subkey,
            generate_conditioning=cond_batch)[0]
        final = np.array(samples[-1])                    # (n, n_pts, d)
        out.append(final)
        got += n
        b += 1
    return np.concatenate(out, axis=0)[:num]
