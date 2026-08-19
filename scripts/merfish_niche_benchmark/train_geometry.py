"""Pretrain a learned geometry on the full MERFISH motor-cortex atlas.

One-time preprocessing step (analogous to the bunny benchmark's mesh spectral-basis computation,
but with a trained neural net instead of a closed-form eigendecomposition): learns a geometry on
**all** cells (not just niche members -- the geometry must reflect the full cell-state landscape
per the experiment spec, Sec. 6). Saves it to disk; every benchmark.py job loads it rather than
retraining it.

Two interchangeable backends (``--backend``):
  - 'pullback'  (default): an invertible flow phi trained with a triplet/contrastive loss on a
    kNN graph -- closed-form log/exp in RWEFM (see ``utils_PullbackFlow``).
  - 'neuralfim': Fasina et al.'s (ICML 2023) Fisher-Rao sphere map, a non-invertible softmax MLP
    trained by cross-entropy against a landmark-diffusion sketch -- log/exp are autodiff +
    marching-scan in RWEFM (see ``utils_NeuralFIM``); pass ``--n-interpolation-steps`` to bound
    that scan's cost, calibrated empirically per the module's cost caveat.

Also runs a validation check (per the experiment spec's risk-mitigation item): compares k-NN
label-purity (``obs['subclass']``) under the learned metric vs. raw Euclidean distance in the
same feature space, to confirm the learned embedding is not degenerate before any benchmark job
consumes it.

``--feature-path`` overrides the cell-feature space (e.g. a cached alternate PCA from
``compute_pca16.py``) instead of the atlas's precomputed ``obsm['X_pca']``.

Example:
    python train_geometry.py --backend pullback --geom-steps 3000
    python train_geometry.py --backend neuralfim --geom-steps 3000
    python train_geometry.py --feature-path results/merfish_niche_benchmark/X_pca16.npy \\
        --out-path results/merfish_niche_benchmark/pca16/pullback_flow_geometry.pkl
"""

import argparse
import os
import pickle
import time

import numpy as np


def _parse_args():
    here = os.path.dirname(os.path.abspath(__file__))
    repo = os.path.abspath(os.path.join(here, '..', '..'))
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--h5ad-path', type=str,
                   default='/cv/data/braid/havivd/merfish_motor_cortex_atlas/st_data_processed.h5ad')
    p.add_argument('--feature-path', type=str, default=None,
                   help='override cell-feature space with a cached (n,d) .npy instead of '
                        "obsm['X_pca'] (e.g. compute_pca16.py's output)")
    p.add_argument('--backend', type=str, default='pullback', choices=['pullback', 'neuralfim'])
    # diffusion-sketch (training target) construction
    p.add_argument('--knn-k', type=int, default=15)
    p.add_argument('--n-landmarks', type=int, default=150)
    p.add_argument('--diffusion-t', type=int, default=3)
    # architecture / training (pullback: flow; neuralfim: encoder)
    p.add_argument('--hidden-dim', type=int, default=128)
    p.add_argument('--n-layers', type=int, default=6, help='pullback backend only')
    p.add_argument('--geom-steps', type=int, default=3000)
    p.add_argument('--geom-batch', type=int, default=512)
    p.add_argument('--margin-frac', type=float, default=0.5, help='pullback backend only')
    p.add_argument('--lambda-iso', type=float, default=0.01, help='pullback backend only')
    p.add_argument('--geom-lr', type=float, default=1e-3)
    p.add_argument('--n-interpolation-steps', type=int, default=32,
                   help='neuralfim backend only -- RWEFM marching-scan substeps; saved into the '
                        'pickle as a default but overridable at benchmark time')
    p.add_argument('--seed', type=int, default=0)
    # validation
    p.add_argument('--val-n', type=int, default=3000, help='cells sampled for the purity check')
    p.add_argument('--val-k', type=int, default=15, help='neighbors per cell in the purity check')
    p.add_argument('--out-path', type=str, default=None,
                   help='default: results/merfish_niche_benchmark/'
                        '{pullback_flow,neural_fim}_geometry.pkl depending on --backend')
    return p.parse_args()


def _knn_purity(D, labels, k):
    """Mean same-label fraction among each point's k nearest neighbors under distance matrix D."""
    Dd = np.array(D, dtype=np.float64)
    np.fill_diagonal(Dd, np.inf)
    nn = np.argpartition(Dd, kth=k, axis=1)[:, :k]
    same = (labels[nn] == labels[:, None])
    return float(same.mean())


def main():
    args = _parse_args()
    t0 = time.time()

    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'src'))
    import jax
    import jax.numpy as jnp
    from jax import random
    from niche_data import load_atlas_fields
    from wassersteinflowmatching.riemannian_wasserstein.utils_PullbackFlow import (
        build_diffusion_sketch, train_pullback_flow, PullbackFlowNet,
    )

    print(f"[geom] backend={args.backend} devices={jax.devices()}", flush=True)

    fields = load_atlas_fields(args.h5ad_path, feature_path=args.feature_path)
    X = fields['X_pca']
    print(f"[geom] loaded features {X.shape} "
          f"({'override: ' + args.feature_path if args.feature_path else 'obsm/X_pca'}) "
          f"({time.time() - t0:.0f}s)", flush=True)

    R, landmark_idx, knn_idx = build_diffusion_sketch(
        X, k_knn=args.knn_k, n_landmarks=args.n_landmarks, diffusion_t=args.diffusion_t,
        seed=args.seed)
    print(f"[geom] kNN graph + diffusion sketch {R.shape} ({time.time() - t0:.0f}s)", flush=True)

    if args.backend == 'pullback':
        net, params, losses = train_pullback_flow(
            X, knn_idx, hidden_dim=args.hidden_dim, n_layers=args.n_layers, steps=args.geom_steps,
            batch_size=args.geom_batch, margin_frac=args.margin_frac, lambda_iso=args.lambda_iso,
            lr=args.geom_lr, key=random.key(args.seed))

        def embed_fn(x):
            z, _ = net.apply({'params': params}, x, method=PullbackFlowNet.forward)
            return z
    else:
        from wassersteinflowmatching.riemannian_wasserstein.utils_NeuralFIM import (
            train_neural_fim,
        )
        net, params, losses = train_neural_fim(
            X, R, hidden_dim=args.hidden_dim, steps=args.geom_steps, batch_size=args.geom_batch,
            lr=args.geom_lr, key=random.key(args.seed))

        def embed_fn(x):
            probs = jax.nn.softmax(net.apply({'params': params}, x))
            return jnp.sqrt(probs + 1e-8)

    print(f"[geom] trained {args.backend}: loss {losses[0]:.3e} -> {losses[-1]:.3e} "
          f"({time.time() - t0:.0f}s)", flush=True)

    # ---- validation: k-NN label purity, learned metric vs. raw Euclidean --------------------
    rng = np.random.default_rng(args.seed)
    val_idx = rng.choice(X.shape[0], size=min(args.val_n, X.shape[0]), replace=False)
    X_val = X[val_idx]
    labels_val = fields['subclass_codes'][val_idx]

    Z_val = np.asarray(embed_fn(jnp.asarray(X_val)))

    if args.backend == 'pullback':
        D_learned = np.sum((Z_val[:, None, :] - Z_val[None, :, :]) ** 2, axis=-1)
    else:
        dot = np.clip(Z_val @ Z_val.T, -1.0, 1.0)
        D_learned = (2.0 * np.arccos(dot)) ** 2
    D_euclidean = np.sum((X_val[:, None, :] - X_val[None, :, :]) ** 2, axis=-1)
    purity_learned = _knn_purity(D_learned, labels_val, args.val_k)
    purity_euclidean = _knn_purity(D_euclidean, labels_val, args.val_k)
    print(f"[geom] validation (subclass k-NN purity, k={args.val_k}, n={val_idx.size}): "
          f"learned={purity_learned:.4f} euclidean={purity_euclidean:.4f} "
          f"({time.time() - t0:.0f}s)", flush=True)
    if purity_learned < 0.5 * purity_euclidean:
        print("[geom] WARNING: learned-metric purity is far below the Euclidean baseline -- "
              "the learned geometry may be degenerate. Inspect before running the benchmark.",
              flush=True)

    out = {
        'backend': args.backend, 'dim': X.shape[1], 'hidden_dim': args.hidden_dim,
        'params': params, 'landmark_idx': landmark_idx,
        'knn_k': args.knn_k, 'n_landmarks': args.n_landmarks, 'diffusion_t': args.diffusion_t,
        'geom_steps': args.geom_steps, 'losses': losses,
        'purity_learned': purity_learned, 'purity_euclidean': purity_euclidean,
        'val_k': args.val_k, 'val_n': int(val_idx.size), 'seed': args.seed,
        'feature_path': args.feature_path,
        'runtime_sec': time.time() - t0,
    }
    if args.backend == 'pullback':
        out['n_layers'] = args.n_layers
    else:
        out['n_interpolation_steps'] = args.n_interpolation_steps

    out_path = args.out_path
    if out_path is None:
        here = os.path.dirname(os.path.abspath(__file__))
        repo = os.path.abspath(os.path.join(here, '..', '..'))
        fname = 'pullback_flow_geometry.pkl' if args.backend == 'pullback' else 'neural_fim_geometry.pkl'
        out_path = os.path.join(repo, 'results', 'merfish_niche_benchmark', fname)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'wb') as f:
        pickle.dump(out, f)
    print(f"[geom] wrote {out_path} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == '__main__':
    main()
