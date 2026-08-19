"""Train one model, generate niches at a fixed timesteps, and save real+generated to disk.

Standalone helper for the niche-generation UMAP notebook (examples/validate_merfish_niche_umap.ipynb):
lets each model be trained on its own GPU as an independent background process, so the notebook
itself only needs to load the saved arrays rather than retrain inline.

Example:
    CUDA_VISIBLE_DEVICES=0 python generate_for_viz.py --model setfm
    CUDA_VISIBLE_DEVICES=1 python generate_for_viz.py --model rwsfm

    # conditional generation: each niche is generated conditioned on its own real anchor cell's
    # embedding (instead of unconditionally), and mini-batch OT is disabled -- the library forces
    # this automatically whenever conditioning is set (conditioning replaces cloud-level OT
    # matching as the noise/data correspondence signal; see RiemannianWassersteinFlowMatching).
    CUDA_VISIBLE_DEVICES=0 python generate_for_viz.py --model rwsfm --conditional
"""

import argparse
import os
import pickle
import time

import numpy as np


def _parse_args():
    here = os.path.dirname(os.path.abspath(__file__))
    repo = os.path.abspath(os.path.join(here, '..', '..'))
    bench_dir = os.path.join(repo, 'results', 'merfish_niche_benchmark')
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--model', type=str, required=True,
                   choices=['rwefm', 'wfm', 'setrfm', 'setfm', 'rwsfm', 'wsfm'])
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--n-train-steps', type=int, default=100000)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--decay-steps', type=int, default=1000)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--timesteps', type=int, default=128)
    p.add_argument('--gen-batch', type=int, default=250)
    p.add_argument('--k', type=int, default=128)
    p.add_argument('--test-n', type=int, default=1024)
    p.add_argument('--center-label', type=str, default='GABAergic')
    p.add_argument('--center-field', type=str, default='cell_label',
                   help="obs field --center-label is a category of (e.g. 'cell_label' or 'subclass')")
    p.add_argument('--h5ad-path', type=str,
                   default='/cv/data/braid/havivd/merfish_motor_cortex_atlas/st_data_processed.h5ad')
    p.add_argument('--feature-path', type=str,
                   default=os.path.join(bench_dir, 'X_pca64_selfcomputed.npy'))
    p.add_argument('--geometry-path', type=str,
                   default=os.path.join(bench_dir, 'pca64_selfcomputed', 'pullback_flow_geometry.pkl'))
    p.add_argument('--out-dir', type=str,
                   default=os.path.join(bench_dir, 'niche_umap_cache'))
    p.add_argument('--conditional', action='store_true',
                   help='condition generation on each niche\'s real anchor-cell embedding '
                        '(instead of unconditional generation); forces mini-batch OT off')
    return p.parse_args()


def main():
    args = _parse_args()
    t0 = time.time()

    import jax
    from niche_data import build_niches
    from models import build_model, generate_clouds
    from wassersteinflowmatching.riemannian_wasserstein.utils_PullbackFlow import PullbackFlowNet

    print(f"[viz] model={args.model} devices={jax.devices()}", flush=True)

    with open(args.geometry_path, 'rb') as f:
        geom_pkl = pickle.load(f)
    flow_net = PullbackFlowNet(dim=geom_pkl['dim'], hidden_dim=geom_pkl['hidden_dim'],
                              n_layers=geom_pkl['n_layers'])
    flow_params = geom_pkl['params']

    # Training keeps the exact same train/test split as the benchmark (so the trained model is
    # identical to the one the reported 1-NNA/MMD numbers come from); the "real" set for
    # visualization is ALL niches (train + held-out test), and we generate an equal-sized fake
    # set so the plot compares the full real distribution, not just the 1024-niche test slice.
    build_kwargs = dict(k=args.k, seed=args.seed, test_n=args.test_n,
                        center_label=args.center_label, center_field=args.center_field,
                        feature_path=args.feature_path)
    if args.conditional:
        train_clouds, test_clouds, train_anchors, test_anchors = build_niches(
            args.h5ad_path, return_anchors=True, **build_kwargs)
        all_anchors = np.concatenate([train_anchors, test_anchors], axis=0)
    else:
        train_clouds, test_clouds = build_niches(args.h5ad_path, **build_kwargs)
        train_anchors = all_anchors = None
    all_real = train_clouds + test_clouds
    n_real = len(all_real)
    print(f"[viz] {len(train_clouds)} train / {len(test_clouds)} test / {n_real} total niches "
          f"({time.time() - t0:.0f}s)", flush=True)

    os.makedirs(args.out_dir, exist_ok=True)
    real_path = os.path.join(args.out_dir, 'real_all_clouds.npy')
    if not os.path.exists(real_path):
        np.save(real_path, np.stack(all_real))
        print(f"[viz] wrote {real_path}", flush=True)
    if args.conditional:
        anchor_path = os.path.join(args.out_dir, 'anchor_all_clouds.npy')
        if not os.path.exists(anchor_path):
            np.save(anchor_path, all_anchors)
            print(f"[viz] wrote {anchor_path}", flush=True)

    model = build_model(args.model, train_clouds, flow_net, flow_params, conditioning=train_anchors)
    model.train(training_steps=args.n_train_steps, batch_size=args.batch_size,
                decay_steps=args.decay_steps, learning_rate=args.lr, key=jax.random.key(args.seed))
    print(f"[viz] trained {args.model} ({time.time() - t0:.0f}s)", flush=True)

    key = jax.random.key(1000 + args.seed)
    fake = generate_clouds(model, n_real, args.k, key, gen_batch=args.gen_batch,
                           timesteps=args.timesteps, conditioning=all_anchors)

    suffix = f"{args.model}_cond" if args.conditional else args.model
    out_path = os.path.join(args.out_dir, f"generated_all_{suffix}.npy")
    np.save(out_path, fake)
    print(f"[viz] wrote {out_path} shape={fake.shape} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == '__main__':
    main()
