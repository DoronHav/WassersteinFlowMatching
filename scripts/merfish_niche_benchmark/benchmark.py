"""Benchmark one (model, seed) on the MERFISH niche-generation task.

Trains the model on the GABAergic-center niche train split, then generates ``--n-gens`` test-set-
sized batches and scores each against the held-out test split with 1-NNA and MMD under both
Chamfer (CD) and Earth Mover's (EMD) distances -- computed under **both** the learned pullback-flow
metric and plain Euclidean X_pca (the experiment spec's robustness check against evaluation
circularity, Sec. 7-bis). Results (per generation and aggregated) are written to a JSON file.
Aggregate across models/seeds with ``aggregate.py``.

Requires a pretrained geometry from ``train_geometry.py`` (default path:
results/merfish_niche_benchmark/pullback_flow_geometry.pkl).

Example:
    python benchmark.py --model rwefm --seed 0
"""

import argparse
import json
import os
import pickle
import time

import numpy as np


def _parse_args():
    here = os.path.dirname(os.path.abspath(__file__))
    repo = os.path.abspath(os.path.join(here, '..', '..'))
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--model', type=str, required=True,
                   choices=['rwefm', 'wfm', 'setrfm', 'setfm', 'rwsfm', 'wsfm'])
    p.add_argument('--seed', type=int, required=True)
    # training
    p.add_argument('--n-train-steps', type=int, default=50000)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--decay-steps', type=int, default=1000)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--time-budget-sec', type=float, default=None,
                   help='if set, calibrate an affordable n_train_steps from a short warmup '
                        'instead of using --n-train-steps directly')
    # evaluation
    p.add_argument('--n-gens', type=int, default=5)
    p.add_argument('--timesteps', type=int, default=100,
                   help='Euler steps integrating the learned flow at generation time')
    p.add_argument('--timesteps-sweep', type=str, default=None,
                   help='comma-separated list (e.g. "8,16,32,64,128") -- if set, trains once '
                        'then generates+scores at each value instead of just --timesteps, '
                        'writing {model}_seed{seed}_timesteps_sweep.json')
    # niches
    p.add_argument('--k', type=int, default=128, help='points per niche (spatial neighbors)')
    p.add_argument('--data-seed', type=int, default=0,
                   help='fixed niche train/test split seed, independent of --seed')
    p.add_argument('--test-n', type=int, default=1024,
                   help='fixed number of held-out test niches (rest go to train)')
    p.add_argument('--center-label', type=str, default='GABAergic',
                   help="category (within --center-field) for niche centers "
                        "(e.g. 'GABAergic'/'Glutamatergic' for cell_label, or 'Sst' for subclass)")
    p.add_argument('--center-field', type=str, default='cell_label',
                   help="obs field --center-label is a category of (e.g. 'cell_label' or 'subclass')")
    p.add_argument('--feature-path', type=str, default=None,
                   help='override cell-feature space (must match the geometry pickle -- e.g. '
                        "compute_pca16.py's output when --geometry-path is a PCA-16 geometry)")
    p.add_argument('--max-train', type=int, default=None)
    p.add_argument('--max-test', type=int, default=None)
    # metric compute
    p.add_argument('--sinkhorn-eps', type=float, default=0.05)
    p.add_argument('--sinkhorn-iters', type=int, default=100)
    p.add_argument('--embed-chunk', type=int, default=8)
    p.add_argument('--cd-chunk', type=int, default=32)
    p.add_argument('--emd-chunk', type=int, default=8)
    p.add_argument('--gen-batch', type=int, default=250)
    p.add_argument('--n-interpolation-steps', type=int, default=None,
                   help='neuralfim geometry only -- overrides the value saved in the geometry '
                        'pickle (RWEFM marching-scan substeps)')
    # io
    p.add_argument('--h5ad-path', type=str,
                   default='/cv/data/braid/havivd/merfish_motor_cortex_atlas/st_data_processed.h5ad')
    p.add_argument('--geometry-path', type=str,
                   default=os.path.join(repo, 'results', 'merfish_niche_benchmark',
                                        'pullback_flow_geometry.pkl'))
    p.add_argument('--out-dir', type=str,
                   default=os.path.join(repo, 'results', 'merfish_niche_benchmark'))
    return p.parse_args()


def _aggregate_gens(gens, metric_key):
    """Mean/std over generations for every scalar metric, for one metric block ('learned'/'euclidean')."""
    def col(path):
        vals = []
        for g in gens:
            v = g[metric_key]
            for kk in path:
                v = v[kk]
            vals.append(v)
        return float(np.mean(vals)), float(np.std(vals))

    keys = [
        ('1nna_cd', 'real'), ('1nna_cd', 'fake'), ('1nna_cd', 'avg'),
        ('1nna_emd', 'real'), ('1nna_emd', 'fake'), ('1nna_emd', 'avg'),
        ('mmd_cd',), ('mmd_emd',),
    ]
    agg = {}
    for path in keys:
        mean, std = col(path)
        agg['_'.join(path)] = {'mean': mean, 'std': std}
    return agg


def main():
    args = _parse_args()
    t0 = time.time()

    # heavy imports after argparse so --help is instant
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'src'))
    import jax
    import jax.numpy as jnp
    from niche_data import build_niches
    from models import build_model, generate_clouds
    from metrics import embed_clouds, block_matrix, one_nna, mmd, assemble_full

    print(f"[bench] model={args.model} seed={args.seed} devices={jax.devices()}", flush=True)

    # ---- geometry (pretrained) -----------------------------------------------------------------
    with open(args.geometry_path, 'rb') as f:
        geom_pkl = pickle.load(f)
    backend = geom_pkl.get('backend', 'pullback')

    if backend == 'pullback':
        from wassersteinflowmatching.riemannian_wasserstein.utils_PullbackFlow import PullbackFlowNet
        geom_net = PullbackFlowNet(dim=geom_pkl['dim'], hidden_dim=geom_pkl['hidden_dim'],
                                   n_layers=geom_pkl['n_layers'])
        geom_params = geom_pkl['params']
        n_interpolation_steps = None  # unused for this backend

        def learned_embed(p):
            z, _ = geom_net.apply({'params': geom_params}, p, method=PullbackFlowNet.forward)
            return z
    else:
        from wassersteinflowmatching.riemannian_wasserstein.utils_NeuralFIM import NeuralFIMEncoder
        geom_net = NeuralFIMEncoder(hidden_dim=geom_pkl['hidden_dim'],
                                    n_landmarks=geom_pkl['n_landmarks'])
        geom_params = geom_pkl['params']
        n_interpolation_steps = (args.n_interpolation_steps if args.n_interpolation_steps
                                 is not None else geom_pkl.get('n_interpolation_steps', 32))

        def learned_embed(p):
            probs = jax.nn.softmax(geom_net.apply({'params': geom_params}, p))
            return jnp.sqrt(probs + 1e-8)

    def euclidean_embed(p):
        return p

    print(f"[bench] loaded {backend} geometry from {args.geometry_path} "
          f"(purity learned={geom_pkl.get('purity_learned'):.3f} "
          f"euclidean={geom_pkl.get('purity_euclidean'):.3f}) ({time.time() - t0:.0f}s)", flush=True)

    # ---- data -----------------------------------------------------------------------------------
    train_clouds, test_clouds = build_niches(
        args.h5ad_path, k=args.k, seed=args.data_seed, test_n=args.test_n,
        max_train=args.max_train, max_test=args.max_test, center_label=args.center_label,
        center_field=args.center_field, feature_path=args.feature_path)
    n_real = len(test_clouds)
    print(f"[bench] built {len(train_clouds)} train / {n_real} test niches "
          f"({time.time() - t0:.0f}s)", flush=True)

    # ---- real reference: embeddings + real-real blocks, under both metrics ----------------------
    metric_embeds = {'learned': learned_embed, 'euclidean': euclidean_embed}
    real_blocks = {}
    for mkey, embed_fn in metric_embeds.items():
        E_real, m_real = embed_clouds(embed_fn, test_clouds, chunk=args.embed_chunk)
        rr_cd = block_matrix(E_real, m_real, E_real, m_real, 'cd', args.cd_chunk)
        rr_emd = block_matrix(E_real, m_real, E_real, m_real, 'emd', args.emd_chunk,
                              args.sinkhorn_eps, args.sinkhorn_iters)
        real_blocks[mkey] = dict(E_real=E_real, m_real=m_real, rr_cd=rr_cd, rr_emd=rr_emd)
    print(f"[bench] real-real blocks done, both metrics ({time.time() - t0:.0f}s)", flush=True)

    # ---- model + training -------------------------------------------------------------------
    model = build_model(args.model, train_clouds, geom_net, geom_params, backend=backend,
                        n_interpolation_steps=n_interpolation_steps)
    if getattr(model, 'monge_map', None) in ('random', 'matched'):
        sink_iters = 0
    else:
        sink_iters = int(getattr(model, 'num_sinkhorn_iters', -1))
    mbot_iters = int(getattr(model, 'mini_batch_ot_num_iter', -1))
    max_p = int(max(max((len(c) for c in train_clouds), default=0),
                    max((len(c) for c in test_clouds), default=0)))

    n_train_steps = args.n_train_steps
    if args.time_budget_sec is not None:
        warmup = min(200, n_train_steps)
        t_warm0 = time.time()
        model.train(training_steps=warmup, batch_size=args.batch_size,
                    decay_steps=args.decay_steps, learning_rate=args.lr,
                    key=jax.random.key(args.seed))
        warm_wall = time.time() - t_warm0
        per_step = warm_wall / max(warmup, 1)
        remaining_budget = max(args.time_budget_sec - (time.time() - t0), 0.0)
        affordable = int(remaining_budget / max(per_step, 1e-6))
        n_train_steps = warmup + max(affordable, 0)
        n_train_steps = min(n_train_steps, args.n_train_steps) if args.n_train_steps else n_train_steps
        print(f"[bench] warmup {warmup} steps in {warm_wall:.0f}s ({per_step * 1e3:.1f} ms/step) "
              f"-> training to {n_train_steps} steps total for the {args.time_budget_sec:.0f}s "
              f"budget ({time.time() - t0:.0f}s)", flush=True)

    t_train0 = time.time()
    model.train(training_steps=n_train_steps, batch_size=args.batch_size,
                decay_steps=args.decay_steps, learning_rate=args.lr,
                key=jax.random.key(args.seed))
    train_wall = time.time() - t_train0
    per_step = train_wall / max(n_train_steps, 1)
    print(f"[bench] trained {n_train_steps} steps in {train_wall:.0f}s ({per_step * 1e3:.1f} ms/step) "
          f"| sink_iters={sink_iters} mbot_iters={mbot_iters} max|P|={max_p} "
          f"({time.time() - t0:.0f}s)", flush=True)

    # ---- generation + scoring per replicate, both metrics ----------------------------------
    def run_gens(timesteps, label):
        gens = []
        for g in range(args.n_gens):
            key = jax.random.fold_in(jax.random.key(1000 + args.seed), g)
            fake = generate_clouds(model, n_real, args.k, key, gen_batch=args.gen_batch,
                                   timesteps=timesteps)

            res = {}
            for mkey, embed_fn in metric_embeds.items():
                rb = real_blocks[mkey]
                E_fake, m_fake = embed_clouds(embed_fn, fake, chunk=args.embed_chunk)
                ff_cd = block_matrix(E_fake, m_fake, E_fake, m_fake, 'cd', args.cd_chunk)
                rf_cd = block_matrix(rb['E_real'], rb['m_real'], E_fake, m_fake, 'cd',
                                     args.cd_chunk)
                ff_emd = block_matrix(E_fake, m_fake, E_fake, m_fake, 'emd', args.emd_chunk,
                                      args.sinkhorn_eps, args.sinkhorn_iters)
                rf_emd = block_matrix(rb['E_real'], rb['m_real'], E_fake, m_fake, 'emd',
                                      args.emd_chunk, args.sinkhorn_eps, args.sinkhorn_iters)
                D_cd = assemble_full(rb['rr_cd'], rf_cd, ff_cd)
                D_emd = assemble_full(rb['rr_emd'], rf_emd, ff_emd)
                res[mkey] = {
                    '1nna_cd': one_nna(D_cd, n_real),
                    '1nna_emd': one_nna(D_emd, n_real),
                    'mmd_cd': mmd(rf_cd),
                    'mmd_emd': mmd(rf_emd),
                }
            gens.append(res)
            print(f"[bench] {label} gen {g + 1}/{args.n_gens}: "
                  f"learned 1NNA-CD(avg)={res['learned']['1nna_cd']['avg']:.3f} "
                  f"MMD-CD={res['learned']['mmd_cd']:.4f} | "
                  f"euclidean 1NNA-CD(avg)={res['euclidean']['1nna_cd']['avg']:.3f} "
                  f"MMD-CD={res['euclidean']['mmd_cd']:.4f} "
                  f"({time.time() - t0:.0f}s)", flush=True)
        return gens

    base_out = {
        'model': args.model, 'seed': args.seed, 'backend': backend,
        'geometry_path': args.geometry_path, 'n_interpolation_steps': n_interpolation_steps,
        'n_real': n_real, 'n_train': len(train_clouds), 'k': args.k, 'max_p': max_p,
        'n_train_steps': n_train_steps, 'data_seed': args.data_seed,
        'sinkhorn_eps': args.sinkhorn_eps, 'sinkhorn_iters': args.sinkhorn_iters,
        'sinkhorn_iters_auto': sink_iters, 'mbot_iters_auto': mbot_iters,
        'train_wall_sec': train_wall, 'per_step_sec': per_step,
    }

    os.makedirs(args.out_dir, exist_ok=True)
    if args.timesteps_sweep:
        sweep_values = [int(v) for v in args.timesteps_sweep.split(',')]
        sweeps = {}
        for ts in sweep_values:
            gens = run_gens(ts, label=f"timesteps={ts}")
            sweeps[str(ts)] = {
                'timesteps': ts, 'gens': gens,
                'agg_learned': _aggregate_gens(gens, 'learned'),
                'agg_euclidean': _aggregate_gens(gens, 'euclidean'),
            }
        out = dict(base_out, timesteps_sweep=sweep_values, sweeps=sweeps,
                  runtime_sec=time.time() - t0)
        out_path = os.path.join(args.out_dir, f"{args.model}_seed{args.seed}_timesteps_sweep.json")
    else:
        gens = run_gens(args.timesteps, label="")
        out = dict(base_out, timesteps=args.timesteps, gens=gens,
                  agg_learned=_aggregate_gens(gens, 'learned'),
                  agg_euclidean=_aggregate_gens(gens, 'euclidean'),
                  runtime_sec=time.time() - t0)
        out_path = os.path.join(args.out_dir, f"{args.model}_seed{args.seed}.json")

    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"[bench] wrote {out_path} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == '__main__':
    main()
