"""Benchmark one (digit, model, seed) on the bunny-MNIST task.

Trains the model on the digit's train split, then generates ``--n-gens`` test-set-sized batches and
scores each against the digit's test split with 1-NNA and MMD under both Chamfer (CD) and Earth
Mover's (EMD) distances, using the mesh spectral metric as the base distance. Results (per generation
and aggregated over generations) are written to a JSON file. Aggregate across seeds with
``aggregate.py``.

Example:
    python benchmark.py --digit 0 --model rwefm --seed 0
"""

import argparse
import json
import os
import time

import numpy as np


def _parse_args():
    here = os.path.dirname(os.path.abspath(__file__))
    repo = os.path.abspath(os.path.join(here, '..', '..'))
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--digit', type=int, required=True, choices=range(10),
                   metavar='{0-9}')
    p.add_argument('--model', type=str, required=True,
                   choices=['rwefm', 'wfm', 'setrfm', 'setfm', 'rwsfm', 'wsfm'])
    p.add_argument('--seed', type=int, required=True)
    # training (defaults mirror the notebook)
    p.add_argument('--n-train-steps', type=int, default=100000)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--decay-steps', type=int, default=1000)
    p.add_argument('--lr', type=float, default=3e-4)
    # evaluation
    p.add_argument('--n-gens', type=int, default=5)
    # geometry / clouds
    p.add_argument('--k', type=int, default=100)
    p.add_argument('--n-interp', type=int, default=100)
    p.add_argument('--n-pts', type=int, default=150)
    p.add_argument('--n-grid', type=int, default=32)
    p.add_argument('--max-train', type=int, default=None,
                   help='cap number of train clouds (default: all)')
    p.add_argument('--max-test', type=int, default=None,
                   help='cap number of test clouds / generated clouds (default: all)')
    # metric compute
    p.add_argument('--sinkhorn-eps', type=float, default=0.05)
    p.add_argument('--sinkhorn-iters', type=int, default=100)
    p.add_argument('--embed-chunk', type=int, default=8)
    p.add_argument('--cd-chunk', type=int, default=32)
    p.add_argument('--emd-chunk', type=int, default=8)
    p.add_argument('--gen-batch', type=int, default=250)
    # io
    p.add_argument('--bunny-path', type=str,
                   default=os.path.join(repo, 'data', 'stanford-bunny.obj'))
    p.add_argument('--out-dir', type=str,
                   default=os.path.join(repo, 'results', 'bunny_mnist_benchmark'))
    return p.parse_args()


def _aggregate_gens(gens):
    """Mean/std over generations for every scalar metric."""
    def col(path):
        vals = []
        for g in gens:
            v = g
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
    import jax
    from geometry_data import build_geometry, chart_frame, build_digit_clouds
    from models import build_model, generate_clouds
    from metrics import (embed_clouds, block_matrix, one_nna, mmd, assemble_full)

    print(f"[bench] digit={args.digit} model={args.model} seed={args.seed} "
          f"devices={jax.devices()}", flush=True)

    # ---- geometry + data -----------------------------------------------------------------------
    geom, V, F = build_geometry(args.bunny_path, n_grid=args.n_grid, k=args.k,
                                normalize=False, n_interpolation_steps=args.n_interp)
    frame = chart_frame(geom, V, F)
    train_clouds = build_digit_clouds(geom, frame, args.digit, 'train',
                                      n_pts=args.n_pts, max_clouds=args.max_train)
    test_clouds = build_digit_clouds(geom, frame, args.digit, 'test',
                                     n_pts=args.n_pts, max_clouds=args.max_test)
    n_real = len(test_clouds)
    print(f"[bench] built {len(train_clouds)} train / {n_real} test clouds "
          f"({time.time() - t0:.0f}s)", flush=True)

    # ---- real reference: embeddings + real-real blocks (cached across generations) -------------
    E_real, m_real = embed_clouds(geom, test_clouds, chunk=args.embed_chunk)
    rr_cd = block_matrix(E_real, m_real, E_real, m_real, 'cd', args.cd_chunk)
    rr_emd = block_matrix(E_real, m_real, E_real, m_real, 'emd', args.emd_chunk,
                          args.sinkhorn_eps, args.sinkhorn_iters)
    print(f"[bench] real-real blocks done ({time.time() - t0:.0f}s)", flush=True)

    # ---- model + training ----------------------------------------------------------------------
    model, project = build_model(args.model, train_clouds, (V, F),
                                 k=args.k, n_interpolation_steps=args.n_interp)
    # 'random'/'matched' couplings never run Sinkhorn (num_sinkhorn_iters stays at the unused
    # config default), so report 0 for them rather than the stale default value.
    if getattr(model, 'monge_map', None) in ('random', 'matched'):
        sink_iters = 0
    else:
        sink_iters = int(getattr(model, 'num_sinkhorn_iters', -1))   # auto inner Sinkhorn iters
    mbot_iters = int(getattr(model, 'mini_batch_ot_num_iter', -1))   # auto mini-batch OT iters
    max_p = int(max(max((len(c) for c in train_clouds), default=0),
                    max((len(c) for c in test_clouds), default=0)))   # largest point-cloud size

    t_train0 = time.time()
    model.train(training_steps=args.n_train_steps, batch_size=args.batch_size,
                decay_steps=args.decay_steps, learning_rate=args.lr,
                key=jax.random.key(args.seed))
    train_wall = time.time() - t_train0
    # extrapolate training wall-time to 500k steps (one-time compile is <1% at 100k steps)
    per_step = train_wall / max(args.n_train_steps, 1)
    time_500k_sec = per_step * 500_000
    print(f"[bench] trained {args.n_train_steps} steps in {train_wall:.0f}s "
          f"({per_step * 1e3:.1f} ms/step -> 500k ~= {time_500k_sec / 3600:.2f} h) "
          f"| sink_iters={sink_iters} mbot_iters={mbot_iters} max|P|={max_p} "
          f"({time.time() - t0:.0f}s)", flush=True)

    # ---- generation + scoring per replicate ----------------------------------------------------
    gens = []
    for g in range(args.n_gens):
        key = jax.random.fold_in(jax.random.key(1000 + args.seed), g)
        fake = generate_clouds(model, n_real, args.n_pts, key, gen_batch=args.gen_batch)
        E_fake, m_fake = embed_clouds(geom, fake, chunk=args.embed_chunk)

        ff_cd = block_matrix(E_fake, m_fake, E_fake, m_fake, 'cd', args.cd_chunk)
        rf_cd = block_matrix(E_real, m_real, E_fake, m_fake, 'cd', args.cd_chunk)
        ff_emd = block_matrix(E_fake, m_fake, E_fake, m_fake, 'emd', args.emd_chunk,
                              args.sinkhorn_eps, args.sinkhorn_iters)
        rf_emd = block_matrix(E_real, m_real, E_fake, m_fake, 'emd', args.emd_chunk,
                              args.sinkhorn_eps, args.sinkhorn_iters)

        D_cd = assemble_full(rr_cd, rf_cd, ff_cd)
        D_emd = assemble_full(rr_emd, rf_emd, ff_emd)
        res = {
            '1nna_cd': one_nna(D_cd, n_real),
            '1nna_emd': one_nna(D_emd, n_real),
            'mmd_cd': mmd(rf_cd),
            'mmd_emd': mmd(rf_emd),
        }
        gens.append(res)
        print(f"[bench] gen {g + 1}/{args.n_gens}: "
              f"1NNA-CD(avg)={res['1nna_cd']['avg']:.3f} "
              f"1NNA-EMD(avg)={res['1nna_emd']['avg']:.3f} "
              f"MMD-CD={res['mmd_cd']:.4f} MMD-EMD={res['mmd_emd']:.4f} "
              f"({time.time() - t0:.0f}s)", flush=True)

    out = {
        'digit': args.digit, 'model': args.model, 'seed': args.seed,
        'n_real': n_real, 'n_train': len(train_clouds), 'n_pts': args.n_pts,
        'train_n': len(train_clouds), 'test_n': n_real, 'max_p': max_p,
        'n_train_steps': args.n_train_steps, 'project_generated': bool(project),
        'k': args.k, 'n_interp': args.n_interp,
        'sinkhorn_eps': args.sinkhorn_eps, 'sinkhorn_iters': args.sinkhorn_iters,
        'sinkhorn_iters_auto': sink_iters, 'mbot_iters_auto': mbot_iters,
        'train_wall_sec': train_wall, 'per_step_sec': per_step,
        'time_500k_sec': time_500k_sec,
        'gens': gens, 'agg': _aggregate_gens(gens),
        'runtime_sec': time.time() - t0,
    }
    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir,
                            f"digit{args.digit}_{args.model}_seed{args.seed}.json")
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"[bench] wrote {out_path} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == '__main__':
    main()
