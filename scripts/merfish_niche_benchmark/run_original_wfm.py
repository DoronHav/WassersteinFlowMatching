"""Sanity check: train the *original* (non-Riemannian-generalized) WFM implementation
(``wassersteinflowmatching.wasserstein.WassersteinFlowMatching``) on the Sst/k=16/16-PC
replication, with ``monge_map='sample'``, to test whether the riemannian_wasserstein module's
"wfm"/"wsfm" builds (used throughout the benchmark) diverge from the paper's actual original code
path -- most notably its default ``noise_type='chol_normal'`` (the Appendix D stochastic-covariance
source measure), vs. riemannian_wasserstein's default ``noise_type='ambient_gaussian'``.

Scores unconditional generation with the same CD/EMD/1-NNA/MMD machinery as benchmark.py, in plain
ambient (euclidean) space -- there is no learned geometry involved here.

Example:
    python run_original_wfm.py
"""

import argparse
import os
import time

import numpy as np


def _parse_args():
    here = os.path.dirname(os.path.abspath(__file__))
    repo = os.path.abspath(os.path.join(here, '..', '..'))
    bench_dir = os.path.join(repo, 'results', 'merfish_niche_benchmark')
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--monge-map', type=str, default='sample')
    p.add_argument('--k', type=int, default=16)
    p.add_argument('--center-field', type=str, default='subclass')
    p.add_argument('--center-label', type=str, default='Sst')
    p.add_argument('--feature-path', type=str,
                   default=os.path.join(bench_dir, 'X_pca16.npy'))
    p.add_argument('--h5ad-path', type=str,
                   default='/cv/data/braid/havivd/merfish_motor_cortex_atlas/st_data_processed.h5ad')
    p.add_argument('--n-train-steps', type=int, default=100000)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--decay-steps', type=int, default=1000)
    p.add_argument('--lr', type=float, default=2e-4)
    p.add_argument('--timesteps', type=int, default=100)
    p.add_argument('--n-gens', type=int, default=5)
    p.add_argument('--gen-batch', type=int, default=250)
    p.add_argument('--embed-chunk', type=int, default=8)
    p.add_argument('--cd-chunk', type=int, default=32)
    p.add_argument('--emd-chunk', type=int, default=8)
    p.add_argument('--sinkhorn-eps', type=float, default=0.05)
    p.add_argument('--sinkhorn-iters', type=int, default=100)
    p.add_argument('--num-sinkhorn-iters', type=str, default='converge',
                   help="Sinkhorn iterations (main map + mini-batch): 'converge' (None), '-1' (auto-find), or an int")
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--out-dir', type=str, default=bench_dir)
    return p.parse_args()


def _iters(v):
    s = str(v).strip().lower()
    if s in ('none', 'converge', ''):
        return None
    return int(v)


def main():
    args = _parse_args()
    t0 = time.time()

    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'src'))
    import jax
    from niche_data import build_niches
    from metrics import embed_clouds, block_matrix, one_nna, mmd, assemble_full
    from wassersteinflowmatching.wasserstein import WassersteinFlowMatching

    print(f"[orig] monge_map={args.monge_map} devices={jax.devices()}", flush=True)

    train_clouds, test_clouds = build_niches(
        args.h5ad_path, k=args.k, seed=args.seed, center_label=args.center_label,
        center_field=args.center_field, feature_path=args.feature_path)
    n_real = len(test_clouds)
    print(f"[orig] {len(train_clouds)} train / {n_real} test niches ({time.time() - t0:.0f}s)",
          flush=True)

    def embed_fn(p):
        return p

    E_real, m_real = embed_clouds(embed_fn, test_clouds, chunk=args.embed_chunk)
    rr_cd = block_matrix(E_real, m_real, E_real, m_real, 'cd', args.cd_chunk)
    rr_emd = block_matrix(E_real, m_real, E_real, m_real, 'emd', args.emd_chunk,
                          args.sinkhorn_eps, args.sinkhorn_iters)
    print(f"[orig] real-real blocks done ({time.time() - t0:.0f}s)", flush=True)

    model = WassersteinFlowMatching(point_clouds=train_clouds, monge_map=args.monge_map,
                                    num_sinkhorn_iters=_iters(args.num_sinkhorn_iters))

    t_train0 = time.time()
    model.train(training_steps=args.n_train_steps, batch_size=args.batch_size,
               decay_steps=args.decay_steps, learning_rate=args.lr,
               key=jax.random.key(args.seed))
    train_wall = time.time() - t_train0
    print(f"[orig] trained {args.n_train_steps} steps in {train_wall:.0f}s "
          f"({time.time() - t0:.0f}s)", flush=True)

    gens = []
    for g in range(args.n_gens):
        key = jax.random.fold_in(jax.random.key(1000 + args.seed), g)
        got, out = 0, []
        b = 0
        while got < n_real:
            n = int(min(args.gen_batch, n_real - got))
            subkey = jax.random.fold_in(key, b)
            samples, _ = model.generate_samples(size=args.k, num_samples=n,
                                                timesteps=args.timesteps, key=subkey)
            out.append(np.array(samples[-1]))
            got += n
            b += 1
        fake = np.concatenate(out, axis=0)[:n_real]

        E_fake, m_fake = embed_clouds(embed_fn, fake, chunk=args.embed_chunk)
        ff_cd = block_matrix(E_fake, m_fake, E_fake, m_fake, 'cd', args.cd_chunk)
        rf_cd = block_matrix(E_real, m_real, E_fake, m_fake, 'cd', args.cd_chunk)
        ff_emd = block_matrix(E_fake, m_fake, E_fake, m_fake, 'emd', args.emd_chunk,
                              args.sinkhorn_eps, args.sinkhorn_iters)
        rf_emd = block_matrix(E_real, m_real, E_fake, m_fake, 'emd', args.emd_chunk,
                              args.sinkhorn_eps, args.sinkhorn_iters)
        D_cd = assemble_full(rr_cd, rf_cd, ff_cd)
        D_emd = assemble_full(rr_emd, rf_emd, ff_emd)
        res = {
            '1nna_cd': one_nna(D_cd, n_real), '1nna_emd': one_nna(D_emd, n_real),
            'mmd_cd': mmd(rf_cd), 'mmd_emd': mmd(rf_emd),
        }
        gens.append(res)
        print(f"[orig] gen {g + 1}/{args.n_gens}: "
              f"1NNA-CD(avg)={res['1nna_cd']['avg']:.3f} 1NNA-EMD(avg)={res['1nna_emd']['avg']:.3f} "
              f"MMD-CD={res['mmd_cd']:.4f} ({time.time() - t0:.0f}s)", flush=True)

    cd_avg = np.mean([g['1nna_cd']['avg'] for g in gens])
    emd_avg = np.mean([g['1nna_emd']['avg'] for g in gens])
    print(f"[orig] FINAL monge_map={args.monge_map}: "
          f"1NNA-CD(avg)={cd_avg * 100:.2f}% 1NNA-EMD(avg)={emd_avg * 100:.2f}% "
          f"({time.time() - t0:.0f}s)", flush=True)


if __name__ == '__main__':
    main()
