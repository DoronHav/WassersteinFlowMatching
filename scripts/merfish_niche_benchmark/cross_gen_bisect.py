"""Weights-vs-harness bisection.

Trains the ORIGINAL wfm (``wassersteinflowmatching.wasserstein.WassersteinFlowMatching``) once,
then generates two ways from the SAME trained weights:

  A. wfm's OWN generation harness (RK2 midpoint)                 -> expected ~55% 1-NNA (paper).
  B. the RIEMANNIAN module's harness (Euler + reorder/max_size), after loading the wfm weights
     into a ``RiemannianWassersteinFlowMatching(geom='euclidean')`` model via a lossless
     parameter-tree key rename (``EncoderBlock_i`` -> ``AttentionEncoderBlock_i``).

Both harnesses are fed the *identical* per-batch init noise, so the only thing that differs is the
generation code path itself. Interpretation:

  * B ~= A (~55%)  -> the rwfm generation harness is correct; the whole discrepancy lives in
                      rwfm's TRAINING (same loss, different weights -- the init-scale story).
  * B ~= 99%       -> the rwfm generation harness is the culprit (noise init / reorder / dt),
                      independent of the weights.

Example:
    CUDA_VISIBLE_DEVICES=0 python cross_gen_bisect.py
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
    p.add_argument('--monge-map', type=str, default='sample')
    p.add_argument('--noise-type', type=str, default='chol_normal')
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
    p.add_argument('--n-gens', type=int, default=3)
    p.add_argument('--gen-batch', type=int, default=250)
    p.add_argument('--embed-chunk', type=int, default=8)
    p.add_argument('--cd-chunk', type=int, default=32)
    p.add_argument('--emd-chunk', type=int, default=8)
    p.add_argument('--sinkhorn-eps', type=float, default=0.05)
    p.add_argument('--sinkhorn-iters', type=int, default=100)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--save-weights', type=str,
                   default=os.path.join(bench_dir, 'cross_gen_wfm_params.pkl'))
    return p.parse_args()


def _rename_to_rwfm(wfm_params):
    """wfm param tree -> rwfm param tree: only the top-level block class name differs."""
    return {k.replace('EncoderBlock', 'AttentionEncoderBlock'): v
            for k, v in dict(wfm_params).items()}


def _sample_noise(noise_func, noise_config, n, k, d, key):
    noise = noise_func(size=[n, k, d], noise_config=noise_config, key=key)
    if isinstance(noise, tuple) and len(noise) == 2:
        noise = noise[0]
    return np.asarray(noise)


def main():
    import jax

    args = _parse_args()
    t0 = time.time()

    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'src'))
    from niche_data import build_niches
    from metrics import embed_clouds, block_matrix, one_nna, mmd, assemble_full
    from wassersteinflowmatching.wasserstein import WassersteinFlowMatching
    from wassersteinflowmatching.riemannian_wasserstein import RiemannianWassersteinFlowMatching

    print(f"[bisect] devices={jax.devices()}", flush=True)

    train_clouds, test_clouds = build_niches(
        args.h5ad_path, k=args.k, seed=args.seed, center_label=args.center_label,
        center_field=args.center_field, feature_path=args.feature_path)
    n_real = len(test_clouds)
    print(f"[bisect] {len(train_clouds)} train / {n_real} test niches ({time.time() - t0:.0f}s)",
          flush=True)

    def embed_fn(p):
        return p

    E_real, m_real = embed_clouds(embed_fn, test_clouds, chunk=args.embed_chunk)
    rr_cd = block_matrix(E_real, m_real, E_real, m_real, 'cd', args.cd_chunk)
    rr_emd = block_matrix(E_real, m_real, E_real, m_real, 'emd', args.emd_chunk,
                          args.sinkhorn_eps, args.sinkhorn_iters)
    print(f"[bisect] real-real blocks done ({time.time() - t0:.0f}s)", flush=True)

    # ---- Train the ORIGINAL wfm once ----
    model_wfm = WassersteinFlowMatching(
        point_clouds=train_clouds, monge_map=args.monge_map,
        num_sinkhorn_iters=None)
    t_train0 = time.time()
    model_wfm.train(training_steps=args.n_train_steps, batch_size=args.batch_size,
                    decay_steps=args.decay_steps, learning_rate=args.lr,
                    key=jax.random.key(args.seed))
    print(f"[bisect] wfm trained {args.n_train_steps} steps in {time.time() - t_train0:.0f}s "
          f"({time.time() - t0:.0f}s)", flush=True)

    with open(args.save_weights, 'wb') as f:
        pickle.dump(model_wfm.params, f)
    print(f"[bisect] saved wfm params -> {args.save_weights}", flush=True)

    # ---- Build the rwfm euclidean model and load the wfm weights via key rename ----
    model_rwfm = RiemannianWassersteinFlowMatching(
        point_clouds=train_clouds, geom='euclidean', monge_map=args.monge_map,
        mini_batch_ot_mode=True, num_sinkhorn_iters=None, mini_batch_ot_num_iter=None,
        noise_type=args.noise_type, cpu_projection=False)
    model_rwfm.params = _rename_to_rwfm(model_wfm.params)
    print("[bisect] loaded wfm weights into rwfm model (key-renamed)", flush=True)

    space_dim = int(np.asarray(train_clouds[0]).shape[-1])

    def score(gen_batch_fn, tag):
        gens = []
        for g in range(args.n_gens):
            key = jax.random.fold_in(jax.random.key(1000 + args.seed), g)
            got, b, out = 0, 0, []
            while got < n_real:
                n = int(min(args.gen_batch, n_real - got))
                subkey = jax.random.fold_in(key, b)
                # identical init noise for both harnesses (from the wfm source measure)
                nkey, gkey = jax.random.split(subkey)
                init_noise = _sample_noise(model_wfm.noise_func, model_wfm.noise_config,
                                           n, args.k, space_dim, nkey)
                out.append(gen_batch_fn(n, init_noise, gkey))
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
            res = {'1nna_cd': one_nna(D_cd, n_real), '1nna_emd': one_nna(D_emd, n_real),
                   'mmd_cd': mmd(rf_cd)}
            gens.append(res)
            print(f"[bisect:{tag}] gen {g + 1}/{args.n_gens}: "
                  f"1NNA-CD={res['1nna_cd']['avg']:.3f} 1NNA-EMD={res['1nna_emd']['avg']:.3f} "
                  f"MMD-CD={res['mmd_cd']:.4f} ({time.time() - t0:.0f}s)", flush=True)
        cd = np.mean([g['1nna_cd']['avg'] for g in gens]) * 100
        emd = np.mean([g['1nna_emd']['avg'] for g in gens]) * 100
        mmd_cd = np.mean([g['mmd_cd'] for g in gens])
        print(f"[bisect:{tag}] FINAL 1NNA-CD={cd:.2f}% 1NNA-EMD={emd:.2f}% MMD-CD={mmd_cd:.4f}",
              flush=True)
        return cd, emd

    def gen_wfm(n, init_noise, key):
        samples, _ = model_wfm.generate_samples(
            size=args.k, num_samples=n, timesteps=args.timesteps,
            init_noise=init_noise, key=key)
        return np.array(samples[-1])

    def gen_rwfm(n, init_noise, key):
        all_noises, _ = model_rwfm.generate_samples(
            size=args.k, num_samples=n, timesteps=args.timesteps,
            init_noise=init_noise, key=key)
        return np.array(all_noises[-1])

    print("\n=== A: wfm weights + wfm harness (control, expect ~55%) ===", flush=True)
    a_cd, a_emd = score(gen_wfm, 'A_wfm_harness')

    print("\n=== B: wfm weights + rwfm harness (bisection) ===", flush=True)
    b_cd, b_emd = score(gen_rwfm, 'B_rwfm_harness')

    print("\n================ VERDICT ================", flush=True)
    print(f"A (wfm harness):  1NNA-CD={a_cd:.2f}%  1NNA-EMD={a_emd:.2f}%", flush=True)
    print(f"B (rwfm harness): 1NNA-CD={b_cd:.2f}%  1NNA-EMD={b_emd:.2f}%", flush=True)
    if abs(b_cd - a_cd) < 8:
        print("-> harness INNOCENT: same weights score the same under both harnesses; "
              "the discrepancy is in rwfm TRAINING.", flush=True)
    else:
        print("-> harness CULPRIT: identical weights degrade under the rwfm harness; "
              "the bug is in rwfm GENERATION.", flush=True)


if __name__ == '__main__':
    main()
