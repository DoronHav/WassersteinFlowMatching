"""Fix-verification probe (no retraining).

Loads the wfm weights saved by ``cross_gen_bisect.py`` and scores the SAME weights (renamed
into the rwfm param tree) under three generation harnesses on identical per-batch init noise:

  RK2       : wfm's own harness (midpoint)                 -> control, expect ~54% 1-NNA.
  RWFM_SCAN : rwfm's shipped harness (jax.lax.scan)        -> reproduces the ~99.98% failure.
  RWFM_LOOP : rwfm get_flow driven by a Python for-loop    -> proposed fix, expect ~54%.

RWFM_LOOP calls the EXACT same ``model_rwfm.get_flow`` step function as the scan, differing
only in the driver (Python loop vs lax.scan). If it recovers ~54% while RWFM_SCAN stays ~99.98%,
the root cause is confirmed to be the scan's fused float32 numerics, and the fix is to replace
the scan with a per-step loop.

    CUDA_VISIBLE_DEVICES=0 python verify_loop_fix.py
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
    p.add_argument('--weights-path', type=str,
                   default=os.path.join(bench_dir, 'cross_gen_wfm_params.pkl'))
    p.add_argument('--timesteps', type=int, default=100)
    p.add_argument('--n-gens', type=int, default=3)
    p.add_argument('--gen-batch', type=int, default=250)
    p.add_argument('--embed-chunk', type=int, default=8)
    p.add_argument('--cd-chunk', type=int, default=32)
    p.add_argument('--emd-chunk', type=int, default=8)
    p.add_argument('--sinkhorn-eps', type=float, default=0.05)
    p.add_argument('--sinkhorn-iters', type=int, default=100)
    p.add_argument('--seed', type=int, default=0)
    return p.parse_args()


def _rename_to_rwfm(wfm_params):
    return {k.replace('EncoderBlock', 'AttentionEncoderBlock'): v
            for k, v in dict(wfm_params).items()}


def _sample_noise(noise_func, noise_config, n, k, d, key):
    noise = noise_func(size=[n, k, d], noise_config=noise_config, key=key)
    if isinstance(noise, tuple) and len(noise) == 2:
        noise = noise[0]
    return np.asarray(noise)


def main():
    import jax
    import jax.numpy as jnp

    args = _parse_args()
    t0 = time.time()

    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'src'))
    from niche_data import build_niches
    from metrics import embed_clouds, block_matrix, one_nna, mmd, assemble_full
    from wassersteinflowmatching.wasserstein import WassersteinFlowMatching
    from wassersteinflowmatching.riemannian_wasserstein import RiemannianWassersteinFlowMatching

    print(f"[verify] devices={jax.devices()}", flush=True)

    train_clouds, test_clouds = build_niches(
        args.h5ad_path, k=args.k, seed=args.seed, center_label=args.center_label,
        center_field=args.center_field, feature_path=args.feature_path)
    n_real = len(test_clouds)
    print(f"[verify] {len(train_clouds)} train / {n_real} test niches ({time.time() - t0:.0f}s)",
          flush=True)

    def embed_fn(p):
        return p

    E_real, m_real = embed_clouds(embed_fn, test_clouds, chunk=args.embed_chunk)
    rr_cd = block_matrix(E_real, m_real, E_real, m_real, 'cd', args.cd_chunk)
    rr_emd = block_matrix(E_real, m_real, E_real, m_real, 'emd', args.emd_chunk,
                          args.sinkhorn_eps, args.sinkhorn_iters)
    print(f"[verify] real-real blocks done ({time.time() - t0:.0f}s)", flush=True)

    with open(args.weights_path, 'rb') as f:
        wfm_params = pickle.load(f)
    print(f"[verify] loaded wfm params <- {args.weights_path}", flush=True)

    model_wfm = WassersteinFlowMatching(
        point_clouds=train_clouds, monge_map=args.monge_map, num_sinkhorn_iters=None)
    model_wfm.params = wfm_params

    model_rwfm = RiemannianWassersteinFlowMatching(
        point_clouds=train_clouds, geom='euclidean', monge_map=args.monge_map,
        mini_batch_ot_mode=True, num_sinkhorn_iters=None, mini_batch_ot_num_iter=None,
        noise_type=args.noise_type, cpu_projection=False)
    model_rwfm.params = _rename_to_rwfm(wfm_params)

    space_dim = int(np.asarray(train_clouds[0]).shape[-1])
    dt = 1.0 / args.timesteps
    ts = jnp.linspace(1.0, dt, args.timesteps)

    def score(gen_batch_fn, tag):
        gens = []
        for g in range(args.n_gens):
            key = jax.random.fold_in(jax.random.key(1000 + args.seed), g)
            got, b, out = 0, 0, []
            while got < n_real:
                n = int(min(args.gen_batch, n_real - got))
                subkey = jax.random.fold_in(key, b)
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
            print(f"[verify:{tag}] gen {g + 1}/{args.n_gens}: "
                  f"1NNA-CD={res['1nna_cd']['avg']:.3f} 1NNA-EMD={res['1nna_emd']['avg']:.3f} "
                  f"MMD-CD={res['mmd_cd']:.4f} ({time.time() - t0:.0f}s)", flush=True)
        cd = np.mean([g['1nna_cd']['avg'] for g in gens]) * 100
        emd = np.mean([g['1nna_emd']['avg'] for g in gens]) * 100
        mmd_cd = np.mean([g['mmd_cd'] for g in gens])
        print(f"[verify:{tag}] FINAL 1NNA-CD={cd:.2f}% 1NNA-EMD={emd:.2f}% MMD-CD={mmd_cd:.4f}",
              flush=True)
        return cd, emd

    def gen_wfm_rk2(n, init_noise, key):
        samples, _ = model_wfm.generate_samples(
            size=args.k, num_samples=n, timesteps=args.timesteps, init_noise=init_noise, key=key)
        return np.array(samples[-1])

    def gen_rwfm_scan(n, init_noise, key):
        all_noises, _ = model_rwfm.generate_samples(
            size=args.k, num_samples=n, timesteps=args.timesteps, init_noise=init_noise, key=key)
        return np.array(all_noises[-1])

    def gen_rwfm_loop(n, init_noise, key):
        # Same step function as generate_samples' scan body, driven by a Python loop.
        xt = jnp.asarray(init_noise)
        weights = jnp.ones((n, args.k))
        for i in range(args.timesteps):
            xt = model_rwfm.get_flow(model_rwfm.params, xt, weights, ts[i], dt, None)
        return np.array(xt)

    print("\n=== RK2: wfm weights + wfm harness (control) ===", flush=True)
    rk2_cd, rk2_emd = score(gen_wfm_rk2, 'RK2')
    print("\n=== RWFM_SCAN: rwfm harness (jax.lax.scan) ===", flush=True)
    sc_cd, sc_emd = score(gen_rwfm_scan, 'RWFM_SCAN')
    print("\n=== RWFM_LOOP: rwfm get_flow via Python loop (proposed fix) ===", flush=True)
    lp_cd, lp_emd = score(gen_rwfm_loop, 'RWFM_LOOP')

    print("\n================ VERDICT ================", flush=True)
    print(f"RK2       : 1NNA-CD={rk2_cd:.2f}%  1NNA-EMD={rk2_emd:.2f}%", flush=True)
    print(f"RWFM_SCAN : 1NNA-CD={sc_cd:.2f}%  1NNA-EMD={sc_emd:.2f}%", flush=True)
    print(f"RWFM_LOOP : 1NNA-CD={lp_cd:.2f}%  1NNA-EMD={lp_emd:.2f}%", flush=True)
    if abs(lp_cd - rk2_cd) < 8 and (sc_cd - lp_cd) > 20:
        print("-> FIX CONFIRMED: Python loop recovers wfm-quality; the scan is the bug.", flush=True)
    elif abs(lp_cd - sc_cd) < 8:
        print("-> FIX FAILED: loop matches scan; the bug is not the scan driver.", flush=True)
    else:
        print("-> PARTIAL: loop helps but does not fully recover.", flush=True)


if __name__ == '__main__':
    main()
