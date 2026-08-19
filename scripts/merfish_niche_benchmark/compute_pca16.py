"""One-time: compute a 16-PC PCA embedding of the atlas for the pullback-flow-on-16-PCs run.

``.X`` (275886 x 254 genes) is already log1p-transformed and median-library-size normalized --
confirmed directly: ``expm1(X)`` row-sums are constant (~277.42) across cells, which is exactly
what median-library-size normalization + log1p produces. So this script PCAs ``.X`` as-is (mean-
centered, standard sklearn convention; no additional per-gene scaling), taking the top 16 components.
This is a *different* feature space from the atlas's precomputed ``obsm['X_pca']`` (64 PCs from
whatever the atlas builder used), used only for the PCA-dimensionality ablation.

Run once; caches the result so every downstream job (geometry pretraining + all 8 benchmark
models x 2 datasets) reads the identical embedding via ``--feature-path``.

Example:
    python compute_pca16.py
"""

import argparse
import os
import time

import numpy as np


def _parse_args():
    here = os.path.dirname(os.path.abspath(__file__))
    repo = os.path.abspath(os.path.join(here, '..', '..'))
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--h5ad-path', type=str,
                   default='/cv/data/braid/havivd/merfish_motor_cortex_atlas/st_data_processed.h5ad')
    p.add_argument('--n-components', type=int, default=16)
    p.add_argument('--out-path', type=str,
                   default=os.path.join(repo, 'results', 'merfish_niche_benchmark',
                                        'X_pca16.npy'))
    return p.parse_args()


def main():
    args = _parse_args()
    t0 = time.time()

    import h5py  # type: ignore
    from sklearn.decomposition import PCA  # type: ignore

    with h5py.File(args.h5ad_path, 'r') as f:
        X = f['X'][:].astype(np.float64)
    print(f"[pca16] loaded X {X.shape} ({time.time() - t0:.0f}s)", flush=True)

    # Sanity-check the "already log1p + median-library-size normalized" claim before trusting it.
    counts = np.expm1(X[:2000])
    row_sums = counts.sum(axis=1)
    print(f"[pca16] expm1(X) row-sum check (first 2000 cells): "
          f"mean={row_sums.mean():.4f} std={row_sums.std():.6f} "
          f"(std/mean={row_sums.std() / row_sums.mean():.2e} -- should be ~0)", flush=True)
    if row_sums.std() / row_sums.mean() > 1e-3:
        print("[pca16] WARNING: row sums are not tightly constant -- .X may not be "
              "median-library-size normalized as expected. Proceeding anyway.", flush=True)

    pca = PCA(n_components=args.n_components, svd_solver='auto', random_state=0)
    X_pca16 = pca.fit_transform(X).astype(np.float32)
    print(f"[pca16] fit PCA({args.n_components}) explained_variance_ratio="
          f"{np.array2string(pca.explained_variance_ratio_, precision=4)} "
          f"(total={pca.explained_variance_ratio_.sum():.4f}) ({time.time() - t0:.0f}s)",
          flush=True)

    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)
    np.save(args.out_path, X_pca16)
    print(f"[pca16] wrote {args.out_path} shape={X_pca16.shape} ({time.time() - t0:.0f}s)",
          flush=True)


if __name__ == '__main__':
    main()
