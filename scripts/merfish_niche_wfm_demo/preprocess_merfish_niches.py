"""Prepare slice-local MERFISH niches for the WFM-sample tutorial.

The atlas stores 254-gene expression in ``.X``.  For the current atlas this is
already median-library-size normalized and log1p transformed: ``expm1(X)`` has
the same row sum for every cell.  This script verifies that condition; if the
input instead looks like non-negative count data, it applies the requested
normalization and log1p before fitting PCA.

For every atlas cell, the script constructs one niche from the 128 nearest
*other* cells in physical ``obsm['spatial']`` coordinates.  Neighbor searches
are performed independently for every ``obs['slice_id']`` value, so no niche
can cross a tissue slice.

Artifacts are written beside the source h5ad by default:

* ``merfish_log1p_pca16.npy``: PCA coordinates for every atlas cell.
* ``merfish_log1p_pca16.pkl``: fitted sklearn PCA object.
* ``merfish_niche_member_indices_k128.npy``: global atlas row indices.
* ``merfish_niche_point_clouds_pca16_k128.npy``: materialized PCA point clouds.
* ``merfish_niche_metadata_k128.npz``: centers, slices, genes, and HVG mask.

The large arrays are open-format ``.npy`` files and can be memory-mapped.
"""

from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path

import h5py
import numpy as np
from numpy.lib.format import open_memmap
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors


DEFAULT_ATLAS = Path(
    "/cv/data/braid/havivd/merfish_motor_cortex_atlas/st_data_processed.h5ad"
)


def _decode(values: np.ndarray) -> np.ndarray:
    return np.asarray([
        value.decode("utf-8") if isinstance(value, bytes) else str(value)
        for value in values
    ])


def _read_categorical(group: h5py.Group) -> tuple[np.ndarray, np.ndarray]:
    return group["codes"][:].astype(np.int32), _decode(group["categories"][:])


def _prepare_expression(X: np.ndarray) -> tuple[np.ndarray, dict]:
    """Return median-library-size normalized log1p expression and provenance."""
    probe = np.asarray(X[: min(5000, X.shape[0])], dtype=np.float64)
    if np.any(probe < 0):
        raise ValueError(".X contains negative values; cannot interpret it as counts or log1p data")

    expm1_sums = np.expm1(probe).sum(axis=1)
    positive = expm1_sums > 0
    expm1_cv = float(expm1_sums[positive].std() / expm1_sums[positive].mean())

    # A nearly constant expm1 row sum is the fingerprint of library-size
    # normalization followed by log1p. Avoid applying the transform twice.
    if positive.all() and expm1_cv < 1e-3:
        return np.asarray(X, dtype=np.float64), {
            "expression_transform": "already_median_library_normalized_log1p",
            "target_library_size": float(np.median(expm1_sums)),
            "probe_expm1_library_cv": expm1_cv,
        }

    if not np.all(np.isclose(probe, np.rint(probe), atol=1e-6)):
        raise ValueError(
            ".X is neither recognizably count-valued nor already library-normalized log1p data"
        )

    counts = np.asarray(X, dtype=np.float64)
    library_size = counts.sum(axis=1)
    target = float(np.median(library_size[library_size > 0]))
    normalized = counts * (target / np.maximum(library_size, 1.0))[:, None]
    return np.log1p(normalized), {
        "expression_transform": "median_library_normalization_then_log1p",
        "target_library_size": target,
        "probe_expm1_library_cv": expm1_cv,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--atlas", type=Path, default=DEFAULT_ATLAS)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--n-components", type=int, default=16)
    parser.add_argument("--n-neighbors", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--write-chunk", type=int, default=2048)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir or args.atlas.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    with h5py.File(args.atlas, "r") as handle:
        X_input = handle["X"][:]
        spatial = handle["obsm/spatial"][:].astype(np.float32)
        slice_codes, slice_categories = _read_categorical(handle["obs/slice_id"])
        gene_names = _decode(handle["var/_index"][:])
        hvg_mask = handle["var/highly_variable"][:].astype(bool)

    X_log1p, transform_info = _prepare_expression(X_input)
    del X_input
    n_cells, n_genes = X_log1p.shape
    print(
        f"[prep] expression {X_log1p.shape}; {transform_info['expression_transform']}; "
        f"target library={transform_info['target_library_size']:.4f}",
        flush=True,
    )

    pca = PCA(n_components=args.n_components, svd_solver="auto", random_state=args.seed)
    X_pca = pca.fit_transform(X_log1p).astype(np.float32)
    del X_log1p
    pca_path = out_dir / f"merfish_log1p_pca{args.n_components}.pkl"
    coords_path = out_dir / f"merfish_log1p_pca{args.n_components}.npy"
    with pca_path.open("wb") as stream:
        pickle.dump(pca, stream, protocol=pickle.HIGHEST_PROTOCOL)
    np.save(coords_path, X_pca)
    print(
        f"[prep] PCA{args.n_components} explains "
        f"{pca.explained_variance_ratio_.sum():.3%}; saved coordinates and PCA "
        f"({time.time() - t0:.0f}s)",
        flush=True,
    )

    k = args.n_neighbors
    if np.min(np.bincount(slice_codes)) <= k:
        raise ValueError(f"At least one slice has fewer than k+1={k + 1} cells")

    member_path = out_dir / f"merfish_niche_member_indices_k{k}.npy"
    members = open_memmap(member_path, mode="w+", dtype=np.int32, shape=(n_cells, k))
    for slice_code, slice_name in enumerate(slice_categories):
        global_idx = np.flatnonzero(slice_codes == slice_code)
        nn = NearestNeighbors(n_neighbors=k + 1, algorithm="auto", n_jobs=-1)
        local_neighbors = nn.fit(spatial[global_idx]).kneighbors(return_distance=False)
        # kneighbors(X=None) excludes each training point itself in sklearn. Assert
        # this explicitly because center exclusion is part of the data definition.
        global_neighbors = global_idx[local_neighbors[:, :k]]
        if np.any(global_neighbors == global_idx[:, None]):
            raise RuntimeError(f"Center leakage detected while building slice {slice_name}")
        members[global_idx] = global_neighbors
        print(
            f"[prep] slice {slice_code + 1:02d}/{len(slice_categories)} "
            f"{slice_name}: {len(global_idx):,} niches",
            flush=True,
        )
    members.flush()

    cloud_path = out_dir / (
        f"merfish_niche_point_clouds_pca{args.n_components}_k{k}.npy"
    )
    clouds = open_memmap(
        cloud_path,
        mode="w+",
        dtype=np.float32,
        shape=(n_cells, k, args.n_components),
    )
    for start in range(0, n_cells, args.write_chunk):
        stop = min(start + args.write_chunk, n_cells)
        clouds[start:stop] = X_pca[members[start:stop]]
    clouds.flush()

    metadata_path = out_dir / f"merfish_niche_metadata_k{k}.npz"
    np.savez(
        metadata_path,
        center_indices=np.arange(n_cells, dtype=np.int32),
        center_slice_codes=slice_codes,
        slice_categories=slice_categories,
        gene_names=gene_names,
        hvg_mask=hvg_mask,
    )
    manifest = {
        "atlas": str(args.atlas),
        "n_cells": int(n_cells),
        "n_genes": int(n_genes),
        "n_components": int(args.n_components),
        "n_neighbors": int(k),
        "center_excluded": True,
        "slice_key": "slice_id",
        "all_neighbors_same_slice": True,
        "pca_explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "n_hvg": int(hvg_mask.sum()),
        **transform_info,
        "files": {
            "pca": str(pca_path),
            "cell_pca": str(coords_path),
            "member_indices": str(member_path),
            "niche_point_clouds": str(cloud_path),
            "metadata": str(metadata_path),
        },
    }
    manifest_path = out_dir / f"merfish_niche_manifest_pca{args.n_components}_k{k}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    # Final invariants, checked without materializing the 2.3 GB cloud array.
    probe_idx = np.linspace(0, n_cells - 1, 1000, dtype=int)
    if not np.all(slice_codes[members[probe_idx]] == slice_codes[probe_idx, None]):
        raise RuntimeError("Cross-slice neighbor found in final artifact")
    if not np.allclose(clouds[probe_idx], X_pca[members[probe_idx]]):
        raise RuntimeError("Materialized point clouds do not match saved member indices")
    print(
        f"[prep] wrote {n_cells:,} x {k} x {args.n_components} niches to {out_dir} "
        f"({time.time() - t0:.0f}s total)",
        flush=True,
    )


if __name__ == "__main__":
    main()
