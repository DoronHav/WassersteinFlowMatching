"""MERFISH motor-cortex niche extraction (mirrors bunny_mnist_benchmark's geometry_data.py role).

Builds spatial-neighborhood point clouds in gene-expression (``X_pca``) space around inhibitory
(GABAergic) neuron centers, per the experiment spec
(``claude_md/learned_geometry_niche_experiment.md``, Sec. 3-5):

- Each niche = the ``X_pca`` rows of a GABAergic center cell's 128 nearest *spatial* neighbors,
  restricted to the center's own ``batch`` (tissue section) so niches never cross sections. The
  center cell itself is excluded from its own niche.
- Geometry is learned separately (``train_geometry.py``) on **all** cells, not just niche members.

Reads the atlas h5ad directly via h5py rather than ``anndata.read_h5ad`` -- the installed anndata
version fails to parse this file's ``uns/log1p`` encoding, but the few arrays we need
(``obsm/X_pca``, ``obsm/spatial``, ``obs/batch``, ``obs/cell_label``) are read trivially with h5py.
"""

import argparse

import numpy as np


DEFAULT_H5AD_PATH = '/cv/data/braid/havivd/merfish_motor_cortex_atlas/st_data_processed.h5ad'
CENTER_LABEL = 'GABAergic'  # inhibitory neurons (obs['cell_label'] categories are
                            # ['GABAergic', 'Glutamatergic', 'Non-Neuronal'])


def _decode(cats):
    return [c.decode() if isinstance(c, bytes) else c for c in cats]


def load_atlas_fields(h5ad_path=DEFAULT_H5AD_PATH, feature_path=None):
    """Read the handful of atlas fields the niche pipeline needs, via raw h5py.

    :param feature_path: if given, overrides ``X_pca`` with a cached ``(n, d)`` .npy array (same
        row order as the atlas) instead of ``obsm/X_pca`` -- e.g. an alternate PCA computed by
        ``compute_pca16.py``. Every downstream consumer (niches, geometry training) is agnostic
        to which cell-feature space it's given.
    :returns: dict with X_pca (n, d) float32, spatial (n, 2) float32, batch_codes (n,) int,
        batch_categories (list[str]), label_codes (n,) int, label_categories (list[str]).
    """
    import h5py  # type: ignore

    with h5py.File(h5ad_path, 'r') as f:
        X_pca = (np.load(feature_path).astype(np.float32) if feature_path is not None
                else f['obsm/X_pca'][:].astype(np.float32))
        spatial = f['obsm/spatial'][:].astype(np.float32)
        batch_codes = f['obs/batch/codes'][:]
        batch_categories = _decode(f['obs/batch/categories'][:])
        label_codes = f['obs/cell_label/codes'][:]
        label_categories = _decode(f['obs/cell_label/categories'][:])
        subclass_codes = f['obs/subclass/codes'][:]
        subclass_categories = _decode(f['obs/subclass/categories'][:])
        type_codes = f['obs/cell_type/codes'][:]
        type_categories = _decode(f['obs/cell_type/categories'][:])

    return dict(X_pca=X_pca, spatial=spatial, batch_codes=batch_codes,
               batch_categories=batch_categories, label_codes=label_codes,
               label_categories=label_categories, subclass_codes=subclass_codes,
               subclass_categories=subclass_categories, type_codes=type_codes,
               type_categories=type_categories)


def _niches_for_batch(batch_indices, spatial, X_pca, center_local_pos, k):
    """k=128 spatial-neighbor niches (X_pca rows) for centers within a single batch.

    :param batch_indices: (n_batch,) global row indices of all cells in this batch (sorted).
    :param center_local_pos: (n_centers,) positions into ``batch_indices`` of the center cells.
    :returns: (n_centers, k, d_pca) float32 array.
    """
    from sklearn.neighbors import NearestNeighbors  # type: ignore

    batch_spatial = spatial[batch_indices]
    nbrs = NearestNeighbors(n_neighbors=min(k + 1, batch_indices.size)).fit(batch_spatial)
    _, nbr_local = nbrs.kneighbors(batch_spatial[center_local_pos])

    niches = np.empty((center_local_pos.size, k, X_pca.shape[1]), dtype=np.float32)
    for i, center_local in enumerate(center_local_pos):
        row = nbr_local[i]
        row = row[row != center_local][:k]
        if row.size < k:
            # Fallback (should not trigger given every batch has >=1160 cells, k=128): pad from
            # the unfiltered neighbor list so the niche still has exactly k members.
            pad = nbr_local[i][~np.isin(nbr_local[i], row)][:k - row.size]
            row = np.concatenate([row, pad])
        niches[i] = X_pca[batch_indices[row]]
    return niches


def build_niches(h5ad_path=DEFAULT_H5AD_PATH, k=128, seed=0, test_frac=0.2, test_n=None,
                 max_train=None, max_test=None, center_label=CENTER_LABEL, center_field='cell_label',
                 feature_path=None, return_anchors=False, verbose=True):
    """Build the full center-population niche set and split it into train/test.

    :param center_field: which ``obs`` categorical field ``center_label`` is a category of --
        e.g. ``'cell_label'`` (GABAergic/Glutamatergic/Non-Neuronal, the default) or ``'subclass'``
        (finer subtypes such as ``'Sst'``). Must have ``{center_field}_codes``/
        ``{center_field}_categories`` entries in :func:`load_atlas_fields`'s return dict.
    :param test_n: if given, use exactly this many niches for the test split (overrides
        ``test_frac``); the rest go to train (subject to ``max_train``).
    :param feature_path: see :func:`load_atlas_fields` -- overrides the cell-feature space.
    :param return_anchors: if True, also return each niche's own center (anchor) cell embedding
        -- in the same feature space as the niche members, but not itself a niche member -- for
        use as conditioning (e.g. ``RiemannianWassersteinFlowMatching(conditioning=train_anchors)``).
    :returns: (train_clouds, test_clouds) -- lists of (k, d_pca) float32 numpy arrays -- or, if
        ``return_anchors``, (train_clouds, test_clouds, train_anchors, test_anchors) with
        train_anchors/test_anchors (n, d_pca) float32 arrays aligned with train_clouds/test_clouds.
    """
    fields = load_atlas_fields(h5ad_path, feature_path=feature_path)
    X_pca, spatial = fields['X_pca'], fields['spatial']
    batch_codes, batch_categories = fields['batch_codes'], fields['batch_categories']
    # load_atlas_fields exposes cell_label as 'label_{codes,categories}' (not 'cell_label_*').
    field_key = 'label' if center_field == 'cell_label' else center_field
    center_field_codes = fields[f'{field_key}_codes']
    center_field_categories = fields[f'{field_key}_categories']
    type_codes, type_categories = fields['type_codes'], fields['type_categories']

    center_code = center_field_categories.index(center_label)
    is_center = center_field_codes == center_code

    all_niches = []
    all_batches = []
    all_types = []
    all_anchors = []
    n_dropped_batches = 0
    for b in range(len(batch_categories)):
        batch_indices = np.where(batch_codes == b)[0]
        if batch_indices.size < k + 1:
            n_dropped_batches += 1
            continue
        centers_in_batch = batch_indices[is_center[batch_indices]]
        if centers_in_batch.size == 0:
            continue
        center_local_pos = np.searchsorted(batch_indices, centers_in_batch)
        niches = _niches_for_batch(batch_indices, spatial, X_pca, center_local_pos, k)
        all_niches.append(niches)
        all_batches.append(np.full(centers_in_batch.size, b, dtype=np.int32))
        all_types.append(type_codes[centers_in_batch])
        all_anchors.append(X_pca[centers_in_batch])

    niches = np.concatenate(all_niches, axis=0)
    niche_batches = np.concatenate(all_batches, axis=0)
    niche_types = np.concatenate(all_types, axis=0)
    anchors = np.concatenate(all_anchors, axis=0)
    n_niches = niches.shape[0]

    if verbose:
        print(f"[niche_data] {n_niches} niches from {len(batch_categories) - n_dropped_batches} "
              f"batches ({n_dropped_batches} dropped: <{k + 1} cells) | center='{center_label}' "
              f"| niche shape (k={k}, d_pca={X_pca.shape[1]})", flush=True)
        type_counts = {type_categories[c]: int((niche_types == c).sum())
                      for c in np.unique(niche_types)}
        print(f"[niche_data] center cell_type composition: {type_counts}", flush=True)

    rng = np.random.default_rng(seed)

    if test_n is not None:
        # Stratified by the center's cell_type (obs['cell_type']) so the fixed-size test set
        # covers every inhibitory subtype present, not just whichever a plain random draw hits.
        n_test = int(test_n)
        present_types = np.unique(niche_types)
        type_counts = np.array([(niche_types == c).sum() for c in present_types])
        raw_alloc = n_test * type_counts / type_counts.sum()
        alloc = np.floor(raw_alloc).astype(int)
        # largest-remainder method so allocations sum to exactly n_test
        remainder = n_test - alloc.sum()
        order = np.argsort(-(raw_alloc - alloc))
        alloc[order[:remainder]] += 1
        alloc = np.minimum(alloc, type_counts)  # can't take more than exist in a stratum

        test_idx_parts = []
        for c, n_c in zip(present_types, alloc):
            stratum_idx = np.where(niche_types == c)[0]
            chosen = rng.choice(stratum_idx, size=int(n_c), replace=False)
            test_idx_parts.append(chosen)
        test_idx = rng.permutation(np.concatenate(test_idx_parts))
        train_idx = rng.permutation(np.setdiff1d(np.arange(n_niches), test_idx,
                                                 assume_unique=True))
        if verbose:
            strat_desc = {type_categories[c]: int(n_c) for c, n_c in zip(present_types, alloc)}
            print(f"[niche_data] stratified test set by cell_type: {strat_desc}", flush=True)
    else:
        perm = rng.permutation(n_niches)
        n_test = int(round(n_niches * test_frac))
        test_idx, train_idx = perm[:n_test], perm[n_test:]

    if max_train is not None:
        train_idx = train_idx[:max_train]
    if max_test is not None:
        test_idx = test_idx[:max_test]

    train_clouds = [niches[i] for i in train_idx]
    test_clouds = [niches[i] for i in test_idx]

    if verbose:
        test_desc = f"test_n={test_n}" if test_n is not None else f"test_frac={test_frac}"
        print(f"[niche_data] split -> {len(train_clouds)} train / {len(test_clouds)} test "
              f"(seed={seed}, {test_desc})", flush=True)

    if return_anchors:
        train_anchors = anchors[train_idx]
        test_anchors = anchors[test_idx]
        return train_clouds, test_clouds, train_anchors, test_anchors
    return train_clouds, test_clouds


def load_all_cell_pca(h5ad_path=DEFAULT_H5AD_PATH, feature_path=None):
    """All-cell ``X_pca`` for geometry pretraining (Sec. 6: learn on all cells, not just niches)."""
    with_fields = load_atlas_fields(h5ad_path, feature_path=feature_path)
    return with_fields['X_pca']


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--h5ad-path', type=str, default=DEFAULT_H5AD_PATH)
    p.add_argument('--k', type=int, default=128)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--max-train', type=int, default=20)
    p.add_argument('--max-test', type=int, default=10)
    return p.parse_args()


if __name__ == '__main__':
    # Standalone debug run: small caps, print shapes + a provenance spot-check.
    args = _parse_args()
    fields = load_atlas_fields(args.h5ad_path)
    print("label categories:", fields['label_categories'])
    print("n batches:", len(fields['batch_categories']))

    train_clouds, test_clouds = build_niches(
        args.h5ad_path, k=args.k, seed=args.seed,
        max_train=args.max_train, max_test=args.max_test)

    print(f"train_clouds: {len(train_clouds)} of shape {train_clouds[0].shape}")
    print(f"test_clouds: {len(test_clouds)} of shape {test_clouds[0].shape}")

    # Spot-check: no duplicate rows within a niche (center + neighbors should be 128 distinct cells)
    for name, clouds in [('train', train_clouds), ('test', test_clouds)]:
        for i, c in enumerate(clouds[:5]):
            n_unique_rows = np.unique(c, axis=0).shape[0]
            print(f"  {name}[{i}]: shape={c.shape}, unique rows={n_unique_rows}")
