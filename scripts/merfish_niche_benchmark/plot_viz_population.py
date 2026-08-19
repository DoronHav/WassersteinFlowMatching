"""Standalone per-population niche-composition UMAP plots (same logic as
examples/validate_merfish_niche_umap.ipynb's analyze_population), so a population can be plotted
as soon as its 4 model arrays land -- without waiting for the other population's generation jobs.

Loads real + {wfm,setfm,rwsfm,setrfm} generated niche arrays from a cache dir, reduces each niche
to a cell-type-abundance vector (k-NN cell_type classifier trained on the full-254-gene atlas),
fits one shared UMAP, and writes two PNGs (composition overlap + dominant cell type) to plots/.

Example:
    python plot_viz_population.py --pop GABAergic \\
      --cache-dir /cv/data/braid/havivd/merfish_motor_cortex_atlas/niche_umap_cache_full254/gaba
"""

import argparse
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


MODELS = ['wfm', 'setfm', 'rwsfm', 'setrfm']
MODEL_COLORS = {'wfm': '#1f77b4', 'setfm': '#7f7f7f', 'rwsfm': '#d62728', 'setrfm': '#2ca02c'}
REAL_COLOR = '#d0d0d0'


def _parse_args():
    here = os.path.dirname(os.path.abspath(__file__))
    repo = os.path.abspath(os.path.join(here, '..', '..'))
    bench = os.path.join(repo, 'results', 'merfish_niche_benchmark')
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--pop', required=True)
    p.add_argument('--cache-dir', required=True)
    p.add_argument('--feature-path', default=os.path.join(bench, 'X_full254.npy'))
    p.add_argument('--h5ad-path',
                   default='/cv/data/braid/havivd/merfish_motor_cortex_atlas/st_data_processed.h5ad')
    p.add_argument('--knn-k', type=int, default=15)
    p.add_argument('--plots-dir', default=os.path.join(bench, 'plots'))
    return p.parse_args()


def niche_abundance(clouds, clf, n_categories, chunk=20000):
    clouds = np.asarray(clouds, dtype=np.float32)
    n, k, d = clouds.shape
    flat = clouds.reshape(n * k, d)
    preds = np.empty(n * k, dtype=np.int32)
    for i in range(0, flat.shape[0], chunk):
        preds[i:i + chunk] = np.asarray(clf.predict(flat[i:i + chunk]))
    preds = preds.reshape(n, k)
    ab = np.zeros((n, n_categories), dtype=np.float32)
    for c in range(n_categories):
        ab[:, c] = (preds == c).mean(axis=1)
    return ab


def main():
    args = _parse_args()
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'src'))
    from niche_data import load_atlas_fields
    from cuml.neighbors import KNeighborsClassifier
    from cuml.manifold import UMAP

    os.makedirs(args.plots_dir, exist_ok=True)
    pop = args.pop

    fields = load_atlas_fields(args.h5ad_path, feature_path=args.feature_path)
    X_all = fields['X_pca'].astype(np.float32)
    type_codes, type_categories = fields['type_codes'], fields['type_categories']
    n_categories = len(type_categories)
    clf = KNeighborsClassifier(n_neighbors=args.knn_k)
    clf.fit(X_all, type_codes)
    print(f"[{pop}] classifier trained on {X_all.shape} ({n_categories} cell types)", flush=True)

    real = np.load(os.path.join(args.cache_dir, 'real_all_clouds.npy'), mmap_mode='r')
    real_ab = niche_abundance(real, clf, n_categories)
    print(f"[{pop}] {real.shape[0]} real niches", flush=True)

    gen_ab = {}
    for name in MODELS:
        f = os.path.join(args.cache_dir, f'generated_all_{name}.npy')
        gen_ab[name] = niche_abundance(np.load(f, mmap_mode='r'), clf, n_categories)
        print(f"[{pop}] {name} done", flush=True)

    labels = ['real'] * len(real_ab)
    stacked = [real_ab]
    for name in MODELS:
        stacked.append(gen_ab[name]); labels += [name] * len(gen_ab[name])
    stacked = np.concatenate(stacked, axis=0); labels = np.array(labels)
    coords = UMAP(n_neighbors=30, min_dist=0.3, metric='euclidean', random_state=0).fit_transform(stacked)
    real_xy = coords[labels == 'real']
    xlim = (coords[:, 0].min() - 1, coords[:, 0].max() + 1)
    ylim = (coords[:, 1].min() - 1, coords[:, 1].max() + 1)

    fig, axes = plt.subplots(1, len(MODELS), figsize=(5 * len(MODELS), 5), sharex=True, sharey=True)
    for ax, name in zip(np.atleast_1d(axes), MODELS):
        ax.scatter(real_xy[:, 0], real_xy[:, 1], s=6, c=REAL_COLOR, label='real', linewidths=0)
        gxy = coords[labels == name]
        ax.scatter(gxy[:, 0], gxy[:, 1], s=6, c=MODEL_COLORS[name], label=name, linewidths=0, alpha=0.7)
        ax.set_title(name); ax.set_xlim(xlim); ax.set_ylim(ylim)
        ax.set_xticks([]); ax.set_yticks([])
        ax.legend(loc='upper right', markerscale=2, frameon=False, fontsize=9)
    fig.suptitle(f'Niche cell-type composition, real vs. generated -- {pop} '
                 f'(full-254 pullback geometry, 128-step unconditional)', y=1.03)
    plt.tight_layout()
    out1 = os.path.join(args.plots_dir, f'niche_umap_full254_{pop.lower()}.png')
    plt.savefig(out1, dpi=150, bbox_inches='tight'); plt.close(fig)
    print(f"[{pop}] wrote {out1}", flush=True)

    cmap = plt.get_cmap('tab20', n_categories)
    panels = [('real', real_ab, real_xy)] + [(m, gen_ab[m], coords[labels == m]) for m in MODELS]
    fig, axes = plt.subplots(1, len(panels), figsize=(6 * len(panels), 6), sharex=True, sharey=True)
    for ax, (name, ab, xy) in zip(np.atleast_1d(axes), panels):
        dom = ab.argmax(axis=1)
        for c in np.unique(dom):
            m = dom == c
            ax.scatter(xy[m, 0], xy[m, 1], s=8, color=cmap(c), linewidths=0)
        ax.set_title(name); ax.set_xlim(xlim); ax.set_ylim(ylim)
        ax.set_xticks([]); ax.set_yticks([])
    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=cmap(c), markersize=7,
                          label=type_categories[c]) for c in range(n_categories)]
    fig.legend(handles=handles, loc='center left', bbox_to_anchor=(1.0, 0.5), frameon=False,
              fontsize=8, title='cell type')
    fig.suptitle(f'Dominant cell type per niche, real vs. generated -- {pop}', y=1.03)
    plt.tight_layout()
    out2 = os.path.join(args.plots_dir, f'niche_umap_full254_{pop.lower()}_dominant.png')
    plt.savefig(out2, dpi=150, bbox_inches='tight'); plt.close(fig)
    print(f"[{pop}] wrote {out2}", flush=True)


if __name__ == '__main__':
    main()
