"""Aggregate per-(model, seed) benchmark JSONs into summary tables (mirrors
bunny_mnist_benchmark/aggregate.py, generalized from one base metric to two: the learned
pullback-flow metric and plain Euclidean X_pca -- the experiment spec's robustness check, Sec. 7-bis).

Pools every generation across all seeds for each model and reports mean +/- std for:
    1-NNA-CD (real / fake / avg), 1-NNA-EMD (real / fake / avg), MMD-CD, MMD-EMD.

Usage:
    python aggregate.py [--in-dir results/merfish_niche_benchmark] [--csv summary.csv]
"""

import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np

MODELS = ['rwefm', 'wfm', 'setrfm', 'setfm', 'rwsfm', 'wsfm']
METRICS = ['learned', 'euclidean']
RANDOM_MODELS = {'setrfm', 'setfm'}   # random coupling -> no Sinkhorn iterations

_SCALARS = [
    ('1nna_cd', 'real'), ('1nna_cd', 'fake'), ('1nna_cd', 'avg'),
    ('1nna_emd', 'real'), ('1nna_emd', 'fake'), ('1nna_emd', 'avg'),
    ('mmd_cd',), ('mmd_emd',),
]


def _get(gen, metric, path):
    v = gen[metric]
    for k in path:
        v = v[k]
    return v


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    repo = os.path.abspath(os.path.join(here, '..', '..'))
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--in-dir', type=str,
                    default=os.path.join(repo, 'results', 'merfish_niche_benchmark'))
    ap.add_argument('--csv', type=str, default=None)
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.in_dir, '*_seed*.json')))
    if not files:
        raise SystemExit(f"no result JSONs found in {args.in_dir}")

    # pool per-generation values across seeds -> {(model, metric): {metric_key: [values]}}
    pooled = defaultdict(lambda: defaultdict(list))
    seeds = defaultdict(set)
    meta = defaultdict(lambda: {'train_n': None, 'test_n': None, 'max_p': None, 'sink': None,
                                'train_wall_h': []})
    for fp in files:
        with open(fp) as f:
            d = json.load(f)
        model = d['model']
        for metric in METRICS:
            key = (model, metric)
            seeds[key].add(d['seed'])
            m = meta[key]
            m['train_n'] = d.get('n_train')
            m['test_n'] = d.get('n_real')
            m['max_p'] = d.get('max_p')
            m['sink'] = d.get('sinkhorn_iters_auto')
            if d.get('train_wall_sec') is not None:
                m['train_wall_h'].append(d['train_wall_sec'] / 3600.0)
            for gen in d['gens']:
                for path in _SCALARS:
                    pooled[key]['_'.join(path)].append(_get(gen, metric, path))

    stats = {}
    for model in MODELS:
        for metric in METRICS:
            key = (model, metric)
            if key not in pooled:
                continue
            p = pooled[key]
            mt = meta[key]

            def dist_from_half(real_key, fake_key):
                real = np.asarray(p[real_key])
                fake = np.asarray(p[fake_key])
                d = 0.5 * (np.abs(real - 0.5) + np.abs(fake - 0.5))
                return float(d.mean()), float(d.std())

            train_h = (float(np.mean(mt['train_wall_h'])) if mt['train_wall_h'] else float('nan'))
            sink = '-' if model in RANDOM_MODELS else mt['sink']
            stats[key] = {
                'seeds': len(seeds[key]), 'n_gen': len(p['1nna_cd_real']),
                'nna_cd': dist_from_half('1nna_cd_real', '1nna_cd_fake'),
                'nna_emd': dist_from_half('1nna_emd_real', '1nna_emd_fake'),
                'mmd_cd': (float(np.mean(p['mmd_cd'])), float(np.std(p['mmd_cd']))),
                'mmd_emd': (float(np.mean(p['mmd_emd'])), float(np.std(p['mmd_emd']))),
                'train_h': train_h, 'sink': sink,
            }

    present_models = [m for m in MODELS if any((m, metric) in stats for metric in METRICS)]

    def render(metric, title, subtitle, metric_labels):
        col_headers = ['model'] + [sub for sub, _ in metric_labels]
        table = []
        for model in present_models:
            st = stats.get((model, metric))
            row = [model] + [(fn(st) if st else '-') for _, fn in metric_labels]
            table.append(row)
        widths = [max(len(col_headers[i]), max((len(r[i]) for r in table), default=0))
                  for i in range(len(col_headers))]
        print(f"{title} [{metric} metric]")
        if subtitle:
            print('  ' + subtitle)
        print('  ' + '  '.join(col_headers[i].ljust(widths[i]) for i in range(len(col_headers))))
        for r in table:
            print('  ' + '  '.join(r[i].ljust(widths[i]) for i in range(len(col_headers))))
        print()

    for metric in METRICS:
        if not any((m, metric) in stats for m in MODELS):
            continue
        print()
        render(metric, "=== 1-NNA: mean distance from 0.5  (0 = indistinguishable / best) ===",
               "per generation: mean(|acc_real - 0.5|, |acc_fake - 0.5|), pooled over seeds x gens",
               [('1NNA-CD', lambda st: f"{st['nna_cd'][0]:.3f}±{st['nna_cd'][1]:.3f}"),
                ('1NNA-EMD', lambda st: f"{st['nna_emd'][0]:.3f}±{st['nna_emd'][1]:.3f}")])
        render(metric, "=== MMD: minimum matching distance  (lower = better) ===", None,
               [('MMD-CD', lambda st: f"{st['mmd_cd'][0]:.2e}±{st['mmd_cd'][1]:.1e}"),
                ('MMD-EMD', lambda st: f"{st['mmd_emd'][0]:.2e}±{st['mmd_emd'][1]:.1e}")])
        render(metric, "=== training time & auto Sinkhorn iterations ===", None,
               [('train(h)', lambda st: f"{st['train_h']:.2f}"),
                ('sink', lambda st: str(st['sink']))])

    if args.csv:
        import csv
        with open(args.csv, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['model', 'metric', 'seeds', 'n_gen',
                        'nna_cd_dist', 'nna_cd_dist_std', 'nna_emd_dist', 'nna_emd_dist_std',
                        'mmd_cd', 'mmd_cd_std', 'mmd_emd', 'mmd_emd_std', 'train_h', 'sink'])
            for model in present_models:
                for metric in METRICS:
                    st = stats.get((model, metric))
                    if not st:
                        continue
                    w.writerow([model, metric, st['seeds'], st['n_gen'],
                                st['nna_cd'][0], st['nna_cd'][1],
                                st['nna_emd'][0], st['nna_emd'][1],
                                st['mmd_cd'][0], st['mmd_cd'][1],
                                st['mmd_emd'][0], st['mmd_emd'][1],
                                st['train_h'], st['sink']])
        print(f"wrote {args.csv}")


if __name__ == '__main__':
    main()
