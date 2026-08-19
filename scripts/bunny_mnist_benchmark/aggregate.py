"""Aggregate per-(digit, model, seed) benchmark JSONs into summary tables.

Pools every generation across all seeds for each (digit, model) and reports mean +/- std for:
    1-NNA-CD  (real / fake / avg), 1-NNA-EMD (real / fake / avg), MMD-CD, MMD-EMD.

Usage:
    python aggregate.py [--in-dir results/bunny_mnist_benchmark] [--csv summary.csv]
"""

import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np

MODELS = ['rwefm', 'wfm', 'setrfm', 'setfm', 'rwsfm', 'wsfm']
DIGITS = [0, 1, 2, 9]
RANDOM_MODELS = {'setrfm', 'setfm'}   # random coupling -> no Sinkhorn iterations

_SCALARS = [
    ('1nna_cd', 'real'), ('1nna_cd', 'fake'), ('1nna_cd', 'avg'),
    ('1nna_emd', 'real'), ('1nna_emd', 'fake'), ('1nna_emd', 'avg'),
    ('mmd_cd',), ('mmd_emd',),
]


def _get(gen, path):
    v = gen
    for k in path:
        v = v[k]
    return v


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    repo = os.path.abspath(os.path.join(here, '..', '..'))
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--in-dir', type=str,
                    default=os.path.join(repo, 'results', 'bunny_mnist_benchmark'))
    ap.add_argument('--csv', type=str, default=None)
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.in_dir, 'digit*_*_seed*.json')))
    if not files:
        raise SystemExit(f"no result JSONs found in {args.in_dir}")

    # pool per-generation values across seeds -> {(digit, model): {metric_key: [values]}}
    pooled = defaultdict(lambda: defaultdict(list))
    seeds = defaultdict(set)
    meta = defaultdict(lambda: {'train_n': None, 'test_n': None, 'max_p': None,
                                'sink': None, 't500k': []})
    for fp in files:
        with open(fp) as f:
            d = json.load(f)
        key = (d['digit'], d['model'])
        seeds[key].add(d['seed'])
        m = meta[key]
        m['train_n'] = d.get('train_n', d.get('n_train'))
        m['test_n'] = d.get('test_n', d.get('n_real'))
        m['max_p'] = d.get('max_p', d.get('n_pts'))
        m['sink'] = d.get('sinkhorn_iters_auto')
        if d.get('time_500k_sec') is not None:
            m['t500k'].append(d['time_500k_sec'])
        for gen in d['gens']:
            for path in _SCALARS:
                pooled[key]['_'.join(path)].append(_get(gen, path))

    # per (digit, model) computed stats
    stats = {}
    for digit in DIGITS:
        for model in MODELS:
            key = (digit, model)
            if key not in pooled:
                continue
            p = pooled[key]
            mt = meta[key]

            def dist_from_half(real_key, fake_key):
                # per-generation mean class distance from chance: mean(|real-0.5|, |fake-0.5|)
                real = np.asarray(p[real_key])
                fake = np.asarray(p[fake_key])
                d = 0.5 * (np.abs(real - 0.5) + np.abs(fake - 0.5))
                return float(d.mean()), float(d.std())

            t500k_h = (float(np.mean(mt['t500k'])) / 3600.0) if mt['t500k'] else float('nan')
            sink = '-' if model in RANDOM_MODELS else mt['sink']   # random coupling: no Sinkhorn
            stats[key] = {
                'seeds': len(seeds[key]), 'n_gen': len(p['1nna_cd_real']),
                'nna_cd': dist_from_half('1nna_cd_real', '1nna_cd_fake'),
                'nna_emd': dist_from_half('1nna_emd_real', '1nna_emd_fake'),
                'mmd_cd': (float(np.mean(p['mmd_cd'])), float(np.std(p['mmd_cd']))),
                'mmd_emd': (float(np.mean(p['mmd_emd'])), float(np.std(p['mmd_emd']))),
                't500k_h': t500k_h, 'sink': sink,
            }

    present_models = [m for m in MODELS if any((d, m) in stats for d in DIGITS)]
    present_digits = [d for d in DIGITS if any((d, m) in stats for m in MODELS)]

    def render(title, subtitle, metric_labels):
        """metric_labels: list of (sub_label, fn) where fn(stats_entry) -> str, applied per digit."""
        col_headers = ['model']
        for digit in present_digits:
            for sub, _ in metric_labels:
                col_headers.append(f"d{digit}-{sub}")
        table = []
        for model in present_models:
            row = [model]
            for digit in present_digits:
                st = stats.get((digit, model))
                for _, fn in metric_labels:
                    row.append(fn(st) if st else '-')
            table.append(row)
        widths = [max(len(col_headers[i]), max((len(r[i]) for r in table), default=0))
                  for i in range(len(col_headers))]
        print(title)
        if subtitle:
            print('  ' + subtitle)
        print('  ' + '  '.join(col_headers[i].ljust(widths[i]) for i in range(len(col_headers))))
        for r in table:
            print('  ' + '  '.join(r[i].ljust(widths[i]) for i in range(len(col_headers))))
        print()

    print()
    render("=== 1-NNA: mean distance from 0.5  (0 = indistinguishable / best) ===",
           "per generation: mean(|acc_real - 0.5|, |acc_fake - 0.5|), pooled over seeds x gens",
           [('CD', lambda st: f"{st['nna_cd'][0]:.3f}±{st['nna_cd'][1]:.3f}"),
            ('EMD', lambda st: f"{st['nna_emd'][0]:.3f}±{st['nna_emd'][1]:.3f}")])

    render("=== MMD: minimum matching distance  (lower = better) ===", None,
           [('CD', lambda st: f"{st['mmd_cd'][0]:.2e}±{st['mmd_cd'][1]:.1e}"),
            ('EMD', lambda st: f"{st['mmd_emd'][0]:.2e}±{st['mmd_emd'][1]:.1e}")])

    render("=== training time (extrapolated to 500k steps) & auto Sinkhorn iterations ===", None,
           [('t500k(h)', lambda st: f"{st['t500k_h']:.2f}"),
            ('sink', lambda st: str(st['sink']))])

    if args.csv:
        import csv
        with open(args.csv, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['digit', 'model', 'seeds', 'n_gen',
                        'nna_cd_dist', 'nna_cd_dist_std', 'nna_emd_dist', 'nna_emd_dist_std',
                        'mmd_cd', 'mmd_cd_std', 'mmd_emd', 'mmd_emd_std', 't500k_h', 'sink'])
            for digit in present_digits:
                for model in present_models:
                    st = stats.get((digit, model))
                    if not st:
                        continue
                    w.writerow([digit, model, st['seeds'], st['n_gen'],
                                st['nna_cd'][0], st['nna_cd'][1],
                                st['nna_emd'][0], st['nna_emd'][1],
                                st['mmd_cd'][0], st['mmd_cd'][1],
                                st['mmd_emd'][0], st['mmd_emd'][1],
                                st['t500k_h'], st['sink']])
        print(f"wrote {args.csv}")


if __name__ == '__main__':
    main()
