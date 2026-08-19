"""Emit LaTeX tables (1-NNA + MMD, compact + mean/std) for bunny-MNIST digits 0, 2, 9.

Methods (as named in the paper tables):
    SetFM  <- setfm      SetRFM <- setrfm
    WFM (sample) <- wsfm     RWEFM (sample) <- rwsfm
1-NNA is reported as mean distance from 0.5 (lower is better), matching aggregate.py.
"""
import glob
import json
import os
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
IN_DIR = os.path.join(REPO, 'results', 'bunny_mnist_benchmark')

DIGITS = [2, 7, 9]
# model_key -> (compact header, shortstack header)
MODELS = [
    ('setfm',  'SetFM',            'SetFM'),
    ('setrfm', 'SetRFM',           'SetRFM'),
    ('wsfm',   r'\shortstack{WFM\\(sample)}',   r'\shortstack{WFM\\(sample)}'),
    ('rwsfm',  r'\shortstack{RWEFM\\(sample)}', r'\shortstack{RWEFM\\(sample)}'),
]

# --- pool per-generation values across seeds -------------------------------
pooled = defaultdict(lambda: defaultdict(list))
for fp in sorted(glob.glob(os.path.join(IN_DIR, 'digit*_*_seed*.json'))):
    with open(fp) as f:
        d = json.load(f)
    if d['digit'] not in DIGITS:
        continue
    key = (d['digit'], d['model'])
    for gen in d['gens']:
        pooled[key]['1nna_cd_real'].append(gen['1nna_cd']['real'])
        pooled[key]['1nna_cd_fake'].append(gen['1nna_cd']['fake'])
        pooled[key]['1nna_emd_real'].append(gen['1nna_emd']['real'])
        pooled[key]['1nna_emd_fake'].append(gen['1nna_emd']['fake'])
        pooled[key]['mmd_cd'].append(gen['mmd_cd'])
        pooled[key]['mmd_emd'].append(gen['mmd_emd'])


def nna_dist(key, metric):
    p = pooled[key]
    real = np.asarray(p[f'{metric}_real'])
    fake = np.asarray(p[f'{metric}_fake'])
    d = 0.5 * (np.abs(real - 0.5) + np.abs(fake - 0.5))
    return float(d.mean()), float(d.std())


def mmd_stat(key, metric):
    p = np.asarray(pooled[key][metric])
    return float(p.mean()), float(p.std())


# gather stats: stats[metric][(digit, model_key)] = (mean, std)
stats = {'nna_cd': {}, 'nna_emd': {}, 'mmd_cd': {}, 'mmd_emd': {}}
for digit in DIGITS:
    for mkey, _, _ in MODELS:
        k = (digit, mkey)
        if k not in pooled:
            continue
        stats['nna_cd'][k] = nna_dist(k, '1nna_cd')
        stats['nna_emd'][k] = nna_dist(k, '1nna_emd')
        stats['mmd_cd'][k] = mmd_stat(k, 'mmd_cd')
        stats['mmd_emd'][k] = mmd_stat(k, 'mmd_emd')

# --- highlighting: top-3 (lowest) per (digit, metric-CD/EMD) column ---------
C1, C2, C3 = '9DC183', '88C1D7', 'FFBB7D'   # 1st, 2nd, 3rd


def rank_colors(values):
    """values: dict model_key -> mean. return model_key -> color-or-None (top3 lowest)."""
    order = sorted(values, key=lambda m: values[m])
    colors = {}
    for i, m in enumerate(order[:3]):
        colors[m] = (C1, C2, C3)[i]
    return colors


def cell(color, txt):
    return (f'\\cellcolor[HTML]{{{color}}} {txt}' if color else txt)


# ---------------------------------------------------------------------------
# TABLE 1 & 2: compact single-value (mean) with top-3 highlighting
# ---------------------------------------------------------------------------
def compact_table(metric_cd, metric_emd, fmt, caption, label):
    lines = []
    lines.append(r'\begin{table}[t]')
    lines.append(r'\centering')
    lines.append(r'\resizebox{\textwidth}{!}{%')
    lines.append(r'\begin{tabular}{lcccccc}')
    lines.append(r'\toprule')
    lines.append(r'& \multicolumn{6}{c}{\textbf{Manifold: } Stanford bunny ($\mathcal{M}\subset\mathbb{R}^3$)} \\')
    lines.append(r'\cmidrule(lr){2-7}')
    lines.append(' & ' + ' & '.join(fr'\multicolumn{{2}}{{c}}{{{d}}}' for d in DIGITS) + r' \\')
    lines.append(r'\cmidrule(lr){2-3} \cmidrule(lr){4-5} \cmidrule(lr){6-7}')
    lines.append(r'Method & CD & EMD & CD & EMD & CD & EMD \\')
    lines.append(r'\midrule')

    # colors per digit per metric
    col_colors = {}
    for digit in DIGITS:
        for metric in (metric_cd, metric_emd):
            vals = {m: stats[metric][(digit, m)][0] for m, _, _ in MODELS
                    if (digit, m) in stats[metric]}
            col_colors[(digit, metric)] = rank_colors(vals)

    for mkey, header, _ in MODELS:
        cells = []
        for digit in DIGITS:
            for metric in (metric_cd, metric_emd):
                mean = stats[metric][(digit, mkey)][0]
                col = col_colors[(digit, metric)].get(mkey)
                cells.append(cell(col, fmt(mean)))
        lines.append(f'{header} & ' + ' & '.join(cells) + r' \\')

    lines.append(r'\bottomrule')
    lines.append(r'\end{tabular}%')
    lines.append(r'}')
    lines.append(r'\vspace{1mm}')
    lines.append(fr'\caption{{{caption}}}')
    lines.append(fr'\label{{{label}}}')
    lines.append(r'\end{table}')
    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# TABLE 3 & 4: mean +/- std, rows = digit x {CD, EMD}, cols = methods
# ---------------------------------------------------------------------------
def stdev_table(metric_cd, metric_emd, fmt, caption, label):
    lines = []
    lines.append(r'\begin{table}[h!]')
    lines.append(r'\centering')
    lines.append(r'\resizebox{\textwidth}{!}{%')
    lines.append(r'\begin{tabular}{lllcccc}')
    lines.append(r'\toprule')
    lines.append('& & & ' + ' & '.join(h for _, _, h in MODELS) + r' \\')
    lines.append(r'\midrule')

    for di, digit in enumerate(DIGITS):
        for ri, (mlabel, metric) in enumerate([('CD', metric_cd), ('EMD', metric_emd)]):
            prefix = (fr'$\mathcal{{M}}$ & {digit} & ' if ri == 0 else '&   & ')
            cells = []
            for mkey, _, _ in MODELS:
                mean, std = stats[metric][(digit, mkey)]
                cells.append(fmt(mean, std))
            lines.append(prefix + mlabel + '  & ' + ' & '.join(cells) + r' \\')
        if di != len(DIGITS) - 1:
            lines.append(r'\addlinespace')

    lines.append(r'\bottomrule')
    lines.append(r'\end{tabular}%')
    lines.append(r'}')
    lines.append(fr'\caption{{{caption}}}')
    lines.append(fr'\label{{{label}}}')
    lines.append(r'\end{table}')
    return '\n'.join(lines)


import math

f2 = lambda x: f'{x:.2f}'
fe = lambda x: f'{x:.1e}'.replace('e-0', 'e-').replace('e+0', 'e')
fstd2 = lambda m, s: fr'${m:.3f} \pm {s:.3f}$'


def fstde(m, s):
    # common exponent taken from the mean, rendered as (mean +/- std) x 10^exp
    exp = int(math.floor(math.log10(m))) if m > 0 else 0
    scale = 10.0 ** exp
    return fr'$({m / scale:.2f} \pm {s / scale:.2f})\times 10^{{{exp}}}$'

_digit_str = ', '.join(str(d) for d in DIGITS)
print(compact_table('nna_cd', 'nna_emd', f2,
      rf"\textbf{{1-NNA on bunny-MNIST (digits {_digit_str}).}} 1-NN accuracy distance from 0.5 "
      r"using Chamfer (CD) and Earth Mover's (EMD); lower is better. Averaged over 3 seeds "
      r"$\times$ 5 samplings. Top 3 per column highlighted "
      r"(\textcolor[HTML]{9DC183}{1st}, \textcolor[HTML]{88C1D7}{2nd}, \textcolor[HTML]{FFBB7D}{3rd}).",
      'tab:nna_bunny'))
print('\n\n')
print(compact_table('mmd_cd', 'mmd_emd', fe,
      rf"\textbf{{MMD on bunny-MNIST (digits {_digit_str}).}} Minimum matching distance "
      r"under CD and EMD; lower is better. Averaged over 3 seeds $\times$ 5 samplings. "
      r"Top 3 per column highlighted "
      r"(\textcolor[HTML]{9DC183}{1st}, \textcolor[HTML]{88C1D7}{2nd}, \textcolor[HTML]{FFBB7D}{3rd}).",
      'tab:mmd_bunny'))
print('\n\n')
print(stdev_table('nna_cd', 'nna_emd', fstd2,
      r"\textbf{1-NNA mean $\pm$ std on bunny-MNIST.} Distance from 0.5; lower is better. "
      r"Values averaged over 3 seeds $\times$ 5 samplings.",
      'tab:nna_std_bunny'))
print('\n\n')
print(stdev_table('mmd_cd', 'mmd_emd', fstde,
      r"\textbf{MMD mean $\pm$ std on bunny-MNIST.} Lower is better. "
      r"Values averaged over 3 seeds $\times$ 5 samplings.",
      'tab:mmd_std_bunny'))
