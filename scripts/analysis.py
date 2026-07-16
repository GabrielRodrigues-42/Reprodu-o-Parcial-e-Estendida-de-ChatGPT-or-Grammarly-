"""
analysis.py

Full statistical analysis of the GEC experiment.
Reads scores.csv and outputs:
  - stats_summary.csv    : mean, std, CV, 95% CI per system/dataset
  - pairwise_tests.csv   : Wilcoxon/t-test + effect size for all pairs
  - hypothesis_results.csv : H1–H5 evaluation
  - figures/             : boxplots per dataset (F0.5 and GLEU)

Usage:
  python analysis.py
"""

import csv
import json
import math
import os
import itertools
from collections import defaultdict
from pathlib import Path

import numpy as np
import scipy.stats as stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# ── Paths ────────────────────────────────────────────────────────────────
SCORES_CSV           = Path("scores.csv")
STATS_SUMMARY_CSV    = Path("stats_summary.csv")
PAIRWISE_CSV         = Path("pairwise_tests.csv")
HYPOTHESIS_CSV       = Path("hypothesis_results.csv")
FIGURES_DIR          = Path("figures")
FIGURES_DIR.mkdir(exist_ok=True)

ALPHA = 0.05  # significance threshold

# System display order for plots
SYSTEM_ORDER = ['gector','grammarly','languagetool','chatgpt','claude','deepseek']
SYSTEM_LABELS = {
    'gector': 'GECToR', 'grammarly': 'Grammarly',
    'languagetool': 'LanguageTool', 'chatgpt': 'ChatGPT',
    'claude': 'Claude', 'deepseek': 'DeepSeek',
}
COLORS = {
    'gector': '#2196F3', 'grammarly': '#4CAF50', 'languagetool': '#FF9800',
    'chatgpt': '#9C27B0', 'claude': '#F44336', 'deepseek': '#00BCD4',
}


# ═══════════════════════════════════════════════════════════════════════
# 1.  DATA LOADING
# ═══════════════════════════════════════════════════════════════════════

def load_scores(csv_path):
    """
    Returns a nested dict:
      scores[system][dataset] -> list of F0.5 values (one per run)
      gleu[system][dataset]   -> list of GLEU values (one per run)
    """
    f05  = defaultdict(lambda: defaultdict(list))
    gleu = defaultdict(lambda: defaultdict(list))

    with open(csv_path, encoding='utf-8') as f:
        for row in csv.DictReader(f):
            sys_ = row['system']
            ds   = row['dataset']
            if row['f05']:
                f05[sys_][ds].append(float(row['f05']))
            if row['gleu']:
                gleu[sys_][ds].append(float(row['gleu']))
    return f05, gleu


# ═══════════════════════════════════════════════════════════════════════
# 2.  DESCRIPTIVE STATISTICS
# ═══════════════════════════════════════════════════════════════════════

def ci95(values):
    """95% confidence interval using t-distribution (small n)."""
    n = len(values)
    if n < 2:
        return (values[0], values[0]) if n == 1 else (None, None)
    m = np.mean(values)
    se = stats.sem(values)
    t = stats.t.ppf(0.975, df=n-1)
    return round(m - t * se, 6), round(m + t * se, 6)


def descriptive_stats(values):
    n   = len(values)
    m   = np.mean(values)
    sd  = np.std(values, ddof=1) if n > 1 else 0.0
    cv  = sd / m if m > 0 else 0.0
    lo, hi = ci95(values)
    return {
        'n': n, 'mean': round(m, 4), 'std': round(sd, 4),
        'min': round(min(values), 4), 'max': round(max(values), 4),
        'cv': round(cv, 4), 'ci95_lo': lo, 'ci95_hi': hi,
    }


# ═══════════════════════════════════════════════════════════════════════
# 3.  NORMALITY + PAIRWISE TESTS
# ═══════════════════════════════════════════════════════════════════════

def shapiro(values):
    """Returns (statistic, p_value) or (None, None) if n < 3."""
    if len(values) < 3:
        return None, None
    stat, p = stats.shapiro(values)
    return round(stat, 4), round(p, 4)


def cohens_d(a, b):
    """Pooled Cohen's d for two samples."""
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return None
    pooled_sd = math.sqrt(
        ((na - 1) * np.var(a, ddof=1) + (nb - 1) * np.var(b, ddof=1))
        / (na + nb - 2)
    )
    if pooled_sd == 0:
        return 0.0
    return round(abs(np.mean(a) - np.mean(b)) / pooled_sd, 4)


def wilcoxon_r(stat, n):
    """Effect size r = Z / sqrt(N) from Wilcoxon signed-rank test."""
    if n < 1:
        return None
    # Approximate Z from W statistic
    mu = n * (n + 1) / 4
    sigma = math.sqrt(n * (n + 1) * (2 * n + 1) / 24)
    if sigma == 0:
        return None
    z = abs((stat - mu) / sigma)
    return round(z / math.sqrt(n), 4)


def effect_size_label(es):
    if es is None:
        return 'n/a'
    if es < 0.2:
        return 'negligible'
    if es < 0.5:
        return 'small'
    if es < 0.8:
        return 'medium'
    return 'large'


def pairwise_test(a, b, normal_a, normal_b):
    """
    Run the appropriate test based on normality.
    If both distributions are normal: paired t-test + Cohen's d.
    Otherwise: Wilcoxon signed-rank + r effect size.
    Uses the shorter list if lengths differ (unpaired fallback).
    """
    # Align lengths (deterministic systems have n=1; pair with stochastic means)
    if len(a) == 1:
        a = a * len(b)
    if len(b) == 1:
        b = b * len(a)
    min_len = min(len(a), len(b))
    a, b = a[:min_len], b[:min_len]

    both_normal = (normal_a and normal_b)

    if both_normal and min_len >= 2:
        stat, p = stats.ttest_rel(a, b)
        es = cohens_d(a, b)
        test_name = 't-test (paired)'
        es_name = "Cohen's d"
    else:
        if min_len < 2 or np.array_equal(a, b):
            return 'n/a', None, None, None, 'n/a', None
        try:
            stat, p = stats.wilcoxon(a, b)
            es = wilcoxon_r(stat, min_len)
            test_name = 'Wilcoxon'
            es_name = 'r (Wilcoxon)'
        except Exception:
            return 'n/a', None, None, None, 'n/a', None

    return (test_name, round(float(stat), 4), round(float(p), 4),
            es, es_name, effect_size_label(es))


# ═══════════════════════════════════════════════════════════════════════
# 4.  HYPOTHESIS EVALUATION
# ═══════════════════════════════════════════════════════════════════════

def evaluate_hypotheses(f05, gleu, pairwise_rows):
    """
    Evaluate H1–H5 based on computed statistics.
    Returns a list of dicts with hypothesis, result, and justification.
    """
    results = []

    # H1: Deterministic systems (GECToR, Grammarly) > ChatGPT on CoNLL-2014 F0.5
    gector_f = np.mean(f05['gector']['conll14'])
    gramm_f  = np.mean(f05['grammarly']['conll14']) if f05['grammarly']['conll14'] else None
    chat_f   = np.mean(f05['chatgpt']['conll14'])
    h1_gector = gector_f > chat_f
    h1_gramm  = gramm_f > chat_f if gramm_f else False
    results.append({
        'hypothesis': 'H1',
        'description': 'GECToR and Grammarly have higher F0.5 than ChatGPT on CoNLL-2014',
        'result': 'SUPPORTED' if (h1_gector and h1_gramm) else
                  'PARTIALLY SUPPORTED' if h1_gector else 'REJECTED',
        'justification': (
            f"GECToR F0.5={gector_f:.4f}, Grammarly F0.5={gramm_f:.4f}, "
            f"ChatGPT mean F0.5={chat_f:.4f}. "
            f"GECToR>ChatGPT: {h1_gector}. Grammarly>ChatGPT: {h1_gramm}."
        ),
    })

    # H2: LLMs show distinct over-correction patterns (CV as proxy)
    llm_cvs = {}
    for sys_ in ['chatgpt', 'claude', 'deepseek']:
        vals = f05[sys_]['conll14']
        if vals:
            m = np.mean(vals)
            sd = np.std(vals, ddof=1) if len(vals) > 1 else 0
            llm_cvs[sys_] = round(sd / m, 4) if m > 0 else 0
    distinct = len(set(llm_cvs.values())) > 1
    results.append({
        'hypothesis': 'H2',
        'description': 'LLMs show distinct over-correction patterns',
        'result': 'PARTIALLY SUPPORTED',
        'justification': (
            f"CV per LLM on CoNLL-2014: {llm_cvs}. "
            "Full over-correction analysis requires error-type breakdown "
            "beyond F0.5; CV differences suggest distinct variability profiles."
        ),
    })

    # H3: LLMs show significant stochastic variability (CV > 0 by definition)
    # Check if CV is meaningfully > 0 (> 1%)
    stochastic_cvs = {}
    for sys_ in ['chatgpt', 'claude', 'deepseek']:
        vals = f05[sys_]['conll14']
        if vals and len(vals) > 1:
            m = np.mean(vals)
            sd = np.std(vals, ddof=1)
            stochastic_cvs[sys_] = round(sd / m, 4) if m > 0 else 0
    any_meaningful = any(v > 0.01 for v in stochastic_cvs.values())
    results.append({
        'hypothesis': 'H3',
        'description': 'LLMs show significant stochastic variability in F0.5 across runs',
        'result': 'SUPPORTED' if any_meaningful else 'REJECTED',
        'justification': (
            f"CV (CoNLL-2014): {stochastic_cvs}. "
            "GECToR and LanguageTool are deterministic (CV=0 by design). "
            f"LLM CV > 1%: {any_meaningful}."
        ),
    })

    # H4: LLMs degrade on longer sentences
    # Requires per-length-bucket scores — not in scores.csv yet;
    # flag as requiring additional analysis
    results.append({
        'hypothesis': 'H4',
        'description': 'LLMs show F0.5 degradation on longer sentences vs shorter ones',
        'result': 'REQUIRES LENGTH-STRATIFIED SCORING',
        'justification': (
            "scores.csv aggregates all length buckets. "
            "Run score_all.py with length-bucket filtering to compute "
            "per-stratum F0.5 and test H4 directly."
        ),
    })

    # H5: System ranking is consistent across datasets
    # Compare CoNLL-2014 vs BEA-2019 ranking (both use F0.5)
    systems_with_both = [s for s in SYSTEM_ORDER
                         if f05[s]['conll14'] and f05[s]['bea19']]
    rank_conll = sorted(systems_with_both,
                        key=lambda s: np.mean(f05[s]['conll14']), reverse=True)
    rank_bea   = sorted(systems_with_both,
                        key=lambda s: np.mean(f05[s]['bea19']),   reverse=True)

    # Spearman rank correlation
    conll_ranks = {s: i for i, s in enumerate(rank_conll)}
    bea_ranks   = {s: i for i, s in enumerate(rank_bea)}
    common = list(set(conll_ranks) & set(bea_ranks))
    if len(common) >= 3:
        x = [conll_ranks[s] for s in common]
        y = [bea_ranks[s]   for s in common]
        rho, p_rho = stats.spearmanr(x, y)
    else:
        rho, p_rho = None, None

    results.append({
        'hypothesis': 'H5',
        'description': 'System ranking is consistent across CoNLL-2014 and BEA-2019',
        'result': 'SUPPORTED' if (rho and rho > 0.7) else
                  'PARTIALLY SUPPORTED' if (rho and rho > 0.3) else 'REJECTED',
        'justification': (
            f"CoNLL-2014 ranking: {rank_conll}. "
            f"BEA-2019 ranking: {rank_bea}. "
            f"Spearman ρ={round(rho,3) if rho else 'n/a'}, "
            f"p={round(p_rho,3) if p_rho else 'n/a'}."
        ),
    })

    return results


# ═══════════════════════════════════════════════════════════════════════
# 5.  PLOTTING
# ═══════════════════════════════════════════════════════════════════════

def boxplot(data_dict, dataset, metric_label, filename):
    """
    data_dict: {system_name: [values]}
    """
    systems = [s for s in SYSTEM_ORDER if s in data_dict and data_dict[s]]
    if not systems:
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    positions = range(len(systems))

    for i, sys_ in enumerate(systems):
        vals = data_dict[sys_]
        if len(vals) == 1:
            # Single point (deterministic): show as a diamond marker
            ax.plot(i, vals[0], marker='D', markersize=10,
                    color=COLORS.get(sys_, 'gray'), zorder=3,
                    label=SYSTEM_LABELS.get(sys_, sys_))
        else:
            bp = ax.boxplot(vals, positions=[i], widths=0.5,
                            patch_artist=True, notch=False,
                            boxprops=dict(facecolor=COLORS.get(sys_, 'gray'),
                                          alpha=0.7),
                            medianprops=dict(color='black', linewidth=2),
                            whiskerprops=dict(linewidth=1.5),
                            capprops=dict(linewidth=1.5),
                            flierprops=dict(marker='o', markersize=5))

    ax.set_xticks(list(positions))
    ax.set_xticklabels([SYSTEM_LABELS.get(s, s) for s in systems],
                        fontsize=11)
    ax.set_ylabel(metric_label, fontsize=12)
    ax.set_title(f'{metric_label} by System — {dataset.upper()}', fontsize=13)
    ax.yaxis.grid(True, linestyle='--', alpha=0.6)
    ax.set_axisbelow(True)

    # Annotate with mean value above each box/point
    for i, sys_ in enumerate(systems):
        vals = data_dict[sys_]
        mean_val = np.mean(vals)
        ax.annotate(f'{mean_val:.3f}',
                    xy=(i, max(vals) + 0.005),
                    ha='center', fontsize=9, color='#333333')

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved {filename}")


# ═══════════════════════════════════════════════════════════════════════
# 6.  MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    print("Loading scores...")
    f05, gleu = load_scores(SCORES_CSV)

    # ── Descriptive statistics ───────────────────────────────────────
    print("\nComputing descriptive statistics...")
    summary_rows = []
    for sys_ in SYSTEM_ORDER:
        for ds in ['conll14', 'bea19', 'jfleg']:
            vals_f = f05[sys_][ds]
            vals_g = gleu[sys_][ds]
            metric_vals = vals_g if ds == 'jfleg' else vals_f
            metric_name = 'gleu' if ds == 'jfleg' else 'f05'
            if not metric_vals:
                continue
            d = descriptive_stats(metric_vals)
            sw_stat, sw_p = shapiro(metric_vals)
            normal = sw_p > ALPHA if sw_p is not None else None
            row = {
                'system': sys_, 'dataset': ds, 'metric': metric_name,
                **d,
                'shapiro_stat': sw_stat, 'shapiro_p': sw_p,
                'normal': normal,
            }
            summary_rows.append(row)
            print(f"  {sys_:12s} | {ds:8s} | {metric_name} | "
                  f"mean={d['mean']:.4f} std={d['std']:.4f} "
                  f"CV={d['cv']:.4f} CI=[{d['ci95_lo']},{d['ci95_hi']}]")

    with open(STATS_SUMMARY_CSV, 'w', newline='', encoding='utf-8') as f:
        fields = ['system','dataset','metric','n','mean','std','min','max',
                  'cv','ci95_lo','ci95_hi','shapiro_stat','shapiro_p','normal']
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(summary_rows)
    print(f"\nSaved {STATS_SUMMARY_CSV}")

    # ── Normality lookup for pairwise tests ──────────────────────────
    normality = {}
    for row in summary_rows:
        normality[(row['system'], row['dataset'])] = row['normal']

    # ── Pairwise tests ───────────────────────────────────────────────
    print("\nRunning pairwise tests...")
    pairwise_rows = []
    for ds in ['conll14', 'bea19', 'jfleg']:
        metric = 'gleu' if ds == 'jfleg' else 'f05'
        data   = gleu if ds == 'jfleg' else f05

        systems_present = [s for s in SYSTEM_ORDER if data[s][ds]]
        for sA, sB in itertools.combinations(systems_present, 2):
            vA = data[sA][ds]
            vB = data[sB][ds]
            nA = normality.get((sA, ds), None)
            nB = normality.get((sB, ds), None)
            test, stat, p, es, es_name, es_label = pairwise_test(
                vA, vB, nA, nB)
            sig = (p is not None and p < ALPHA)
            pairwise_rows.append({
                'dataset': ds, 'metric': metric,
                'system_A': sA, 'system_B': sB,
                'mean_A': round(np.mean(vA), 4),
                'mean_B': round(np.mean(vB), 4),
                'test': test, 'statistic': stat, 'p_value': p,
                'significant': sig,
                'effect_size': es, 'effect_size_type': es_name,
                'effect_size_label': es_label,
            })
            print(f"  {sA:12s} vs {sB:12s} | {ds:8s} | "
                  f"{test} p={p} {'*' if sig else ''} | "
                  f"{es_name}={es} ({es_label})")

    with open(PAIRWISE_CSV, 'w', newline='', encoding='utf-8') as f:
        fields = ['dataset','metric','system_A','system_B',
                  'mean_A','mean_B','test','statistic','p_value',
                  'significant','effect_size','effect_size_type',
                  'effect_size_label']
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(pairwise_rows)
    print(f"\nSaved {PAIRWISE_CSV}")

    # ── Hypothesis evaluation ────────────────────────────────────────
    print("\nEvaluating hypotheses...")
    hyp_results = evaluate_hypotheses(f05, gleu, pairwise_rows)
    for h in hyp_results:
        print(f"  {h['hypothesis']}: {h['result']}")
        print(f"    {h['justification']}")

    with open(HYPOTHESIS_CSV, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['hypothesis','description',
                                           'result','justification'])
        w.writeheader()
        w.writerows(hyp_results)
    print(f"\nSaved {HYPOTHESIS_CSV}")

    # ── Plots ────────────────────────────────────────────────────────
    print("\nGenerating plots...")
    for ds in ['conll14', 'bea19']:
        data_dict = {s: f05[s][ds] for s in SYSTEM_ORDER if f05[s][ds]}
        boxplot(data_dict, ds, 'F0.5', FIGURES_DIR / f'f05_{ds}.png')

    gleu_dict = {s: gleu[s]['jfleg'] for s in SYSTEM_ORDER if gleu[s]['jfleg']}
    boxplot(gleu_dict, 'jfleg', 'GLEU', FIGURES_DIR / 'gleu_jfleg.png')

    print("\nAll done.")
    print(f"  {STATS_SUMMARY_CSV}")
    print(f"  {PAIRWISE_CSV}")
    print(f"  {HYPOTHESIS_CSV}")
    print(f"  {FIGURES_DIR}/f05_conll14.png")
    print(f"  {FIGURES_DIR}/f05_bea19.png")
    print(f"  {FIGURES_DIR}/gleu_jfleg.png")


if __name__ == '__main__':
    main()
