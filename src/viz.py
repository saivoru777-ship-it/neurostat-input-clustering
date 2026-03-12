"""Visualization functions for input-specific clustering analysis.

All figures: Wong 2011 colorblind palette, 300 DPI PDF, single/double column.
Reuses style constants from neurostat.viz.figures.
"""

import numpy as np
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba

from neurostat.viz.figures import _apply_style, _save_fig, COLORS, SINGLE_COL, DOUBLE_COL

# Extended Wong 2011 palette for partner coloring
PARTNER_COLORS = [
    '#0072B2', '#D55E00', '#009E73', '#E69F00', '#56B4E9',
    '#CC79A7', '#F0E442', '#000000', '#882255', '#44AA99',
]


def plot_soma_distance_profiles(neuron_profiles, save_path=None):
    """Histograms of synapse soma distance per neuron + fitted KDE.

    Parameters
    ----------
    neuron_profiles : list of dict
        Each dict: {label, soma_dists, bin_centers, density}
    """
    _apply_style()
    n = len(neuron_profiles)
    cols = min(4, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(DOUBLE_COL, rows * 1.8))
    if n == 1:
        axes = np.array([axes])
    axes = np.atleast_2d(axes)

    for i, prof in enumerate(neuron_profiles):
        ax = axes.flat[i]
        soma_raw = prof['soma_dists']
        soma_raw = soma_raw[np.isfinite(soma_raw)]
        soma_um = soma_raw / 1000.0
        ax.hist(soma_um, bins=25, density=True, color=COLORS['fill'],
                alpha=0.5, edgecolor='white', linewidth=0.3)

        if 'bin_centers' in prof and 'density' in prof:
            bc_um = prof['bin_centers'] / 1000.0
            # Normalize density for display
            dens = prof['density']
            if dens.max() > 0:
                dens_norm = dens / dens.max() * ax.get_ylim()[1] * 0.8
                ax.plot(bc_um, dens_norm, '-', color=COLORS['multiscale'],
                        linewidth=1.5, label='KDE density')

        ax.set_xlabel('Soma distance (μm)')
        ax.set_ylabel('Density')
        ax.set_title(prof['label'], fontsize=7)

    for i in range(n, len(axes.flat)):
        axes.flat[i].set_visible(False)

    fig.suptitle('Synapse soma-distance profiles', fontsize=10, fontweight='bold')
    fig.tight_layout()
    if save_path:
        _save_fig(fig, save_path)
    return fig


def plot_null_comparison(neuron_comparisons, save_path=None):
    """VMR curves with both null envelopes (uniform vs soma-distance) per neuron.

    Parameters
    ----------
    neuron_comparisons : list of dict
        Each: {label, scales, real_variance, uniform_envelope, soma_envelope}
        Each envelope: {mean, p5, p95}
    """
    _apply_style()
    n = len(neuron_comparisons)
    cols = min(3, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(DOUBLE_COL, rows * 2.2))
    if n == 1:
        axes = np.array([axes])
    axes = np.atleast_2d(axes)

    for i, comp in enumerate(neuron_comparisons):
        ax = axes.flat[i]
        scales_um = np.array(comp['scales']) / 1000.0

        # Real data
        ax.plot(scales_um, comp['real_variance'], 'o-', color='k',
                markersize=3, linewidth=1.2, label='Data', zorder=5)

        # Uniform null envelope
        ue = comp['uniform_envelope']
        if ue['mean']:
            ax.fill_between(scales_um, ue['p5'], ue['p95'],
                            color=COLORS['fill'], alpha=0.25, label='Uniform null')

        # Soma-distance null envelope
        se = comp['soma_envelope']
        if se['mean']:
            ax.fill_between(scales_um, se['p5'], se['p95'],
                            color=COLORS['path_nn'], alpha=0.25, label='Soma-dist null')

        ax.set_xlabel('Scale (μm)')
        ax.set_ylabel('VMR')
        ax.set_xscale('log')
        ax.set_title(comp['label'], fontsize=7)
        if i == 0:
            ax.legend(frameon=False, fontsize=5)

    for i in range(n, len(axes.flat)):
        axes.flat[i].set_visible(False)

    fig.suptitle('Null model comparison', fontsize=10, fontweight='bold')
    fig.tight_layout()
    if save_path:
        _save_fig(fig, save_path)
    return fig


def plot_partner_volcano(neuron_volcanos, save_path=None):
    """Per-neuron volcano plot: z-score vs -log10(p), colored by cell type.

    Parameters
    ----------
    neuron_volcanos : list of dict
        Each: {label, partner_results (list with z_distance, p_distance, pre_cell_type)}
    """
    _apply_style()
    n = len(neuron_volcanos)
    cols = min(3, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(DOUBLE_COL, rows * 2.5))
    if n == 1:
        axes = np.array([axes])
    axes = np.atleast_2d(axes)

    type_color_map = {
        'excitatory': '#D55E00',
        'inhibitory': '#0072B2',
        'other': '#999999',
        'unknown': '#CCCCCC',
    }

    for i, vol in enumerate(neuron_volcanos):
        ax = axes.flat[i]
        for r in vol['partner_results']:
            z = r['z_distance']
            p = max(r['p_distance'], 1e-10)  # avoid log(0)
            neg_log_p = -np.log10(p)
            ctype = r.get('pre_cell_type', 'unknown')
            color = type_color_map.get(ctype, '#999999')
            ax.scatter(z, neg_log_p, c=color, s=15, alpha=0.7, edgecolors='none')

        # Significance threshold
        ax.axhline(-np.log10(0.05), color='k', linestyle='--', linewidth=0.5,
                    alpha=0.5)
        ax.set_xlabel('Z-score (distance)')
        ax.set_ylabel('-log10(p)')
        ax.set_title(vol['label'], fontsize=7)

    for i in range(n, len(axes.flat)):
        axes.flat[i].set_visible(False)

    fig.suptitle('Partner clustering volcano plots', fontsize=10, fontweight='bold')
    fig.tight_layout()
    if save_path:
        _save_fig(fig, save_path)
    return fig


def plot_enrichment_summary(enrichment_data, save_path=None):
    """Bar chart: enrichment ratio per neuron, grouped by type, all k tiers.

    Parameters
    ----------
    enrichment_data : list of dict
        Each: {label, neuron_type, enrichment (dict from compute_enrichment_summary)}
    """
    _apply_style()
    n = len(enrichment_data)
    tiers = ['k>=3', 'k>=5', 'k>=8']
    tier_colors = ['#56B4E9', '#0072B2', '#003F5C']

    fig, ax = plt.subplots(figsize=(DOUBLE_COL, 3.5))

    x = np.arange(n)
    width = 0.25

    for ti, tier in enumerate(tiers):
        ratios = []
        for ed in enrichment_data:
            enr = ed['enrichment'].get(tier, {})
            ratios.append(enr.get('enrichment_ratio', 0.0))
        bars = ax.bar(x + ti * width, ratios, width, label=tier,
                      color=tier_colors[ti], alpha=0.8)

        # Add significance stars
        for j, ed in enumerate(enrichment_data):
            enr = ed['enrichment'].get(tier, {})
            bp = enr.get('binomial_p', 1.0)
            if bp < 0.001:
                star = '***'
            elif bp < 0.01:
                star = '**'
            elif bp < 0.05:
                star = '*'
            else:
                star = ''
            if star:
                ax.text(x[j] + ti * width, ratios[j] + 0.1, star,
                        ha='center', va='bottom', fontsize=6)

    ax.axhline(1.0, color='k', linestyle='--', linewidth=0.5, alpha=0.5,
               label='Expected (no enrichment)')
    ax.set_xticks(x + width)
    ax.set_xticklabels([ed['label'] for ed in enrichment_data],
                       rotation=45, ha='right', fontsize=6)
    ax.set_ylabel('Enrichment ratio\n(observed / expected clustering)')
    ax.set_title('Partner clustering enrichment', fontweight='bold')
    ax.legend(frameon=False, fontsize=6, ncol=4)

    fig.tight_layout()
    if save_path:
        _save_fig(fig, save_path)
    return fig


def plot_label_shuffle(shuffle_results, save_path=None):
    """Histogram: real vs label-shuffled clustering fractions.

    Parameters
    ----------
    shuffle_results : list of dict
        Each: {label, real_frac_significant, shuffled_fracs}
    """
    _apply_style()
    n = len(shuffle_results)
    fig, axes = plt.subplots(1, n, figsize=(DOUBLE_COL, 2.5))
    if n == 1:
        axes = [axes]

    for i, sr in enumerate(shuffle_results):
        ax = axes[i]
        ax.hist(sr['shuffled_fracs'], bins=20, density=True,
                color=COLORS['neutral'], alpha=0.6, label='Label-shuffled')
        ax.axvline(sr['real_frac_significant'], color=COLORS['path_nn'],
                   linewidth=2, label=f"Real ({sr['real_frac_significant']:.2f})")
        ax.set_xlabel('Fraction significant')
        ax.set_ylabel('Density')
        ax.set_title(sr['label'], fontsize=7)
        ax.legend(frameon=False, fontsize=5)

    fig.suptitle('Label-shuffle control', fontsize=10, fontweight='bold')
    fig.tight_layout()
    if save_path:
        _save_fig(fig, save_path)
    return fig


def plot_branch_concentration(concentration_data, save_path=None):
    """Scatter: branch entropy vs mean pairwise distance, colored by significance.

    Parameters
    ----------
    concentration_data : list of dict
        Each: {label, partner_results (list with observed stats and p-values)}
    """
    _apply_style()
    n = len(concentration_data)
    cols = min(3, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(DOUBLE_COL, rows * 2.5))
    if n == 1:
        axes = np.array([axes])
    axes = np.atleast_2d(axes)

    for i, cd in enumerate(concentration_data):
        ax = axes.flat[i]
        for r in cd['partner_results']:
            obs = r['observed']
            is_sig = r.get('bh_significant_k5', False)
            color = COLORS['path_nn'] if is_sig else COLORS['neutral']
            alpha = 0.9 if is_sig else 0.4
            ax.scatter(obs['mean_pairwise_distance'] / 1000.0,
                       obs['branch_entropy'],
                       c=color, s=max(10, r['k'] * 2), alpha=alpha,
                       edgecolors='none')

        ax.set_xlabel('Mean pairwise dist (μm)')
        ax.set_ylabel('Branch entropy (bits)')
        ax.set_title(cd['label'], fontsize=7)

    for i in range(n, len(axes.flat)):
        axes.flat[i].set_visible(False)

    fig.suptitle('Branch concentration vs geodesic distance', fontsize=10,
                 fontweight='bold')
    fig.tight_layout()
    if save_path:
        _save_fig(fig, save_path)
    return fig


def plot_type_fingerprints(type_results_per_neuron, save_path=None):
    """Grid: multiscale fingerprints per broad presynaptic cell type.

    Parameters
    ----------
    type_results_per_neuron : list of dict
        Each: {label, type_results (from per_type_multiscale_test)}
    """
    _apply_style()

    # Collect all cell types across neurons
    all_types = set()
    for nr in type_results_per_neuron:
        all_types.update(nr['type_results'].keys())
    all_types = sorted(all_types)

    n_types = len(all_types)
    n_neurons = len(type_results_per_neuron)
    if n_types == 0 or n_neurons == 0:
        return None

    fig, axes = plt.subplots(n_types, n_neurons,
                              figsize=(DOUBLE_COL, n_types * 1.5),
                              squeeze=False)

    type_colors = {
        'excitatory': '#D55E00',
        'inhibitory': '#0072B2',
    }

    for ti, ctype in enumerate(all_types):
        for ni, nr in enumerate(type_results_per_neuron):
            ax = axes[ti, ni]
            tr = nr['type_results'].get(ctype)

            if tr is None:
                ax.text(0.5, 0.5, 'N/A', ha='center', va='center',
                        transform=ax.transAxes, fontsize=6, color='gray')
                ax.set_visible(True)
            else:
                curves = tr['curves']
                scales_um = np.array(curves['scales']) / 1000.0
                color = type_colors.get(ctype, '#999999')
                ax.plot(scales_um, curves['variance_values'], 'o-',
                        color=color, markersize=2, linewidth=0.8)

                # Null envelope
                env = tr['mock_envelope']
                if env['variance_mean']:
                    ax.fill_between(scales_um, env['variance_p5'],
                                    env['variance_p95'],
                                    color=color, alpha=0.15)

                ax.set_xscale('log')
                p_var = tr['variance_test']['p_value']
                if p_var < 0.05:
                    ax.set_title(f'p={p_var:.3f}*', fontsize=5, color='red')
                else:
                    ax.set_title(f'p={p_var:.3f}', fontsize=5)

            if ti == n_types - 1:
                ax.set_xlabel('Scale', fontsize=5)
            if ni == 0:
                ax.set_ylabel(f'{ctype}\n(n={tr["n_synapses"] if tr else 0})',
                              fontsize=5)
            if ti == 0:
                ax.set_title(nr['label'], fontsize=6, fontweight='bold')

    fig.suptitle('Per-type multiscale fingerprints', fontsize=10, fontweight='bold')
    fig.tight_layout()
    if save_path:
        _save_fig(fig, save_path)
    return fig


def plot_dendrite_input_map(skeleton, snap_result, pre_root_ids,
                             top_n_partners=8, title='', save_path=None):
    """3D dendrite with synapses colored by presynaptic partner.

    Parameters
    ----------
    skeleton : NeuronSkeleton
    snap_result : SnapResult
    pre_root_ids : ndarray
    top_n_partners : int
        Color the top N partners by synapse count; rest in gray.
    title : str
    save_path : str or Path
    """
    _apply_style()
    fig = plt.figure(figsize=(SINGLE_COL * 2, SINGLE_COL * 2))
    ax = fig.add_subplot(111, projection='3d')

    # Draw skeleton
    for branch in skeleton.branches:
        xs = [skeleton.nodes[nid].x for nid in branch.node_ids]
        ys = [skeleton.nodes[nid].y for nid in branch.node_ids]
        zs = [skeleton.nodes[nid].z for nid in branch.node_ids]
        ax.plot(xs, ys, zs, '-', color='#CCCCCC', linewidth=0.3, alpha=0.5)

    # Reconstruct 3D positions from snap
    n = len(snap_result.branch_ids)
    positions = np.empty((n, 3))
    for i in range(n):
        bi = snap_result.branch_ids[i]
        pos = snap_result.branch_positions[i]
        branch = skeleton.branches[bi]
        cum = 0.0
        for ei in range(len(branch.edge_lengths)):
            elen = branch.edge_lengths[ei]
            if cum + elen >= pos or ei == len(branch.edge_lengths) - 1:
                t = (pos - cum) / elen if elen > 0 else 0
                t = np.clip(t, 0, 1)
                n0 = skeleton.nodes[branch.node_ids[ei]]
                n1 = skeleton.nodes[branch.node_ids[ei + 1]]
                positions[i] = [
                    n0.x + t * (n1.x - n0.x),
                    n0.y + t * (n1.y - n0.y),
                    n0.z + t * (n1.z - n0.z),
                ]
                break
            cum += elen

    # Find top partners
    from collections import Counter
    partner_counts = Counter(pre_root_ids)
    top_partners = [pid for pid, _ in partner_counts.most_common(top_n_partners)]
    partner_to_color = {pid: PARTNER_COLORS[i % len(PARTNER_COLORS)]
                        for i, pid in enumerate(top_partners)}

    # Plot synapses
    for pid in top_partners:
        mask = pre_root_ids == pid
        if mask.sum() == 0:
            continue
        color = partner_to_color[pid]
        ax.scatter(positions[mask, 0], positions[mask, 1], positions[mask, 2],
                   c=color, s=12, alpha=0.8, label=f'{pid} (n={mask.sum()})',
                   edgecolors='white', linewidths=0.2)

    # Other synapses in gray
    other_mask = ~np.isin(pre_root_ids, top_partners)
    if other_mask.sum() > 0:
        ax.scatter(positions[other_mask, 0], positions[other_mask, 1],
                   positions[other_mask, 2],
                   c='#CCCCCC', s=4, alpha=0.3, label='Other')

    ax.set_xlabel('X (nm)', fontsize=6)
    ax.set_ylabel('Y (nm)', fontsize=6)
    ax.set_zlabel('Z (nm)', fontsize=6)
    ax.set_title(title, fontsize=8)
    ax.legend(frameon=False, fontsize=4, loc='upper left', ncol=2)

    fig.tight_layout()
    if save_path:
        _save_fig(fig, save_path)
    return fig


def plot_compactness_cdf(neuron_results, save_path=None):
    """The "killer figure": CDF of partner compactness ratios vs null.

    Panel A: CDF for one example neuron (real vs label-shuffle null).
    Panel B: Pooled CDF across all neurons, stratified by post-synaptic type.
    Panel C: Per-neuron median compactness ratio forest plot with bootstrap CI.

    Parameters
    ----------
    neuron_results : list of dict
        Each: {label, mtype, partner_results}
        partner_results: list of dicts with 'effect_ratio' and 'k' fields.
    """
    _apply_style()

    # ── Collect compactness ratios per neuron (k>=5 partners) ──
    neuron_ratios = {}
    neuron_meta = {}
    for nr in neuron_results:
        label = nr['label']
        ratios = []
        for pr in nr['partner_results']:
            er = pr.get('effect_ratio')
            if er is not None and np.isfinite(er) and pr['k'] >= 5:
                ratios.append(er)
        if ratios:
            neuron_ratios[label] = np.array(ratios)
            neuron_meta[label] = {'mtype': nr.get('mtype', '')}

    if not neuron_ratios:
        return None

    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE_COL, 2.8),
                              gridspec_kw={'width_ratios': [1, 1, 0.8]})

    # ── Panel A: Example neuron CDF ──
    ax = axes[0]
    # Pick strongest example (highest clustering fraction)
    example_label = max(neuron_ratios,
                        key=lambda l: np.mean(neuron_ratios[l] < 1.0))
    ratios = neuron_ratios[example_label]
    sorted_r = np.sort(ratios)
    cdf = np.arange(1, len(sorted_r) + 1) / len(sorted_r)
    ax.plot(sorted_r, cdf, '-', color='k', linewidth=1.5,
            label=f'Real ({example_label})')

    # Null reference: if all partners were at ratio=1.0 (uniform CDF)
    ax.plot([0, 2], [0, 1], '-', color='#CCCCCC', linewidth=1,
            alpha=0.5, label='If all null-like')

    ax.axvline(1.0, color='k', linestyle='--', linewidth=0.5, alpha=0.5)
    ax.set_xlabel('Compactness ratio\n(observed / null distance)')
    ax.set_ylabel('Cumulative fraction of partners')
    ax.set_title('A. Example neuron', fontsize=8, fontweight='bold')
    ax.legend(frameon=False, fontsize=5, loc='lower right')
    ax.set_xlim(0, 2.0)

    # ── Panel B: Pooled CDF by post-synaptic type ──
    ax = axes[1]
    exc_ratios = []
    inh_ratios = []
    bpc_ratios = []
    for label, ratios in neuron_ratios.items():
        mtype = neuron_meta[label]['mtype']
        if mtype == 'BPC':
            bpc_ratios.extend(ratios)
        elif label.startswith('exc_'):
            exc_ratios.extend(ratios)
        else:
            inh_ratios.extend(ratios)

    for pool, color, name in [
        (exc_ratios, '#D55E00', 'Excitatory post'),
        (inh_ratios, '#0072B2', 'Inhibitory post'),
        (bpc_ratios, '#999999', 'BPC post'),
    ]:
        if not pool:
            continue
        sorted_p = np.sort(pool)
        cdf = np.arange(1, len(sorted_p) + 1) / len(sorted_p)
        ax.plot(sorted_p, cdf, '-', color=color, linewidth=1.5, label=name)

    ax.axvline(1.0, color='k', linestyle='--', linewidth=0.5, alpha=0.5)
    ax.set_xlabel('Compactness ratio\n(observed / null distance)')
    ax.set_ylabel('Cumulative fraction of partners')
    ax.set_title('B. Pooled by neuron type', fontsize=8, fontweight='bold')
    ax.legend(frameon=False, fontsize=5, loc='lower right')
    ax.set_xlim(0, 2.0)

    # ── Panel C: Forest plot of per-neuron median with bootstrap CI ──
    ax = axes[2]
    plot_labels = []
    medians = []
    ci_lo = []
    ci_hi = []
    colors = []

    for nr in neuron_results:
        label = nr['label']
        if label not in neuron_ratios:
            continue
        ratios = neuron_ratios[label]
        plot_labels.append(label)
        med = np.median(ratios)
        medians.append(med)
        # Bootstrap 95% CI for median
        rng = np.random.default_rng(42)
        boot_medians = [np.median(rng.choice(ratios, size=len(ratios), replace=True))
                        for _ in range(1000)]
        boot_medians = np.array(boot_medians)
        ci_lo.append(np.percentile(boot_medians, 2.5))
        ci_hi.append(np.percentile(boot_medians, 97.5))

        mtype = neuron_meta[label]['mtype']
        if mtype == 'BPC':
            colors.append('#999999')
        elif label.startswith('exc_'):
            colors.append('#D55E00')
        else:
            colors.append('#0072B2')

    y = np.arange(len(plot_labels))
    for i in range(len(plot_labels)):
        ax.plot([ci_lo[i], ci_hi[i]], [y[i], y[i]], '-',
                color=colors[i], linewidth=1.5)
        ax.plot(medians[i], y[i], 'o', color=colors[i], markersize=5)

    ax.axvline(1.0, color='k', linestyle='--', linewidth=0.5, alpha=0.5)
    ax.set_yticks(y)
    ax.set_yticklabels(plot_labels, fontsize=5)
    ax.set_xlabel('Median compactness\nratio (95% CI)')
    ax.set_title('C. Per-neuron median', fontsize=8, fontweight='bold')
    ax.invert_yaxis()

    fig.suptitle('Partner compactness: observed vs null distance ratio',
                 fontsize=9, fontweight='bold')
    fig.tight_layout()
    if save_path:
        _save_fig(fig, save_path)
    return fig


def plot_k_vs_clustering(neuron_results, save_path=None):
    """Show how clustering strength varies with partner synapse count k.

    Panel A: Clustering fraction by k bin.
    Panel B: Median effect ratio by k bin with IQR.

    Parameters
    ----------
    neuron_results : list of dict
        Each: {label, partner_results}
    """
    _apply_style()

    k_bins = [(3, 4), (5, 7), (8, 12), (13, 999)]
    bin_labels = ['3-4', '5-7', '8-12', '13+']

    sig_fracs = []
    effect_medians = []
    effect_iqr_lo = []
    effect_iqr_hi = []
    n_per_bin = []

    for lo, hi in k_bins:
        sig_count = 0
        total = 0
        effects = []
        for nr in neuron_results:
            for pr in nr['partner_results']:
                if lo <= pr['k'] <= hi:
                    total += 1
                    if pr.get('bh_significant_k3', False):
                        sig_count += 1
                    er = pr.get('effect_ratio')
                    if er is not None and np.isfinite(er):
                        effects.append(er)
        sig_fracs.append(sig_count / total if total > 0 else 0)
        n_per_bin.append(total)
        if effects:
            effect_medians.append(np.median(effects))
            effect_iqr_lo.append(np.percentile(effects, 25))
            effect_iqr_hi.append(np.percentile(effects, 75))
        else:
            effect_medians.append(np.nan)
            effect_iqr_lo.append(np.nan)
            effect_iqr_hi.append(np.nan)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(DOUBLE_COL, 2.8))

    x = np.arange(len(bin_labels))
    ax1.bar(x, sig_fracs, color='#0072B2', alpha=0.8, width=0.6)
    ax1.axhline(0.05, color='k', linestyle='--', linewidth=0.5, alpha=0.5,
                label='Expected (5%)')
    for i, (frac, n) in enumerate(zip(sig_fracs, n_per_bin)):
        ax1.text(i, frac + 0.01, f'n={n}', ha='center', fontsize=6)
    ax1.set_xticks(x)
    ax1.set_xticklabels(bin_labels)
    ax1.set_xlabel('Partner synapse count (k)')
    ax1.set_ylabel('Fraction BH-significant')
    ax1.set_title('A. Clustering detection by k', fontsize=8, fontweight='bold')
    ax1.legend(frameon=False, fontsize=6)

    ax2.plot(x, effect_medians, 'o-', color='#D55E00', linewidth=1.5,
             markersize=6)
    for i in range(len(x)):
        if np.isfinite(effect_iqr_lo[i]):
            ax2.plot([x[i], x[i]], [effect_iqr_lo[i], effect_iqr_hi[i]],
                     '-', color='#D55E00', linewidth=1.5)
    ax2.axhline(1.0, color='k', linestyle='--', linewidth=0.5, alpha=0.5)
    ax2.set_xticks(x)
    ax2.set_xticklabels(bin_labels)
    ax2.set_xlabel('Partner synapse count (k)')
    ax2.set_ylabel('Median compactness ratio\n(observed / null distance)')
    ax2.set_title('B. Effect strength by k', fontsize=8, fontweight='bold')

    fig.suptitle('Partner count vs clustering strength', fontsize=9,
                 fontweight='bold')
    fig.tight_layout()
    if save_path:
        _save_fig(fig, save_path)
    return fig
