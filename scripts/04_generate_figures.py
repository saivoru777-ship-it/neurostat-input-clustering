"""Generate all figures from pre-computed results.

Reads JSON results from scripts 02 and 03, generates publication figures.

Usage:
    python scripts/04_generate_figures.py
"""

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from neurostat.io.swc import NeuronSkeleton, SnapResult

from src.viz import (
    plot_soma_distance_profiles,
    plot_null_comparison,
    plot_partner_volcano,
    plot_enrichment_summary,
    plot_label_shuffle,
    plot_branch_concentration,
    plot_type_fingerprints,
    plot_dendrite_input_map,
    plot_compactness_cdf,
    plot_k_vs_clustering,
)
from src.hdf5_extraction import classify_cell_type_broad

DATA_DIR = PROJECT_DIR / "data" / "microns"
RESULTS_DIR = PROJECT_DIR / "results"
FIGURES_DIR = PROJECT_DIR / "figures"

NEURONS = [
    ("exc_23P",   864691135848859998, "excitatory", "23P"),
    ("exc_23P_2", 864691135866483845, "excitatory", "23P"),
    ("exc_4P",    864691135738528881, "excitatory", "4P"),
    ("exc_5PIT",  864691135256642223, "excitatory", "5P-IT"),
    ("exc_5PET",  864691135884866160, "excitatory", "5P-ET"),
    ("exc_6PCT",  864691135866795798, "excitatory", "6P-CT"),
    ("inh_BC",    864691135293026230, "inhibitory", "BC"),
    ("inh_BC_2",  864691135135829529, "inhibitory", "BC"),
    ("inh_MC",    864691135273485073, "inhibitory", "MC"),
    ("inh_MC_2",  864691136119505176, "inhibitory", "MC"),
    ("inh_BPC",   864691136923311076, "inhibitory", "BPC"),
    ("inh_BPC_2", 864691135715512858, "inhibitory", "BPC"),
]

# Subset for dendrite input maps (computationally expensive)
INPUT_MAP_NEURONS = ["exc_23P", "inh_BC", "exc_5PET"]


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("GENERATING FIGURES")
    print("=" * 70)

    # Load results
    soma_results_path = RESULTS_DIR / "soma_distance_null_results.json"
    cluster_results_path = RESULTS_DIR / "input_clustering_results.json"

    soma_results = []
    cluster_results = []

    if soma_results_path.exists():
        with open(soma_results_path) as f:
            soma_results = json.load(f)
        print(f"Loaded soma-distance results: {len(soma_results)} neurons")
    else:
        print("WARNING: soma_distance_null_results.json not found, skipping related figures")

    if cluster_results_path.exists():
        with open(cluster_results_path) as f:
            cluster_results = json.load(f)
        print(f"Loaded clustering results: {len(cluster_results)} neurons")
    else:
        print("WARNING: input_clustering_results.json not found, skipping related figures")

    # ── Figure 1: Soma distance profiles ──
    if soma_results:
        print("\n  Generating fig_soma_distance_profiles...")
        profiles = []
        for r in soma_results:
            profiles.append({
                'label': r['label'],
                'soma_dists': np.array(r['soma_distances']),
                'bin_centers': np.array(r['density_profile']['bin_centers']),
                'density': np.array(r['density_profile']['density']),
            })
        plot_soma_distance_profiles(
            profiles,
            save_path=FIGURES_DIR / 'fig_soma_distance_profiles.pdf',
        )

    # ── Figure 2: Null model comparison ──
    if soma_results:
        print("  Generating fig_null_comparison...")
        comparisons = []
        for r in soma_results:
            comparisons.append({
                'label': r['label'],
                'scales': r['curves']['scales'],
                'real_variance': r['curves']['variance_values'],
                'uniform_envelope': r['uniform_envelope'],
                'soma_envelope': r['soma_envelope'],
            })
        plot_null_comparison(
            comparisons,
            save_path=FIGURES_DIR / 'fig_null_comparison.pdf',
        )

    # ── Figure 3: Partner volcano plots ──
    if cluster_results:
        print("  Generating fig_partner_volcano...")
        volcanos = []
        for r in cluster_results:
            pr = r['partner_test']['partner_results']
            volcanos.append({
                'label': r['label'],
                'partner_results': pr,
            })
        plot_partner_volcano(
            volcanos,
            save_path=FIGURES_DIR / 'fig_partner_volcano.pdf',
        )

    # ── Figure 4: Enrichment summary ──
    if cluster_results:
        print("  Generating fig_enrichment_summary...")
        enr_data = []
        for r in cluster_results:
            enr_data.append({
                'label': r['label'],
                'neuron_type': r.get('neuron_type', ''),
                'enrichment': r['enrichment'],
            })
        plot_enrichment_summary(
            enr_data,
            save_path=FIGURES_DIR / 'fig_enrichment_summary.pdf',
        )

    # ── Figure 5: Label-shuffle control ──
    if cluster_results:
        shuffle_data = [r for r in cluster_results if 'label_shuffle' in r]
        if shuffle_data:
            print("  Generating fig_label_shuffle...")
            shuffle_plots = []
            for r in shuffle_data:
                ls = r['label_shuffle']
                shuffle_plots.append({
                    'label': r['label'],
                    'real_frac_significant': ls['real_frac_significant'],
                    'shuffled_fracs': np.array(ls['shuffled_fracs']),
                })
            plot_label_shuffle(
                shuffle_plots,
                save_path=FIGURES_DIR / 'fig_label_shuffle.pdf',
            )

    # ── Figure 6: Branch concentration ──
    if cluster_results:
        print("  Generating fig_branch_concentration...")
        conc_data = []
        for r in cluster_results:
            conc_data.append({
                'label': r['label'],
                'partner_results': r['partner_test']['partner_results'],
            })
        plot_branch_concentration(
            conc_data,
            save_path=FIGURES_DIR / 'fig_branch_concentration.pdf',
        )

    # ── Figure 7: Type fingerprints ──
    if cluster_results:
        print("  Generating fig_type_fingerprints...")
        type_data = []
        for r in cluster_results:
            type_data.append({
                'label': r['label'],
                'type_results': r.get('type_results', {}),
            })
        # Only include neurons that have type results
        type_data = [td for td in type_data if td['type_results']]
        if type_data:
            plot_type_fingerprints(
                type_data,
                save_path=FIGURES_DIR / 'fig_type_fingerprints.pdf',
            )

    # ── Figure 8: Dendrite input maps ──
    if cluster_results:
        print("  Generating fig_dendrite_input_map...")
        for label, root_id, etype, mtype in NEURONS:
            if label not in INPUT_MAP_NEURONS:
                continue

            swc_path = DATA_DIR / f"{label}_{root_id}.swc"
            syn_path = DATA_DIR / f"{label}_{root_id}_synapses.csv"
            partner_path = RESULTS_DIR / f"{label}_presynaptic.csv"

            if not all(p.exists() for p in [swc_path, syn_path, partner_path]):
                continue

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                skel = NeuronSkeleton.from_swc_file(str(swc_path), scale_factor=1000.0)
            skel = skel.filter_by_type([1, 3, 4])
            if len(skel.branches) < 3:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    skel = NeuronSkeleton.from_swc_file(str(swc_path), scale_factor=1000.0)

            syn_df = pd.read_csv(syn_path)
            syn_coords_nm = syn_df[['x_um', 'y_um', 'z_um']].values * 1000.0
            snap = skel.snap_points(syn_coords_nm, d_max=50000.0)
            valid = snap.valid

            snap_valid = SnapResult(
                branch_ids=snap.branch_ids[valid],
                branch_positions=snap.branch_positions[valid],
                distances=snap.distances[valid],
                valid=np.ones(valid.sum(), dtype=bool),
            )

            # Match partners
            partner_df = pd.read_csv(partner_path)
            from scipy.spatial import cKDTree
            partner_coords = partner_df[['x_nm', 'y_nm', 'z_nm']].values
            tree = cKDTree(partner_coords)
            _, indices = tree.query(syn_coords_nm[valid])
            pre_ids = partner_df['pre_root_id'].values[indices]

            plot_dendrite_input_map(
                skel, snap_valid, pre_ids,
                top_n_partners=8,
                title=f'{label} ({mtype})',
                save_path=FIGURES_DIR / f'fig_dendrite_input_map_{label}.pdf',
            )

    # ── Figure 9: Compactness CDF (the "killer figure") ──
    if cluster_results:
        print("  Generating fig_compactness_cdf...")
        cdf_data = []
        for r in cluster_results:
            cdf_data.append({
                'label': r['label'],
                'mtype': r['mtype'],
                'partner_results': r['partner_test']['partner_results'],
            })
        plot_compactness_cdf(
            cdf_data,
            save_path=FIGURES_DIR / 'fig_compactness_cdf.pdf',
        )

    # ── Figure 10: k vs clustering strength ──
    if cluster_results:
        print("  Generating fig_k_vs_clustering...")
        k_data = []
        for r in cluster_results:
            k_data.append({
                'label': r['label'],
                'partner_results': r['partner_test']['partner_results'],
            })
        plot_k_vs_clustering(
            k_data,
            save_path=FIGURES_DIR / 'fig_k_vs_clustering.pdf',
        )

    # ── Cross-neuron summary statistics ──
    if cluster_results:
        print("\n" + "=" * 70)
        print("CROSS-NEURON SUMMARY STATISTICS")
        print("=" * 70)

        # Collect all effect ratios (k>=5 partners)
        all_effects = []
        exc_effects = []
        inh_effects = []
        bpc_effects = []
        neuron_medians = []

        for r in cluster_results:
            label = r['label']
            mtype = r['mtype']
            effects = []
            for pr in r['partner_test']['partner_results']:
                er = pr.get('effect_ratio')
                if er is not None and np.isfinite(er) and pr['k'] >= 5:
                    effects.append(er)
                    all_effects.append(er)
                    if mtype == 'BPC':
                        bpc_effects.append(er)
                    elif label.startswith('exc_'):
                        exc_effects.append(er)
                    else:
                        inh_effects.append(er)
            if effects:
                neuron_medians.append((label, mtype, np.median(effects),
                                       len(effects)))

        all_effects = np.array(all_effects)
        print(f"\n  Overall (k>=5 partners, all neurons):")
        print(f"    N partners: {len(all_effects)}")
        print(f"    Median compactness ratio: {np.median(all_effects):.3f}")
        print(f"    IQR: [{np.percentile(all_effects, 25):.3f}, "
              f"{np.percentile(all_effects, 75):.3f}]")
        pct_below_1 = np.mean(all_effects < 1.0) * 100
        print(f"    Partners more compact than null: {pct_below_1:.1f}%")
        median_reduction = (1 - np.median(all_effects)) * 100
        print(f"    Median distance reduction: {median_reduction:.1f}%")

        print(f"\n  By postsynaptic neuron type:")
        for pool, name in [(exc_effects, 'Excitatory'),
                            (inh_effects, 'Inhibitory (non-BPC)'),
                            (bpc_effects, 'BPC')]:
            pool = np.array(pool)
            if len(pool) > 0:
                print(f"    {name}: median={np.median(pool):.3f}, "
                      f"IQR=[{np.percentile(pool, 25):.3f}, "
                      f"{np.percentile(pool, 75):.3f}], "
                      f"n={len(pool)}")

        print(f"\n  Per-neuron median compactness ratio (k>=5):")
        print(f"    {'Neuron':<12} {'mtype':<6} {'median':>8} {'n_partners':>10}")
        print(f"    {'-'*40}")
        for label, mtype, med, n in neuron_medians:
            print(f"    {label:<12} {mtype:<6} {med:>8.3f} {n:>10}")

        # k vs clustering strength summary
        print(f"\n  Clustering by partner synapse count:")
        k_bins = [(3, 4), (5, 7), (8, 12), (13, 999)]
        bin_labels = ['3-4', '5-7', '8-12', '13+']
        print(f"    {'k range':<8} {'n':>5} {'frac_sig':>10} "
              f"{'median_ratio':>13} {'dist_reduction':>15}")
        for (lo, hi), bl in zip(k_bins, bin_labels):
            effects = []
            sig = 0
            total = 0
            for r in cluster_results:
                for pr in r['partner_test']['partner_results']:
                    if lo <= pr['k'] <= hi:
                        total += 1
                        if pr.get('bh_significant_k3', False):
                            sig += 1
                        er = pr.get('effect_ratio')
                        if er is not None and np.isfinite(er):
                            effects.append(er)
            frac = sig / total if total > 0 else 0
            med = np.median(effects) if effects else float('nan')
            red = (1 - med) * 100 if effects else float('nan')
            print(f"    {bl:<8} {total:>5} {frac:>10.3f} "
                  f"{med:>13.3f} {red:>14.1f}%")

    print(f"\nAll figures saved to: {FIGURES_DIR}/")


if __name__ == '__main__':
    main()
