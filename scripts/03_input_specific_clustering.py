"""Input-specific clustering analysis for all 12 MICrONS neurons.

Runs:
  - Per-partner clustering test (tiered k thresholds: 3, 5, 8)
  - Label-shuffle control (for 2-3 example neurons)
  - Per-type multiscale test (broad cell type categories)
  - Neuron-level enrichment summary

Usage:
    python scripts/03_input_specific_clustering.py
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

from src.soma_distance import (
    compute_soma_distances,
    fast_geodesic_distance_matrix,
    precompute_branch_endpoint_distances,
)
from src.input_clustering import (
    per_partner_clustering_test,
    label_shuffle_control,
    per_type_multiscale_test,
    compute_enrichment_summary,
)
from src.hdf5_extraction import classify_cell_type_broad

DATA_DIR = PROJECT_DIR / "data" / "microns"
RESULTS_DIR = PROJECT_DIR / "results"

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

# Run label-shuffle control on these neurons (computationally expensive)
LABEL_SHUFFLE_NEURONS = ["exc_23P", "inh_BC", "exc_5PET"]

N_PERMUTATIONS = 1000
N_LABEL_SHUFFLES = 1000
N_PERM_PER_SHUFFLE = 200


def load_neuron_with_partners(label, root_id):
    """Load skeleton, synapses, and presynaptic partner data."""
    swc_path = DATA_DIR / f"{label}_{root_id}.swc"
    syn_path = DATA_DIR / f"{label}_{root_id}_synapses.csv"
    partner_path = RESULTS_DIR / f"{label}_presynaptic.csv"

    if not swc_path.exists() or not syn_path.exists():
        print(f"  {label}: MISSING skeleton/synapse files, skipping")
        return None

    if not partner_path.exists():
        print(f"  {label}: MISSING partner file, run 01_extract first")
        return None

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        skeleton_raw = NeuronSkeleton.from_swc_file(str(swc_path), scale_factor=1000.0)

    dendrite_skel = skeleton_raw.filter_by_type([1, 3, 4])
    if len(dendrite_skel.branches) < 3:
        dendrite_skel = skeleton_raw

    # Load synapses and snap
    syn_df = pd.read_csv(syn_path)
    syn_coords_nm = syn_df[['x_um', 'y_um', 'z_um']].values * 1000.0
    snap = dendrite_skel.snap_points(syn_coords_nm, d_max=50000.0)

    valid = snap.valid
    n_valid = valid.sum()
    if n_valid < 20:
        print(f"  {label}: only {n_valid} valid synapses, skipping")
        return None

    snap_valid = SnapResult(
        branch_ids=snap.branch_ids[valid],
        branch_positions=snap.branch_positions[valid],
        distances=snap.distances[valid],
        valid=np.ones(n_valid, dtype=bool),
    )

    # Load partner data
    partner_df = pd.read_csv(partner_path)

    # Match partner data to snapped synapses
    # Partners are in nm coordinates from HDF5; synapses are in um from CSV
    # We need to match by position (both now in nm after scaling)
    # Strategy: use the synapse CSV coordinates (valid only) to index into partner_df
    # The partner extraction uses HDF5 coords which may differ slightly from CSV coords
    # So we use nearest-neighbor matching

    partner_coords_nm = partner_df[['x_nm', 'y_nm', 'z_nm']].values
    syn_coords_valid = syn_coords_nm[valid]

    # Match each valid synapse to nearest partner entry
    from scipy.spatial import cKDTree
    if len(partner_coords_nm) > 0 and len(syn_coords_valid) > 0:
        tree = cKDTree(partner_coords_nm)
        dists, indices = tree.query(syn_coords_valid)

        pre_root_ids = partner_df['pre_root_id'].values[indices]

        if 'pre_cell_type_broad' in partner_df.columns:
            pre_cell_types = partner_df['pre_cell_type_broad'].values[indices]
        elif 'pre_cell_type' in partner_df.columns:
            pre_cell_types = np.array([
                classify_cell_type_broad(ct)
                for ct in partner_df['pre_cell_type'].values[indices]
            ])
        else:
            pre_cell_types = np.full(n_valid, 'unknown')

        # Warn about poor matches
        median_match_dist = np.median(dists)
        if median_match_dist > 5000:  # > 5um
            print(f"  {label}: WARNING large matching distance "
                  f"(median={median_match_dist:.0f} nm)")
    else:
        print(f"  {label}: no partner coordinates to match")
        return None

    return dendrite_skel, snap_valid, pre_root_ids, pre_cell_types


def run_neuron_analysis(skeleton, snap, pre_root_ids, pre_cell_types,
                         label, mtype, do_label_shuffle=False):
    """Run full input-specific clustering analysis for one neuron."""
    n = len(snap.branch_ids)
    print(f"\n{'='*60}")
    print(f"  {label} ({mtype}): {n} synapses")

    # Precompute distance data
    print(f"    Precomputing geodesic distances...")
    _, node_to_idx, endpoint_dists = precompute_branch_endpoint_distances(skeleton)
    geodesic_matrix = fast_geodesic_distance_matrix(
        skeleton, snap, node_to_idx, endpoint_dists
    )
    soma_dists = compute_soma_distances(skeleton, snap)

    # === Per-partner clustering test ===
    print(f"    Running per-partner clustering test ({N_PERMUTATIONS} permutations)...")
    partner_result = per_partner_clustering_test(
        skeleton, snap, pre_root_ids,
        soma_dists=soma_dists,
        geodesic_matrix=geodesic_matrix,
        k_thresholds=(3, 5, 8),
        n_permutations=N_PERMUTATIONS,
        seed=42,
    )

    # Print partner count distribution
    pcd = partner_result.get('partner_count_distribution', {})
    print(f"    Testable partners: {pcd.get('n_total_testable', 0)}, "
          f"median k={pcd.get('k_median', 0):.0f}, mean k={pcd.get('k_mean', 0):.1f}")

    # Print summary with effect sizes
    for tier, stats in partner_result['summary'].items():
        effect = stats.get('median_effect_ratio', float('nan'))
        effect_str = f", effect={effect:.2f}" if np.isfinite(effect) else ""
        print(f"    {tier}: {stats['n_partners']} partners "
              f"(median k={stats.get('k_median', 0):.0f}, max k={stats.get('k_max', 0)}), "
              f"{stats.get('n_significant_bh', 0)}/{stats['n_partners']} BH-sig "
              f"({stats.get('frac_significant_bh', 0)*100:.1f}%), "
              f"enrichment={stats.get('enrichment_bh', 0):.1f}x{effect_str}")

    # === Enrichment summary ===
    enrichment = compute_enrichment_summary(partner_result['partner_results'])
    for tier, enr in enrichment.items():
        print(f"    Enrichment {tier}: {enr['enrichment_ratio']:.1f}x, "
              f"binomial p={enr['binomial_p']:.4f}")

    # === Label-shuffle control ===
    shuffle_result = None
    if do_label_shuffle:
        print(f"    Running label-shuffle control ({N_LABEL_SHUFFLES} shuffles)...")
        shuffle_result = label_shuffle_control(
            skeleton, snap, pre_root_ids,
            geodesic_matrix=geodesic_matrix,
            soma_dists=soma_dists,
            k_min=5,
            n_shuffles=N_LABEL_SHUFFLES,
            n_permutations_per_shuffle=N_PERM_PER_SHUFFLE,
            seed=42,
        )
        print(f"    Label-shuffle: real={shuffle_result['real_frac_significant']:.3f}, "
              f"shuffled={shuffle_result['shuffled_mean']:.3f} +/- "
              f"{shuffle_result['shuffled_std']:.3f}, "
              f"p={shuffle_result['p_value']:.4f}")

    # === Per-type multiscale test ===
    print(f"    Running per-type multiscale test...")
    type_results = per_type_multiscale_test(
        skeleton, snap, pre_root_ids, pre_cell_types,
        n_mocks=50, min_synapses=10, seed=42,
    )
    for ctype, tr in type_results.items():
        print(f"    Type '{ctype}': {tr['n_synapses']} synapses, "
              f"var_p={tr['variance_test']['p_value']:.4f}")

    # Compile result
    # Serialize partner_results (convert numpy types)
    serializable_partners = []
    for r in partner_result['partner_results']:
        sr = {k: v for k, v in r.items()}
        sr['partner_id'] = int(sr['partner_id'])
        sr['qualifying_tiers'] = [int(t) for t in sr['qualifying_tiers']]
        serializable_partners.append(sr)

    result = {
        'label': label,
        'mtype': mtype,
        'n_synapses': n,
        'partner_test': {
            'partner_results': serializable_partners,
            'summary': partner_result['summary'],
            'partner_count_distribution': partner_result.get('partner_count_distribution', {}),
        },
        'enrichment': enrichment,
        'type_results': type_results,
    }

    if shuffle_result is not None:
        result['label_shuffle'] = {
            'real_frac_significant': shuffle_result['real_frac_significant'],
            'shuffled_mean': shuffle_result['shuffled_mean'],
            'shuffled_std': shuffle_result['shuffled_std'],
            'p_value': shuffle_result['p_value'],
            'shuffled_fracs': shuffle_result['shuffled_fracs'].tolist(),
        }

    return result


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("INPUT-SPECIFIC CLUSTERING ANALYSIS")
    print("=" * 70)

    all_results = []
    enrichment_rows = []

    for label, root_id, etype, mtype in NEURONS:
        loaded = load_neuron_with_partners(label, root_id)
        if loaded is None:
            continue

        skeleton, snap, pre_root_ids, pre_cell_types = loaded
        do_shuffle = label in LABEL_SHUFFLE_NEURONS

        result = run_neuron_analysis(
            skeleton, snap, pre_root_ids, pre_cell_types,
            label, mtype, do_label_shuffle=do_shuffle,
        )
        all_results.append(result)

        # Enrichment summary row — include effect size from partner test
        pt_summary = result['partner_test']['summary']
        for tier_key, enr in result['enrichment'].items():
            pt_tier = pt_summary.get(tier_key, {})
            enrichment_rows.append({
                'label': label,
                'mtype': mtype,
                'tier': tier_key,
                'n_partners': enr['n_total'],
                'n_significant': enr['n_significant'],
                'observed_frac': enr['observed_frac'],
                'enrichment_ratio': enr['enrichment_ratio'],
                'binomial_p': enr['binomial_p'],
                'median_effect_ratio': pt_tier.get('median_effect_ratio', float('nan')),
                'k_median': pt_tier.get('k_median', 0),
                'k_max': pt_tier.get('k_max', 0),
            })

    # Save results
    results_path = RESULTS_DIR / "input_clustering_results.json"
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    # Save enrichment CSV
    enr_path = RESULTS_DIR / "enrichment_summary.csv"
    pd.DataFrame(enrichment_rows).to_csv(enr_path, index=False)

    # Summary table
    print(f"\n{'=' * 100}")
    print("INPUT CLUSTERING ENRICHMENT SUMMARY")
    print(f"{'=' * 100}")
    print(f"{'Neuron':<12} {'mtype':<6} "
          f"{'k>=3 enr':>9} {'k>=3 p':>8} "
          f"{'k>=5 enr':>9} {'k>=5 p':>8} "
          f"{'k>=8 enr':>9} {'k>=8 p':>8}")
    print("-" * 100)
    for r in all_results:
        enr = r['enrichment']
        e3 = enr.get('k>=3', {})
        e5 = enr.get('k>=5', {})
        e8 = enr.get('k>=8', {})
        print(f"{r['label']:<12} {r['mtype']:<6} "
              f"{e3.get('enrichment_ratio', 0):>8.1f}x {e3.get('binomial_p', 1):>8.4f} "
              f"{e5.get('enrichment_ratio', 0):>8.1f}x {e5.get('binomial_p', 1):>8.4f} "
              f"{e8.get('enrichment_ratio', 0):>8.1f}x {e8.get('binomial_p', 1):>8.4f}")

    print(f"\nResults: {results_path}")
    print(f"Enrichment: {enr_path}")


if __name__ == '__main__':
    main()
