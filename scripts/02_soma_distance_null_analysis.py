"""Soma-distance null analysis: compare uniform vs soma-distance null models.

For each of the 12 neurons:
1. Load skeleton + synapses
2. Compute soma distances, fit distance-density profile
3. Fit SomaDistanceNull, generate 50 mocks
4. Also fit DendriteConstrainedNull (uniform), generate 50 mocks
5. Run chi_squared_with_covariance against BOTH null models
6. Save results JSON + per-neuron comparison

Key output: p-values under both nulls. If significance persists under the
soma-distance null, the structure is not merely a distance gradient artifact.

Usage:
    python scripts/02_soma_distance_null_analysis.py
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
from neurostat.core.null_models import DendriteConstrainedNull
from neurostat.core.tree_statistics import compute_curves, ScaleRange
from neurostat.core.chi_squared import chi_squared_with_covariance

from src.soma_distance import compute_soma_distances
from src.soma_distance_null import SomaDistanceNull

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

N_MOCKS = 50


def load_neuron(label, root_id):
    """Load skeleton and synapses, return (skeleton, snap_valid) or None."""
    swc_path = DATA_DIR / f"{label}_{root_id}.swc"
    syn_path = DATA_DIR / f"{label}_{root_id}_synapses.csv"

    if not swc_path.exists() or not syn_path.exists():
        print(f"  {label}: MISSING files, skipping")
        return None

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        skeleton_raw = NeuronSkeleton.from_swc_file(str(swc_path), scale_factor=1000.0)

    dendrite_skel = skeleton_raw.filter_by_type([1, 3, 4])
    if len(dendrite_skel.branches) < 3:
        dendrite_skel = skeleton_raw

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
    return dendrite_skel, snap_valid


def run_comparison(skeleton, snap, label, mtype):
    """Run dual null model comparison for one neuron."""
    branches = skeleton.branches
    n_synapses = len(snap.branch_ids)
    scales = ScaleRange.for_dendrite(branches)

    print(f"\n  {label} ({mtype}): {n_synapses} synapses, {len(branches)} branches")

    # Real curves
    real_curves = compute_curves(snap, branches, scales)
    n_var_scales = len(real_curves['variance_values'])

    # Soma distances + density profile
    soma_dists = compute_soma_distances(skeleton, snap)
    print(f"    Soma distance: median={np.median(soma_dists)/1000:.1f} um, "
          f"max={soma_dists.max()/1000:.1f} um")

    # === Uniform null ===
    print(f"    Generating {N_MOCKS} uniform null mocks...")
    uniform_null = DendriteConstrainedNull(seed=0)
    uniform_null.fit(skeleton)
    uniform_mocks = uniform_null.generate_mocks(n_synapses, N_MOCKS)
    uniform_curves = [compute_curves(m, branches, scales) for m in uniform_mocks]

    uniform_var = np.array([mc['variance_values'] for mc in uniform_curves
                            if len(mc['variance_values']) == n_var_scales])
    uniform_var_test = {'p_value': 1.0, 'chi_squared': 0.0}
    if len(uniform_var) >= 2:
        uniform_var_test = chi_squared_with_covariance(
            real_curves['variance_values'], uniform_var
        )

    n_skew = len(real_curves['skewness_values'])
    uniform_skew = np.array([mc['skewness_values'] for mc in uniform_curves
                             if len(mc['skewness_values']) == n_skew])
    uniform_skew_test = {'p_value': 1.0, 'chi_squared': 0.0}
    if len(uniform_skew) >= 2:
        uniform_skew_test = chi_squared_with_covariance(
            real_curves['skewness_values'], uniform_skew
        )

    # === Soma-distance null ===
    print(f"    Generating {N_MOCKS} soma-distance null mocks...")
    soma_null = SomaDistanceNull(seed=0)
    soma_null.fit(skeleton, snap)
    soma_mocks = soma_null.generate_mocks(n_synapses, N_MOCKS)
    soma_curves = [compute_curves(m, branches, scales) for m in soma_mocks]

    soma_var = np.array([mc['variance_values'] for mc in soma_curves
                         if len(mc['variance_values']) == n_var_scales])
    soma_var_test = {'p_value': 1.0, 'chi_squared': 0.0}
    if len(soma_var) >= 2:
        soma_var_test = chi_squared_with_covariance(
            real_curves['variance_values'], soma_var
        )

    soma_skew = np.array([mc['skewness_values'] for mc in soma_curves
                          if len(mc['skewness_values']) == n_skew])
    soma_skew_test = {'p_value': 1.0, 'chi_squared': 0.0}
    if len(soma_skew) >= 2:
        soma_skew_test = chi_squared_with_covariance(
            real_curves['skewness_values'], soma_skew
        )

    print(f"    Uniform null:      var_p={uniform_var_test['p_value']:.4f}, "
          f"skew_p={uniform_skew_test['p_value']:.4f}")
    print(f"    Soma-dist null:    var_p={soma_var_test['p_value']:.4f}, "
          f"skew_p={soma_skew_test['p_value']:.4f}")

    # Density profile for plotting
    dp = soma_null.density_profile

    result = {
        'label': label,
        'mtype': mtype,
        'n_synapses': n_synapses,
        'n_branches': len(branches),
        'soma_distance_stats': {
            'median_um': float(np.median(soma_dists) / 1000),
            'mean_um': float(np.mean(soma_dists) / 1000),
            'max_um': float(soma_dists.max() / 1000),
        },
        'uniform_null': {
            'variance_test': {
                'chi_squared': uniform_var_test.get('chi_squared', 0),
                'p_value': uniform_var_test['p_value'],
            },
            'skewness_test': {
                'chi_squared': uniform_skew_test.get('chi_squared', 0),
                'p_value': uniform_skew_test['p_value'],
            },
        },
        'soma_distance_null': {
            'variance_test': {
                'chi_squared': soma_var_test.get('chi_squared', 0),
                'p_value': soma_var_test['p_value'],
            },
            'skewness_test': {
                'chi_squared': soma_skew_test.get('chi_squared', 0),
                'p_value': soma_skew_test['p_value'],
            },
        },
        'curves': {
            'scales': real_curves['scales'].tolist(),
            'variance_values': real_curves['variance_values'].tolist(),
        },
        'uniform_envelope': {
            'mean': uniform_var.mean(axis=0).tolist() if len(uniform_var) > 0 else [],
            'p5': np.percentile(uniform_var, 5, axis=0).tolist() if len(uniform_var) > 0 else [],
            'p95': np.percentile(uniform_var, 95, axis=0).tolist() if len(uniform_var) > 0 else [],
        },
        'soma_envelope': {
            'mean': soma_var.mean(axis=0).tolist() if len(soma_var) > 0 else [],
            'p5': np.percentile(soma_var, 5, axis=0).tolist() if len(soma_var) > 0 else [],
            'p95': np.percentile(soma_var, 95, axis=0).tolist() if len(soma_var) > 0 else [],
        },
        'density_profile': {
            'bin_centers': dp['bin_centers'].tolist() if dp else [],
            'density': dp['density'].tolist() if dp else [],
        },
        'soma_distances': soma_dists.tolist(),
    }
    return result


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("SOMA-DISTANCE NULL MODEL ANALYSIS")
    print("=" * 70)

    all_results = []
    for label, root_id, etype, mtype in NEURONS:
        loaded = load_neuron(label, root_id)
        if loaded is None:
            continue
        skeleton, snap = loaded
        result = run_comparison(skeleton, snap, label, mtype)
        all_results.append(result)

    # Save results
    results_path = RESULTS_DIR / "soma_distance_null_results.json"
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    # Summary table
    print(f"\n{'=' * 90}")
    print("NULL MODEL COMPARISON SUMMARY")
    print(f"{'=' * 90}")
    print(f"{'Neuron':<12} {'mtype':<6} {'syn':>5} "
          f"{'uni_var_p':>10} {'soma_var_p':>11} "
          f"{'uni_skew_p':>11} {'soma_skew_p':>12}")
    print("-" * 90)
    for r in all_results:
        print(f"{r['label']:<12} {r['mtype']:<6} {r['n_synapses']:>5} "
              f"{r['uniform_null']['variance_test']['p_value']:>10.4f} "
              f"{r['soma_distance_null']['variance_test']['p_value']:>11.4f} "
              f"{r['uniform_null']['skewness_test']['p_value']:>11.4f} "
              f"{r['soma_distance_null']['skewness_test']['p_value']:>12.4f}")

    # Save comparison CSV
    rows = []
    for r in all_results:
        rows.append({
            'label': r['label'],
            'mtype': r['mtype'],
            'n_synapses': r['n_synapses'],
            'uniform_var_p': r['uniform_null']['variance_test']['p_value'],
            'soma_var_p': r['soma_distance_null']['variance_test']['p_value'],
            'uniform_skew_p': r['uniform_null']['skewness_test']['p_value'],
            'soma_skew_p': r['soma_distance_null']['skewness_test']['p_value'],
            'median_soma_dist_um': r['soma_distance_stats']['median_um'],
        })
    csv_path = RESULTS_DIR / "null_model_comparison.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    print(f"\nResults: {results_path}")
    print(f"Summary: {csv_path}")


if __name__ == '__main__':
    main()
