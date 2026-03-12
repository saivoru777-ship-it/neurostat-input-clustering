"""Extract presynaptic partner data from MICrONS HDF5 for all 12 neurons.

Outputs per-neuron CSV files with columns:
    x_nm, y_nm, z_nm, pre_root_id, pre_cell_type, pre_vertex_idx

Also prints a summary table: total synapses, unique partners, partners
with >=3 synapses, cell type distribution.

Usage:
    python scripts/01_extract_presynaptic_partners.py
"""

import sys
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd

# Project paths
PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from src.hdf5_extraction import (
    load_vertex_properties,
    extract_presynaptic_partners_by_root_id,
    classify_cell_type_broad,
)

DATA_DIR = PROJECT_DIR / "data" / "microns"
RESULTS_DIR = PROJECT_DIR / "results"
HDF5_PATH = DATA_DIR / "microns_mm3_connectome_v1181.h5"

# Same 12 neurons as neurostat pipeline
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


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if not HDF5_PATH.exists():
        print(f"ERROR: HDF5 file not found: {HDF5_PATH}")
        print("Run download_microns_real_data.py in neurostat first.")
        sys.exit(1)

    print("=" * 70)
    print("PRESYNAPTIC PARTNER EXTRACTION")
    print("=" * 70)
    print(f"HDF5: {HDF5_PATH}")
    print(f"Output: {RESULTS_DIR}/")

    # Load vertex properties once
    print("\nLoading vertex properties...")
    vertex_df = load_vertex_properties(str(HDF5_PATH))
    print(f"  {len(vertex_df)} vertices loaded")

    all_summaries = []

    for label, root_id, etype, mtype in NEURONS:
        print(f"\n--- {label} (root_id={root_id}) ---")

        # Extract presynaptic partners
        partners_df = extract_presynaptic_partners_by_root_id(
            str(HDF5_PATH), root_id, vertex_df
        )

        if len(partners_df) == 0:
            print(f"  No synapses found, skipping")
            all_summaries.append({
                'label': label, 'mtype': mtype, 'n_synapses': 0,
                'n_partners': 0, 'n_partners_k3': 0, 'n_partners_k5': 0,
                'n_partners_k8': 0,
            })
            continue

        # Add broad cell type classification
        partners_df['pre_cell_type_broad'] = partners_df['pre_cell_type'].apply(
            classify_cell_type_broad
        )

        # Save CSV
        csv_path = RESULTS_DIR / f"{label}_presynaptic.csv"
        partners_df.to_csv(csv_path, index=False)

        # Summary statistics
        n_synapses = len(partners_df)
        partner_counts = Counter(partners_df['pre_root_id'])
        n_partners = len(partner_counts)
        n_k3 = sum(1 for c in partner_counts.values() if c >= 3)
        n_k5 = sum(1 for c in partner_counts.values() if c >= 5)
        n_k8 = sum(1 for c in partner_counts.values() if c >= 8)

        type_dist = Counter(partners_df['pre_cell_type_broad'])

        print(f"  {n_synapses} synapses, {n_partners} unique partners")
        print(f"  Partners with k>=3: {n_k3}, k>=5: {n_k5}, k>=8: {n_k8}")
        print(f"  Cell type distribution: {dict(type_dist)}")

        all_summaries.append({
            'label': label,
            'mtype': mtype,
            'n_synapses': n_synapses,
            'n_partners': n_partners,
            'n_partners_k3': n_k3,
            'n_partners_k5': n_k5,
            'n_partners_k8': n_k8,
            'n_exc': type_dist.get('excitatory', 0),
            'n_inh': type_dist.get('inhibitory', 0),
            'n_other': type_dist.get('other', 0) + type_dist.get('unknown', 0),
        })

    # Summary table
    print(f"\n{'=' * 90}")
    print("PRESYNAPTIC PARTNER SUMMARY")
    print(f"{'=' * 90}")
    print(f"{'Neuron':<12} {'mtype':<6} {'syn':>6} {'partners':>9} "
          f"{'k>=3':>5} {'k>=5':>5} {'k>=8':>5} "
          f"{'exc':>5} {'inh':>5} {'oth':>5}")
    print("-" * 90)
    for s in all_summaries:
        print(f"{s['label']:<12} {s['mtype']:<6} {s['n_synapses']:>6} "
              f"{s['n_partners']:>9} "
              f"{s.get('n_partners_k3', 0):>5} "
              f"{s.get('n_partners_k5', 0):>5} "
              f"{s.get('n_partners_k8', 0):>5} "
              f"{s.get('n_exc', 0):>5} "
              f"{s.get('n_inh', 0):>5} "
              f"{s.get('n_other', 0):>5}")

    # Save summary CSV
    summary_df = pd.DataFrame(all_summaries)
    summary_path = RESULTS_DIR / "presynaptic_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSummary saved: {summary_path}")


if __name__ == '__main__':
    main()
