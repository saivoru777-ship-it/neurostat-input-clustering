# Reproducibility Manifest

## Code Version
- **Commit**: `4d651607b7556a808a9f03dabafcd77ce5e39e51`
- **Date**: 2026-03-12
- **Branch**: main

## Execution Sequence

All scripts run from the project root with:
```bash
cd ~/research/neurostat-input-clustering
export PYTHONPATH=~/research/neurostat:~/research/neurostat-input-clustering
```

1. `python3 scripts/01_extract_presynaptic_partners.py`  (~3 min)
2. `python3 scripts/02_soma_distance_null_analysis.py`   (~1-2 hr)
3. `python3 scripts/03_input_specific_clustering.py`     (~40 min)
4. `python3 scripts/04_generate_figures.py`              (~2 min)

## Random Seeds
- Script 02 (soma-distance null): `seed=42` for DendriteConstrainedNull, `seed=43` for SomaDistanceNull
- Script 03 (input clustering): `seed=42` for per-partner test, label-shuffle, and per-type test
- Permutation counts: 1000 (per-partner), 1000 label-shuffles × 200 permutations each
- Mock counts: 50 (null model comparison), 50 (per-type test)

## Dependencies
- Python 3.9.6
- numpy >= 1.24
- scipy >= 1.10
- pandas >= 2.0
- matplotlib >= 3.7
- neurostat (local, pip install -e ~/research/neurostat)

## Data Source
- MICrONS cortical connectome HDF5: `data/microns/microns_v343.h5` (symlink to ~/research/neurostat/data/microns/)
- 12 neurons defined in scripts, SWC + synapse CSV files in `data/microns/`

## Output File Checksums (MD5)

### Core Results
| File | MD5 |
|------|-----|
| `results/input_clustering_results.json` | `787ab5c145f540e1e4cc8aeb0b8e4f99` |
| `results/soma_distance_null_results.json` | `e2a7fecbf9d0566e202f98080964ca31` |
| `results/enrichment_summary.csv` | `3857535a444b4366ab53de8b0b8201a0` |
| `results/null_model_comparison.csv` | `3868930de8f32330ed144cafca1a4b84` |
| `results/presynaptic_summary.csv` | `abb2e1883e087ec68189ce1d9619c431` |

### Per-Neuron Extractions
| File | MD5 |
|------|-----|
| `results/exc_23P_presynaptic.csv` | `85f6e2bdae1f01616865593e8a7aa320` |
| `results/exc_23P_2_presynaptic.csv` | `fa3ccbebd3f55f24504321eb4476048a` |
| `results/exc_4P_presynaptic.csv` | `5908399dc3a8e4a459ee0457e84a1a64` |
| `results/exc_5PIT_presynaptic.csv` | `86f13d6636bddd28623726b69c77a8f2` |
| `results/exc_5PET_presynaptic.csv` | `95282b3d1e0aea42a5961d85ef45d07a` |
| `results/exc_6PCT_presynaptic.csv` | `bef7acd54e19e5b5551f6ff6ba8f9810` |
| `results/inh_BC_presynaptic.csv` | `762332e5c4dc984dd4d4c32745d3f257` |
| `results/inh_BC_2_presynaptic.csv` | `a469033bd4a7a6a6e025b66e50268fa5` |
| `results/inh_MC_presynaptic.csv` | `f09e455324c9dc7041d61bb2a6b7f719` |
| `results/inh_MC_2_presynaptic.csv` | `65d0a076a4c976152d017fe6afee6708` |
| `results/inh_BPC_presynaptic.csv` | `e5eb983925299e934d2adbb0c5757c3d` |
| `results/inh_BPC_2_presynaptic.csv` | `a5285c70628a41a887d74aa5badd8282` |
