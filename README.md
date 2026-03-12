# Input-Specific Synapse Clustering Analysis

Extends the [neurostat](../neurostat/) framework with:

1. **Soma-distance null model** — accounts for distance-dependent synapse density gradients
2. **Input-specific clustering** — tests whether synapses from the same presynaptic partner are spatially clustered (Hebbian clustering prediction)

## Setup

```bash
pip install -e ~/research/neurostat/
pip install -r requirements.txt
```

## Pipeline

```bash
python scripts/01_extract_presynaptic_partners.py   # ~3 min
python scripts/02_soma_distance_null_analysis.py     # ~1-2 hr
python scripts/03_input_specific_clustering.py       # ~30-60 min
python scripts/04_generate_figures.py                # ~5 min
```

## Data

MICrONS mm3 connectome (12 proofread neurons) — symlinked from `../neurostat/data/microns/`.
