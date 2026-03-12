"""Input-specific synapse clustering analysis.

Three levels of analysis:
  6a. Per-partner test (Hebbian clustering) with distance-matched permutation null
  6b. Label-shuffle control
  6c. Per-type multiscale test
  6d. Neuron-level enrichment summary

Distance-matched permutation null: For each partner's k synapses, the null
preserves the soma-distance distribution by binning ALL synapses by soma
distance and sampling k synapses matching the partner's distance bin profile.
"""

import numpy as np
from scipy import stats
from collections import Counter

from neurostat.io.swc import SnapResult
from neurostat.core.tree_statistics import compute_curves, ScaleRange
from neurostat.core.chi_squared import chi_squared_with_covariance

from .soma_distance import (
    compute_soma_distances,
    fast_geodesic_distance_matrix,
    precompute_branch_endpoint_distances,
)


# ── 6a. Per-partner clustering test ─────────────────────────────────────

def per_partner_clustering_test(
    skeleton, snap_result, pre_root_ids, soma_dists=None,
    geodesic_matrix=None, k_thresholds=(3, 5, 8), n_permutations=1000,
    n_distance_bins=10, seed=42,
):
    """Test whether synapses from each presynaptic partner are spatially clustered.

    Uses distance-matched permutation null: for each partner's k synapses,
    null samples preserve the soma-distance bin distribution.

    Parameters
    ----------
    skeleton : NeuronSkeleton
    snap_result : SnapResult
        Valid synapses only.
    pre_root_ids : ndarray
        Presynaptic root_id for each synapse (same length as snap_result).
    soma_dists : ndarray, optional
        Precomputed soma distances. Computed if None.
    geodesic_matrix : ndarray, optional
        Precomputed N×N geodesic distance matrix. Computed if None.
    k_thresholds : tuple
        Synapse count thresholds for tiered analysis.
    n_permutations : int
    n_distance_bins : int
        Number of equal-frequency bins for distance matching.
    seed : int

    Returns
    -------
    dict with keys:
        'partner_results': list of per-partner dicts
        'summary': dict with enrichment stats per k threshold
    """
    rng = np.random.default_rng(seed)
    n = len(snap_result.branch_ids)

    # Precompute if needed
    if soma_dists is None:
        soma_dists = compute_soma_distances(skeleton, snap_result)
    if geodesic_matrix is None:
        geodesic_matrix = fast_geodesic_distance_matrix(skeleton, snap_result)

    # Create distance bins (equal-frequency)
    bin_edges = np.percentile(soma_dists, np.linspace(0, 100, n_distance_bins + 1))
    bin_edges[0] -= 1  # ensure min is included
    bin_edges[-1] += 1
    syn_bins = np.digitize(soma_dists, bin_edges) - 1  # 0-indexed bins

    # Group synapse indices by distance bin for fast sampling
    bin_to_indices = {}
    for i in range(n):
        b = syn_bins[i]
        if b not in bin_to_indices:
            bin_to_indices[b] = []
        bin_to_indices[b].append(i)
    for b in bin_to_indices:
        bin_to_indices[b] = np.array(bin_to_indices[b])

    # Count synapses per partner
    partner_counts = Counter(pre_root_ids)
    all_indices = np.arange(n)

    partner_results = []

    for partner_id, count in partner_counts.items():
        partner_mask = pre_root_ids == partner_id
        partner_indices = np.where(partner_mask)[0]
        k = len(partner_indices)

        # Record which k thresholds this partner qualifies for
        qualifying_tiers = [t for t in k_thresholds if k >= t]
        if not qualifying_tiers:
            continue

        # Observed test statistics
        obs_stats = _compute_partner_stats(
            partner_indices, geodesic_matrix, snap_result
        )

        # Distance-matched permutation null
        partner_bin_profile = Counter(syn_bins[partner_indices])

        null_mean_dists = np.empty(n_permutations)
        null_branch_fracs = np.empty(n_permutations)
        null_entropies = np.empty(n_permutations)

        for perm in range(n_permutations):
            perm_indices = _sample_distance_matched(
                partner_bin_profile, bin_to_indices, k, rng
            )
            null_stats = _compute_partner_stats(
                perm_indices, geodesic_matrix, snap_result
            )
            null_mean_dists[perm] = null_stats['mean_pairwise_distance']
            null_branch_fracs[perm] = null_stats['max_branch_fraction']
            null_entropies[perm] = null_stats['branch_entropy']

        # P-values (one-sided)
        # Clustering = smaller mean distance
        p_distance = np.mean(null_mean_dists <= obs_stats['mean_pairwise_distance'])
        # Clustering = higher branch concentration (larger max_branch_fraction)
        p_branch_frac = np.mean(null_branch_fracs >= obs_stats['max_branch_fraction'])
        # Clustering = lower entropy (more concentrated)
        p_entropy = np.mean(null_entropies <= obs_stats['branch_entropy'])

        # Z-scores
        z_distance = _compute_zscore(
            obs_stats['mean_pairwise_distance'], null_mean_dists
        )
        z_branch_frac = _compute_zscore(
            obs_stats['max_branch_fraction'], null_branch_fracs
        )

        # Effect size: observed distance / null median distance
        null_median_dist = float(np.median(null_mean_dists))
        effect_ratio = (obs_stats['mean_pairwise_distance'] / null_median_dist
                        if null_median_dist > 0 else np.nan)

        partner_results.append({
            'partner_id': int(partner_id),
            'k': k,
            'qualifying_tiers': qualifying_tiers,
            'observed': obs_stats,
            'p_distance': float(p_distance),
            'p_branch_frac': float(p_branch_frac),
            'p_entropy': float(p_entropy),
            'z_distance': float(z_distance),
            'z_branch_frac': float(z_branch_frac),
            'null_mean_dist_mean': float(np.mean(null_mean_dists)),
            'null_mean_dist_std': float(np.std(null_mean_dists)),
            'null_mean_dist_median': null_median_dist,
            'effect_ratio': float(effect_ratio),
        })

    # Multiple testing correction per tier
    summary = {}
    for tier in k_thresholds:
        tier_results = [r for r in partner_results if tier in r['qualifying_tiers']]
        if not tier_results:
            summary[f'k>={tier}'] = {
                'n_partners': 0, 'n_significant_bh': 0, 'n_significant_bonf': 0,
            }
            continue

        p_vals = np.array([r['p_distance'] for r in tier_results])
        bh_reject = _bh_fdr(p_vals, q=0.05)
        bonf_reject = p_vals < (0.05 / len(p_vals))

        # Tag each result
        for i, r in enumerate(tier_results):
            r[f'bh_significant_k{tier}'] = bool(bh_reject[i])
            r[f'bonf_significant_k{tier}'] = bool(bonf_reject[i])

        # Effect size summary: median observed/null distance ratio for significant partners
        sig_effects = [r['effect_ratio'] for r, sig in zip(tier_results, bh_reject)
                       if sig and np.isfinite(r['effect_ratio'])]
        median_effect = float(np.median(sig_effects)) if sig_effects else np.nan

        # k distribution for this tier
        k_values = [r['k'] for r in tier_results]

        summary[f'k>={tier}'] = {
            'n_partners': len(tier_results),
            'n_significant_bh': int(bh_reject.sum()),
            'n_significant_bonf': int(bonf_reject.sum()),
            'frac_significant_bh': float(bh_reject.mean()),
            'frac_significant_bonf': float(bonf_reject.mean()),
            'enrichment_bh': float(bh_reject.mean() / 0.05) if bh_reject.mean() > 0 else 0.0,
            'median_effect_ratio': median_effect,
            'k_median': float(np.median(k_values)),
            'k_mean': float(np.mean(k_values)),
            'k_max': int(np.max(k_values)),
        }

    # Overall partner count distribution
    all_k = [r['k'] for r in partner_results]
    partner_count_dist = {
        'n_total_testable': len(partner_results),
        'k_values': sorted(all_k, reverse=True),
        'k_median': float(np.median(all_k)) if all_k else 0,
        'k_mean': float(np.mean(all_k)) if all_k else 0,
    }

    return {
        'partner_results': partner_results,
        'summary': summary,
        'partner_count_distribution': partner_count_dist,
    }


def _compute_partner_stats(indices, geodesic_matrix, snap_result):
    """Compute clustering statistics for a set of synapse indices."""
    k = len(indices)

    # Mean pairwise geodesic distance
    if k >= 2:
        sub_matrix = geodesic_matrix[np.ix_(indices, indices)]
        triu = sub_matrix[np.triu_indices(k, k=1)]
        mean_pw_dist = float(np.mean(triu))
    else:
        mean_pw_dist = 0.0

    # Branch concentration
    branch_ids = snap_result.branch_ids[indices]
    branch_counts = Counter(branch_ids)
    max_branch_frac = max(branch_counts.values()) / k if k > 0 else 0.0

    # Shannon entropy across branches
    if len(branch_counts) > 1:
        probs = np.array(list(branch_counts.values())) / k
        entropy = float(-np.sum(probs * np.log2(probs)))
    else:
        entropy = 0.0

    return {
        'mean_pairwise_distance': mean_pw_dist,
        'max_branch_fraction': max_branch_frac,
        'branch_entropy': entropy,
        'n_branches_used': len(branch_counts),
    }


def _sample_distance_matched(partner_bin_profile, bin_to_indices, k, rng):
    """Sample k synapses matching the partner's soma-distance bin profile."""
    sampled = []
    for bin_idx, count in partner_bin_profile.items():
        available = bin_to_indices.get(bin_idx, np.array([], dtype=int))
        if len(available) == 0:
            continue
        n_sample = min(count, len(available))
        chosen = rng.choice(available, size=n_sample, replace=False)
        sampled.extend(chosen)

    # If we couldn't match exactly (rare edge case), fill randomly
    if len(sampled) < k:
        all_indices = np.concatenate(list(bin_to_indices.values()))
        remaining = k - len(sampled)
        extra = rng.choice(all_indices, size=remaining, replace=False)
        sampled.extend(extra)

    return np.array(sampled[:k])


def _bh_fdr(p_values, q=0.05):
    """Benjamini-Hochberg FDR correction.

    Returns boolean array: True if rejected at FDR level q.
    """
    n = len(p_values)
    if n == 0:
        return np.array([], dtype=bool)

    sorted_idx = np.argsort(p_values)
    sorted_p = p_values[sorted_idx]

    # BH thresholds: (rank / n) * q
    thresholds = (np.arange(1, n + 1) / n) * q

    # Find largest k where p_(k) <= threshold_(k)
    reject = np.zeros(n, dtype=bool)
    max_k = -1
    for k in range(n):
        if sorted_p[k] <= thresholds[k]:
            max_k = k

    if max_k >= 0:
        reject[sorted_idx[:max_k + 1]] = True

    return reject


def _compute_zscore(observed, null_distribution):
    """Compute z-score of observed value relative to null distribution."""
    finite = null_distribution[np.isfinite(null_distribution)]
    if len(finite) < 2:
        return 0.0
    std = np.std(finite)
    if std < 1e-10 or not np.isfinite(std):
        return 0.0
    mean = np.mean(finite)
    if not np.isfinite(mean) or not np.isfinite(observed):
        return 0.0
    return float((observed - mean) / std)


# ── 6b. Label-shuffle control ───────────────────────────────────────────

def label_shuffle_control(
    skeleton, snap_result, pre_root_ids, geodesic_matrix=None,
    soma_dists=None, k_min=5, n_shuffles=1000, n_distance_bins=10,
    n_permutations_per_shuffle=200, seed=42,
):
    """Label-shuffle control: permute partner labels, re-run per-partner test.

    Tests whether specific partner-to-position assignments are non-random.
    If shuffled data yields similar clustering fractions, the result is spurious.

    Parameters
    ----------
    skeleton : NeuronSkeleton
    snap_result : SnapResult
    pre_root_ids : ndarray
    geodesic_matrix : ndarray, optional
    soma_dists : ndarray, optional
    k_min : int
        Minimum synapse count per partner to include.
    n_shuffles : int
    n_distance_bins : int
    n_permutations_per_shuffle : int
        Fewer permutations per shuffle (speed tradeoff).
    seed : int

    Returns
    -------
    dict with:
        'real_frac_significant': fraction of partners significant in real data
        'shuffled_fracs': array of fractions significant in each shuffle
        'p_value': fraction of shuffles with >= real fraction
    """
    rng = np.random.default_rng(seed)
    n = len(snap_result.branch_ids)

    if soma_dists is None:
        soma_dists = compute_soma_distances(skeleton, snap_result)
    if geodesic_matrix is None:
        geodesic_matrix = fast_geodesic_distance_matrix(skeleton, snap_result)

    # Real test
    real_result = per_partner_clustering_test(
        skeleton, snap_result, pre_root_ids,
        soma_dists=soma_dists, geodesic_matrix=geodesic_matrix,
        k_thresholds=(k_min,), n_permutations=n_permutations_per_shuffle,
        n_distance_bins=n_distance_bins, seed=seed,
    )
    real_summary = real_result['summary'].get(f'k>={k_min}', {})
    real_frac = real_summary.get('frac_significant_bh', 0.0)

    # Shuffled tests
    shuffled_fracs = np.empty(n_shuffles)
    for si in range(n_shuffles):
        shuffled_ids = rng.permutation(pre_root_ids)
        shuf_result = per_partner_clustering_test(
            skeleton, snap_result, shuffled_ids,
            soma_dists=soma_dists, geodesic_matrix=geodesic_matrix,
            k_thresholds=(k_min,), n_permutations=n_permutations_per_shuffle,
            n_distance_bins=n_distance_bins, seed=seed + si + 1,
        )
        shuf_summary = shuf_result['summary'].get(f'k>={k_min}', {})
        shuffled_fracs[si] = shuf_summary.get('frac_significant_bh', 0.0)

    p_value = float(np.mean(shuffled_fracs >= real_frac))

    return {
        'real_frac_significant': float(real_frac),
        'shuffled_fracs': shuffled_fracs,
        'shuffled_mean': float(np.mean(shuffled_fracs)),
        'shuffled_std': float(np.std(shuffled_fracs)),
        'p_value': p_value,
        'k_min': k_min,
        'n_shuffles': n_shuffles,
    }


# ── 6c. Per-type multiscale test ────────────────────────────────────────

def per_type_multiscale_test(
    skeleton, snap_result, pre_root_ids, pre_cell_types,
    null_model_class=None, n_mocks=50, min_synapses=10, seed=42,
):
    """Run full Neurostat multiscale analysis per broad presynaptic cell type.

    Parameters
    ----------
    skeleton : NeuronSkeleton
    snap_result : SnapResult
    pre_root_ids : ndarray
    pre_cell_types : ndarray
        Broad cell type for each synapse ('excitatory', 'inhibitory', etc.).
    null_model_class : class, optional
        Null model class with fit/generate_mocks interface.
        Default: DendriteConstrainedNull (uniform).
    n_mocks : int
    min_synapses : int
        Skip types with fewer synapses.
    seed : int

    Returns
    -------
    dict: type_name -> {curves, test_results, n_synapses}
    """
    from neurostat.core.null_models import DendriteConstrainedNull

    if null_model_class is None:
        null_model_class = DendriteConstrainedNull

    branches = skeleton.branches
    scales = ScaleRange.for_dendrite(branches)
    unique_types = np.unique(pre_cell_types)

    type_results = {}

    for ctype in unique_types:
        if ctype in ('unknown', 'other', ''):
            continue

        type_mask = pre_cell_types == ctype
        n_syn = type_mask.sum()

        if n_syn < min_synapses:
            continue

        # Create sub-SnapResult
        sub_snap = SnapResult(
            branch_ids=snap_result.branch_ids[type_mask],
            branch_positions=snap_result.branch_positions[type_mask],
            distances=snap_result.distances[type_mask],
            valid=np.ones(type_mask.sum(), dtype=bool),
        )

        # Compute real curves
        real_curves = compute_curves(sub_snap, branches, scales)
        n_valid_scales = len(real_curves['variance_values'])

        if n_valid_scales < 3:
            continue

        # Generate null mocks (uniform)
        null = null_model_class(seed=seed)
        if hasattr(null, 'fit'):
            if null_model_class == DendriteConstrainedNull:
                null.fit(skeleton)
            else:
                # SomaDistanceNull needs snap_result
                null.fit(skeleton, snap_result)

        mocks = null.generate_mocks(n_syn, n_mocks)
        mock_curves = [compute_curves(m, branches, scales) for m in mocks]

        # Chi-squared test (variance)
        mock_var = np.array([mc['variance_values'] for mc in mock_curves
                             if len(mc['variance_values']) == n_valid_scales])
        var_test = {'p_value': 1.0, 'chi_squared': 0.0}
        if len(mock_var) >= 2:
            var_test = chi_squared_with_covariance(
                real_curves['variance_values'], mock_var
            )

        # Chi-squared test (skewness)
        n_skew = len(real_curves['skewness_values'])
        mock_skew = np.array([mc['skewness_values'] for mc in mock_curves
                              if len(mc['skewness_values']) == n_skew])
        skew_test = {'p_value': 1.0, 'chi_squared': 0.0}
        if len(mock_skew) >= 2:
            skew_test = chi_squared_with_covariance(
                real_curves['skewness_values'], mock_skew
            )

        type_results[ctype] = {
            'n_synapses': int(n_syn),
            'curves': {
                'scales': real_curves['scales'].tolist(),
                'variance_values': real_curves['variance_values'].tolist(),
                'skewness_scales': real_curves['skewness_scales'].tolist(),
                'skewness_values': real_curves['skewness_values'].tolist(),
            },
            'variance_test': {
                'chi_squared': var_test.get('chi_squared', 0),
                'p_value': var_test['p_value'],
            },
            'skewness_test': {
                'chi_squared': skew_test.get('chi_squared', 0),
                'p_value': skew_test['p_value'],
            },
            'mock_envelope': {
                'variance_mean': mock_var.mean(axis=0).tolist() if len(mock_var) > 0 else [],
                'variance_p5': np.percentile(mock_var, 5, axis=0).tolist() if len(mock_var) > 0 else [],
                'variance_p95': np.percentile(mock_var, 95, axis=0).tolist() if len(mock_var) > 0 else [],
            },
        }

    return type_results


# ── 6d. Neuron-level enrichment summary ─────────────────────────────────

def compute_enrichment_summary(partner_results, k_thresholds=(3, 5, 8)):
    """Compute neuron-level enrichment: fraction of partners with significant clustering.

    Parameters
    ----------
    partner_results : list of dict
        From per_partner_clustering_test.
    k_thresholds : tuple

    Returns
    -------
    dict: tier -> {observed_frac, expected_frac, enrichment_ratio, binomial_p}
    """
    enrichment = {}

    for tier in k_thresholds:
        tier_key = f'k>={tier}'
        tier_partners = [r for r in partner_results if tier in r['qualifying_tiers']]
        n_total = len(tier_partners)

        if n_total == 0:
            enrichment[tier_key] = {
                'n_total': 0, 'n_significant': 0,
                'observed_frac': 0.0, 'expected_frac': 0.05,
                'enrichment_ratio': 0.0, 'binomial_p': 1.0,
            }
            continue

        bh_key = f'bh_significant_k{tier}'
        n_sig = sum(1 for r in tier_partners if r.get(bh_key, False))
        obs_frac = n_sig / n_total
        expected_frac = 0.05

        # Enrichment ratio
        enrichment_ratio = obs_frac / expected_frac if expected_frac > 0 else 0.0

        # Binomial test: is n_sig significantly > expected?
        binom_p = float(stats.binom_test(
            n_sig, n_total, expected_frac, alternative='greater'
        )) if hasattr(stats, 'binom_test') else float(
            stats.binomtest(n_sig, n_total, expected_frac, alternative='greater').pvalue
        )

        enrichment[tier_key] = {
            'n_total': n_total,
            'n_significant': n_sig,
            'observed_frac': float(obs_frac),
            'expected_frac': expected_frac,
            'enrichment_ratio': float(enrichment_ratio),
            'binomial_p': float(binom_p),
        }

    return enrichment
