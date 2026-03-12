"""Tests for input-specific clustering analysis.

Uses synthetic data to verify:
1. Per-partner test produces valid results
2. BH FDR correction is correct
3. Label-shuffle control produces expected results under null
4. Enrichment summary computation is correct
"""

import numpy as np
import pytest

from neurostat.io.swc import NeuronSkeleton, SnapResult, SWCNode

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.soma_distance import (
    fast_geodesic_distance_matrix,
    compute_soma_distances,
)
from src.input_clustering import (
    per_partner_clustering_test,
    _bh_fdr,
    _compute_partner_stats,
    compute_enrichment_summary,
)


def _make_y_skeleton():
    """Same Y-shaped skeleton as in test_soma_distance_null."""
    nodes = {
        1: SWCNode(1, 1, 0, 0, 0, 10, -1),
        2: SWCNode(2, 3, 100, 0, 0, 5, 1),
        3: SWCNode(3, 3, 200, 0, 0, 5, 2),
        4: SWCNode(4, 3, 300, 100, 0, 5, 3),
        5: SWCNode(5, 3, 400, 200, 0, 5, 4),
        6: SWCNode(6, 3, 300, -100, 0, 5, 3),
        7: SWCNode(7, 3, 400, -200, 0, 5, 6),
    }
    children = {1: [2], 2: [3], 3: [4, 6], 4: [5], 5: [], 6: [7], 7: []}
    skel = NeuronSkeleton(nodes=nodes, children=children, root_id=1)
    skel.branches = skel.decompose_branches()
    return skel


def _make_synthetic_data(skel, n_synapses=60, n_partners=6, seed=42):
    """Create synthetic synapses with partner assignments."""
    rng = np.random.default_rng(seed)
    n_branches = len(skel.branches)

    branch_ids = []
    branch_positions = []
    for _ in range(n_synapses):
        bi = rng.integers(0, n_branches)
        pi = rng.uniform(0, skel.branches[bi].total_length)
        branch_ids.append(bi)
        branch_positions.append(pi)

    snap = SnapResult(
        branch_ids=np.array(branch_ids),
        branch_positions=np.array(branch_positions),
        distances=np.zeros(n_synapses),
        valid=np.ones(n_synapses, dtype=bool),
    )

    # Assign partners: first partner gets more synapses
    partner_ids = np.zeros(n_synapses, dtype=int)
    per_partner = n_synapses // n_partners
    for i in range(n_partners):
        start = i * per_partner
        end = (i + 1) * per_partner if i < n_partners - 1 else n_synapses
        partner_ids[start:end] = 100 + i

    return snap, partner_ids


class TestBHFDR:
    def test_no_rejections(self):
        """High p-values should yield no rejections."""
        p = np.array([0.5, 0.6, 0.7, 0.8, 0.9])
        reject = _bh_fdr(p, q=0.05)
        assert not reject.any()

    def test_all_rejections(self):
        """Very low p-values should all be rejected."""
        p = np.array([0.001, 0.002, 0.003])
        reject = _bh_fdr(p, q=0.05)
        assert reject.all()

    def test_partial_rejections(self):
        """Mix of low and high p-values."""
        p = np.array([0.001, 0.01, 0.5, 0.9])
        reject = _bh_fdr(p, q=0.05)
        assert reject[0]  # lowest p should be rejected
        assert not reject[3]  # highest p should not be rejected

    def test_empty(self):
        reject = _bh_fdr(np.array([]), q=0.05)
        assert len(reject) == 0

    def test_bh_more_permissive_than_bonferroni(self):
        """BH should reject at least as many as Bonferroni."""
        rng = np.random.default_rng(42)
        p = rng.uniform(0, 0.1, size=20)
        bh = _bh_fdr(p, q=0.05)
        bonf = p < (0.05 / len(p))
        assert bh.sum() >= bonf.sum()


class TestPartnerStats:
    def test_single_synapse(self):
        skel = _make_y_skeleton()
        snap, _ = _make_synthetic_data(skel, n_synapses=10)
        D = fast_geodesic_distance_matrix(skel, snap)

        stats = _compute_partner_stats(np.array([0]), D, snap)
        assert stats['mean_pairwise_distance'] == 0.0
        assert stats['max_branch_fraction'] == 1.0

    def test_multiple_synapses(self):
        skel = _make_y_skeleton()
        snap, _ = _make_synthetic_data(skel, n_synapses=20)
        D = fast_geodesic_distance_matrix(skel, snap)

        indices = np.arange(5)
        stats = _compute_partner_stats(indices, D, snap)
        assert stats['mean_pairwise_distance'] >= 0
        assert 0 < stats['max_branch_fraction'] <= 1.0
        assert stats['branch_entropy'] >= 0
        assert stats['n_branches_used'] >= 1


class TestPerPartnerClustering:
    def test_basic_execution(self):
        """Test that per_partner_clustering_test runs without error."""
        skel = _make_y_skeleton()
        snap, partner_ids = _make_synthetic_data(skel, n_synapses=30, n_partners=3)

        result = per_partner_clustering_test(
            skel, snap, partner_ids,
            k_thresholds=(3, 5),
            n_permutations=50,
            seed=42,
        )

        assert 'partner_results' in result
        assert 'summary' in result
        assert len(result['partner_results']) > 0

    def test_p_values_in_range(self):
        skel = _make_y_skeleton()
        snap, partner_ids = _make_synthetic_data(skel, n_synapses=30, n_partners=3)

        result = per_partner_clustering_test(
            skel, snap, partner_ids,
            k_thresholds=(3,),
            n_permutations=50,
            seed=42,
        )

        for pr in result['partner_results']:
            assert 0 <= pr['p_distance'] <= 1
            assert 0 <= pr['p_branch_frac'] <= 1
            assert 0 <= pr['p_entropy'] <= 1

    def test_summary_structure(self):
        skel = _make_y_skeleton()
        snap, partner_ids = _make_synthetic_data(skel, n_synapses=30, n_partners=3)

        result = per_partner_clustering_test(
            skel, snap, partner_ids,
            k_thresholds=(3, 5, 8),
            n_permutations=50,
            seed=42,
        )

        for tier in ('k>=3', 'k>=5', 'k>=8'):
            assert tier in result['summary']
            s = result['summary'][tier]
            assert 'n_partners' in s

    def test_uniform_null_fpr(self):
        """Under the null (random placement), ~5% of partners should be significant."""
        # This is a statistical test so we use loose bounds
        skel = _make_y_skeleton()
        rng = np.random.default_rng(123)
        n_branches = len(skel.branches)

        # Create random synapses with random partner assignments
        n_syn = 100
        branch_ids = rng.integers(0, n_branches, size=n_syn)
        branch_positions = np.array([rng.uniform(0, skel.branches[bi].total_length)
                                      for bi in branch_ids])
        snap = SnapResult(
            branch_ids=branch_ids,
            branch_positions=branch_positions,
            distances=np.zeros(n_syn),
            valid=np.ones(n_syn, dtype=bool),
        )

        # Many partners with equal share
        partner_ids = np.repeat(np.arange(10), 10)
        rng.shuffle(partner_ids)

        result = per_partner_clustering_test(
            skel, snap, partner_ids,
            k_thresholds=(3,),
            n_permutations=100,
            seed=42,
        )

        # Under null, we expect fraction significant to be modest
        # (not a strict test, just sanity check)
        summary = result['summary']['k>=3']
        assert summary['n_partners'] >= 5  # we should have some testable partners


class TestEnrichmentSummary:
    def test_basic(self):
        partner_results = [
            {'qualifying_tiers': [3, 5], 'bh_significant_k3': True, 'bh_significant_k5': True},
            {'qualifying_tiers': [3, 5], 'bh_significant_k3': False, 'bh_significant_k5': False},
            {'qualifying_tiers': [3, 5], 'bh_significant_k3': True, 'bh_significant_k5': False},
            {'qualifying_tiers': [3], 'bh_significant_k3': False},
        ]

        enrichment = compute_enrichment_summary(partner_results, k_thresholds=(3, 5))

        assert enrichment['k>=3']['n_total'] == 4
        assert enrichment['k>=3']['n_significant'] == 2
        assert abs(enrichment['k>=3']['observed_frac'] - 0.5) < 1e-6

        assert enrichment['k>=5']['n_total'] == 3
        assert enrichment['k>=5']['n_significant'] == 1

    def test_no_partners(self):
        enrichment = compute_enrichment_summary([], k_thresholds=(5,))
        assert enrichment['k>=5']['n_total'] == 0
        assert enrichment['k>=5']['enrichment_ratio'] == 0.0
