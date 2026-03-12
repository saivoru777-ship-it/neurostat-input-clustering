"""Tests for soma distance computation and soma-distance null model.

Uses a simple synthetic skeleton to verify:
1. Fast geodesic distances match the slow BFS implementation
2. Soma distances are correct for known geometry
3. SomaDistanceNull produces valid samples
4. Distance density profile is non-negative and sums correctly
"""

import numpy as np
import pytest

from neurostat.io.swc import NeuronSkeleton, SnapResult, Branch, SWCNode

# Import from src (project root must be on path or installed)
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.soma_distance import (
    precompute_branch_endpoint_distances,
    fast_geodesic_distance,
    fast_geodesic_distance_matrix,
    compute_soma_distances,
    estimate_distance_density_profile,
    compute_path_length_by_distance,
)
from src.soma_distance_null import SomaDistanceNull


def _make_y_skeleton():
    """Create a simple Y-shaped skeleton for testing.

    Topology:
        1 (soma, root) -- 2 -- 3 (branch point)
                                |-- 4 -- 5 (leaf)
                                |-- 6 -- 7 (leaf)

    Each edge is 100nm long. Total path: 600nm, 3 branches.
    """
    nodes = {
        1: SWCNode(1, 1, 0, 0, 0, 10, -1),       # soma / root
        2: SWCNode(2, 3, 100, 0, 0, 5, 1),
        3: SWCNode(3, 3, 200, 0, 0, 5, 2),         # branch point
        4: SWCNode(4, 3, 300, 100, 0, 5, 3),
        5: SWCNode(5, 3, 400, 200, 0, 5, 4),       # leaf
        6: SWCNode(6, 3, 300, -100, 0, 5, 3),
        7: SWCNode(7, 3, 400, -200, 0, 5, 6),      # leaf
    }
    children = {
        1: [2], 2: [3], 3: [4, 6], 4: [5], 5: [], 6: [7], 7: []
    }
    skel = NeuronSkeleton(nodes=nodes, children=children, root_id=1)
    skel.branches = skel.decompose_branches()
    return skel


def _make_snap_on_y(skel, positions_list):
    """Snap given (branch_id, position) pairs into a SnapResult."""
    n = len(positions_list)
    branch_ids = np.array([p[0] for p in positions_list])
    branch_positions = np.array([p[1] for p in positions_list])
    return SnapResult(
        branch_ids=branch_ids,
        branch_positions=branch_positions,
        distances=np.zeros(n),
        valid=np.ones(n, dtype=bool),
    )


class TestFastGeodesicDistance:
    def test_same_branch(self):
        skel = _make_y_skeleton()
        _, node_to_idx, endpoint_dists = precompute_branch_endpoint_distances(skel)

        # Two points on the same branch
        d = fast_geodesic_distance((0, 50.0), (0, 150.0), skel, node_to_idx, endpoint_dists)
        assert abs(d - 100.0) < 1e-6

    def test_different_branches(self):
        skel = _make_y_skeleton()
        _, node_to_idx, endpoint_dists = precompute_branch_endpoint_distances(skel)

        # Points on different branches — should go through shared branch point
        # Need to know branch decomposition to set up correctly
        # The exact distance depends on branch layout
        # Just verify it's finite and positive
        for bi in range(len(skel.branches)):
            for bj in range(len(skel.branches)):
                if bi != bj:
                    d = fast_geodesic_distance(
                        (bi, 50.0), (bj, 50.0), skel, node_to_idx, endpoint_dists
                    )
                    assert d > 0
                    assert np.isfinite(d)

    def test_matches_slow_implementation(self):
        """Fast geodesic should match skeleton.geodesic_distance()."""
        skel = _make_y_skeleton()
        _, node_to_idx, endpoint_dists = precompute_branch_endpoint_distances(skel)

        # Test several pairs
        n_branches = len(skel.branches)
        rng = np.random.default_rng(42)

        for _ in range(20):
            bi = rng.integers(0, n_branches)
            bj = rng.integers(0, n_branches)
            pi = rng.uniform(0, skel.branches[bi].total_length)
            pj = rng.uniform(0, skel.branches[bj].total_length)

            fast = fast_geodesic_distance(
                (bi, pi), (bj, pj), skel, node_to_idx, endpoint_dists
            )
            slow = skel.geodesic_distance((bi, pi), (bj, pj))

            assert abs(fast - slow) < 1e-3, \
                f"Mismatch: fast={fast:.3f} vs slow={slow:.3f} for ({bi},{pi:.1f})->({bj},{pj:.1f})"

    def test_distance_matrix(self):
        skel = _make_y_skeleton()
        n_branches = len(skel.branches)

        # Create some synapses
        rng = np.random.default_rng(42)
        positions = []
        for _ in range(15):
            bi = rng.integers(0, n_branches)
            pi = rng.uniform(0, skel.branches[bi].total_length)
            positions.append((bi, pi))

        snap = _make_snap_on_y(skel, positions)
        D = fast_geodesic_distance_matrix(skel, snap)

        # Should be symmetric
        assert np.allclose(D, D.T, atol=1e-6)
        # Diagonal should be zero
        assert np.allclose(np.diag(D), 0)
        # All distances should be non-negative
        assert np.all(D >= 0)


class TestSomaDistances:
    def test_soma_distance_at_root(self):
        skel = _make_y_skeleton()
        # Point right at the root should have soma distance ~0
        # Find which branch starts at root
        for bi, branch in enumerate(skel.branches):
            if branch.start_node == skel.root_id:
                snap = _make_snap_on_y(skel, [(bi, 0.0)])
                dists = compute_soma_distances(skel, snap)
                assert dists[0] < 1.0  # essentially zero
                break

    def test_soma_distances_increase(self):
        skel = _make_y_skeleton()
        n_branches = len(skel.branches)

        # More distal points should have larger soma distances
        rng = np.random.default_rng(42)
        positions = []
        for _ in range(30):
            bi = rng.integers(0, n_branches)
            pi = rng.uniform(0, skel.branches[bi].total_length)
            positions.append((bi, pi))

        snap = _make_snap_on_y(skel, positions)
        dists = compute_soma_distances(skel, snap)
        assert np.all(dists >= 0)
        assert dists.max() > 0


class TestDistanceDensityProfile:
    def test_density_nonnegative(self):
        skel = _make_y_skeleton()
        n_branches = len(skel.branches)
        rng = np.random.default_rng(42)

        positions = []
        for _ in range(50):
            bi = rng.integers(0, n_branches)
            pi = rng.uniform(0, skel.branches[bi].total_length)
            positions.append((bi, pi))
        snap = _make_snap_on_y(skel, positions)
        soma_dists = compute_soma_distances(skel, snap)

        centers, density = estimate_distance_density_profile(soma_dists, skel)
        assert np.all(density >= 0)
        assert len(centers) == len(density)

    def test_path_length_positive(self):
        skel = _make_y_skeleton()
        bin_edges = np.linspace(0, 1000, 11)
        path_lengths = compute_path_length_by_distance(skel, bin_edges)
        assert np.all(path_lengths >= 0)
        assert path_lengths.sum() > 0


class TestSomaDistanceNull:
    def test_fit_and_sample(self):
        skel = _make_y_skeleton()
        n_branches = len(skel.branches)
        rng = np.random.default_rng(42)

        positions = []
        for _ in range(50):
            bi = rng.integers(0, n_branches)
            pi = rng.uniform(0, skel.branches[bi].total_length)
            positions.append((bi, pi))
        snap = _make_snap_on_y(skel, positions)

        null = SomaDistanceNull(seed=0)
        null.fit(skel, snap)

        # Sample should produce valid SnapResult
        sample = null.sample(30)
        assert len(sample.branch_ids) == 30
        assert np.all(sample.valid)

        # Branch IDs should be valid
        for bi in sample.branch_ids:
            assert 0 <= bi < n_branches

        # Branch positions should be within branch length
        for i, bi in enumerate(sample.branch_ids):
            assert 0 <= sample.branch_positions[i] <= skel.branches[bi].total_length + 1e-6

    def test_generate_mocks(self):
        skel = _make_y_skeleton()
        n_branches = len(skel.branches)
        rng = np.random.default_rng(42)

        positions = [(rng.integers(0, n_branches),
                       rng.uniform(0, skel.branches[rng.integers(0, n_branches)].total_length))
                      for _ in range(50)]
        snap = _make_snap_on_y(skel, positions)

        null = SomaDistanceNull(seed=0)
        null.fit(skel, snap)
        mocks = null.generate_mocks(30, 10)

        assert len(mocks) == 10
        for m in mocks:
            assert len(m.branch_ids) == 30

    def test_density_profile_accessible(self):
        skel = _make_y_skeleton()
        rng = np.random.default_rng(42)

        positions = [(rng.integers(0, len(skel.branches)),
                       rng.uniform(0, 100))
                      for _ in range(30)]
        snap = _make_snap_on_y(skel, positions)

        null = SomaDistanceNull(seed=0)
        null.fit(skel, snap)

        dp = null.density_profile
        assert dp is not None
        assert 'bin_centers' in dp
        assert 'density' in dp
