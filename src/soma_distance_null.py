"""Soma-distance null model for synapse placement.

Synapses are placed with probability proportional to the observed
distance-dependent density profile, rather than uniformly along path length.
This captures first-order (distance gradient) structure, allowing tests of
higher-order (e.g., partner-specific) clustering.

Interface matches DendriteConstrainedNull: fit(), sample(), generate_mocks().
"""

import numpy as np
from neurostat.io.swc import SnapResult

from .soma_distance import (
    compute_soma_distances,
    estimate_distance_density_profile,
    compute_segment_weights,
    precompute_branch_endpoint_distances,
)


class SomaDistanceNull:
    """Null model: synapse placement weighted by soma-distance density profile.

    The density profile is estimated from the real synapse data using KDE,
    normalized by available path length at each distance. This means the
    null accounts for known distance-dependent gradients (e.g., proximal
    enrichment) but not for partner-specific clustering.
    """

    def __init__(self, seed=None):
        self.rng = np.random.default_rng(seed)
        self.skeleton = None
        self.branches = None
        self._segments = None      # list of (branch_idx, start_pos, end_pos, weight)
        self._seg_weights = None   # normalized weights for sampling
        self._bin_centers = None
        self._density = None
        self._bin_edges = None

    def fit(self, skeleton, snap_result, n_bins=30, bandwidth=None):
        """Fit the soma-distance density profile from real synapse data.

        Parameters
        ----------
        skeleton : NeuronSkeleton
        snap_result : SnapResult
            Real synapse data (valid synapses only).
        n_bins : int
            Number of distance bins for density estimation.
        bandwidth : float, optional
            KDE bandwidth in nm.

        Returns
        -------
        self
        """
        self.skeleton = skeleton
        self.branches = skeleton.branches

        # Compute soma distances for real synapses
        soma_dists = compute_soma_distances(skeleton, snap_result)

        # Filter out infinite distances (disconnected skeleton branches)
        finite_mask = np.isfinite(soma_dists)
        finite_dists = soma_dists[finite_mask]
        if len(finite_dists) == 0:
            finite_dists = np.array([0.0])

        # Estimate density profile
        d_max = finite_dists.max() * 1.05
        bin_edges = np.linspace(0.0, d_max, n_bins + 1)
        bin_centers, density = estimate_distance_density_profile(
            soma_dists, skeleton, n_bins=n_bins, bandwidth=bandwidth
        )
        self._bin_centers = bin_centers
        self._density = density
        self._bin_edges = bin_edges

        # Compute segment weights
        segments, total_weight = compute_segment_weights(
            skeleton, snap_result, bin_centers, density, bin_edges
        )
        self._segments = segments

        if total_weight > 0:
            self._seg_weights = np.array([s[3] for s in segments]) / total_weight
        else:
            # Fallback to uniform if density estimation fails
            n_segs = len(segments)
            self._seg_weights = np.ones(n_segs) / max(n_segs, 1)

        return self

    def sample(self, n):
        """Generate n synapses weighted by the soma-distance density profile.

        Parameters
        ----------
        n : int
            Number of synapses to generate.

        Returns
        -------
        SnapResult
        """
        if self._segments is None:
            raise RuntimeError("Must call fit() before sample()")

        # Sample segments proportional to weights
        seg_indices = self.rng.choice(
            len(self._segments), size=n, p=self._seg_weights
        )

        branch_ids = np.empty(n, dtype=int)
        branch_positions = np.empty(n, dtype=float)

        for i, si in enumerate(seg_indices):
            bi, start_pos, end_pos, _ = self._segments[si]
            branch_ids[i] = bi
            branch_positions[i] = self.rng.uniform(start_pos, end_pos)

        return SnapResult(
            branch_ids=branch_ids,
            branch_positions=branch_positions,
            distances=np.zeros(n),
            valid=np.ones(n, dtype=bool),
        )

    def generate_mocks(self, n_points, n_mocks):
        """Generate multiple null SnapResults.

        Parameters
        ----------
        n_points : int
            Number of synapses per mock.
        n_mocks : int
            Number of mock samples.

        Returns
        -------
        list of SnapResult
        """
        return [self.sample(n_points) for _ in range(n_mocks)]

    @property
    def density_profile(self):
        """Return the fitted density profile for inspection/plotting."""
        if self._bin_centers is None:
            return None
        return {
            'bin_centers': self._bin_centers,
            'density': self._density,
            'bin_edges': self._bin_edges,
        }
