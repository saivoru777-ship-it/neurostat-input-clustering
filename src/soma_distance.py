"""Fast geodesic distance computation and soma-distance utilities.

The existing skeleton.geodesic_distance() does BFS per pair — O(B) per call.
For N=1000 synapses, the N×N matrix takes ~30 min. This module precomputes
a B×B endpoint distance matrix using Dijkstra, enabling O(1) pairwise lookups.

Also provides soma-distance computation and distance-density profile estimation
for the soma-distance null model.
"""

import numpy as np
from collections import defaultdict
import heapq

from neurostat.io.swc import NeuronSkeleton, SnapResult


def precompute_branch_endpoint_distances(skeleton):
    """Precompute shortest-path distances between all branch endpoints.

    Uses Dijkstra on the branch adjacency graph. Each branch is an edge
    with weight = branch.total_length.

    Parameters
    ----------
    skeleton : NeuronSkeleton
        Must have branches already decomposed.

    Returns
    -------
    endpoint_nodes : list
        Unique endpoint node IDs (branch points / terminals).
    node_to_idx : dict
        Maps node_id -> index in the distance matrix.
    endpoint_dists : ndarray, shape (E, E)
        Pairwise shortest-path distances between endpoints.
    """
    # Collect unique endpoint nodes
    endpoint_set = set()
    for branch in skeleton.branches:
        endpoint_set.add(branch.start_node)
        endpoint_set.add(branch.end_node)
    endpoint_nodes = sorted(endpoint_set)
    node_to_idx = {nid: i for i, nid in enumerate(endpoint_nodes)}
    E = len(endpoint_nodes)

    # Build adjacency: node -> list of (neighbor_node, distance)
    adj = defaultdict(list)
    for branch in skeleton.branches:
        s, e = branch.start_node, branch.end_node
        w = branch.total_length
        adj[s].append((e, w))
        adj[e].append((s, w))

    # Dijkstra from each endpoint
    endpoint_dists = np.full((E, E), np.inf)
    np.fill_diagonal(endpoint_dists, 0.0)

    for src_idx, src_node in enumerate(endpoint_nodes):
        dist = {src_node: 0.0}
        heap = [(0.0, src_node)]
        while heap:
            d, u = heapq.heappop(heap)
            if d > dist.get(u, np.inf):
                continue
            for v, w in adj[u]:
                nd = d + w
                if nd < dist.get(v, np.inf):
                    dist[v] = nd
                    heapq.heappush(heap, (nd, v))
        for nid, d in dist.items():
            if nid in node_to_idx:
                endpoint_dists[src_idx, node_to_idx[nid]] = d

    return endpoint_nodes, node_to_idx, endpoint_dists


def fast_geodesic_distance(snap_i, snap_j, skeleton, node_to_idx, endpoint_dists):
    """O(1) geodesic distance between two snapped points using precomputed data.

    Parameters
    ----------
    snap_i, snap_j : tuple (branch_id, branch_position)
    skeleton : NeuronSkeleton
    node_to_idx : dict from precompute_branch_endpoint_distances
    endpoint_dists : ndarray from precompute_branch_endpoint_distances

    Returns
    -------
    float : geodesic distance in nm
    """
    bi, pi = snap_i
    bj, pj = snap_j

    if bi == bj:
        return abs(pi - pj)

    branch_i = skeleton.branches[bi]
    branch_j = skeleton.branches[bj]

    # From point on branch_i, can exit via start (cost pi) or end (cost L_i - pi)
    si_s, si_e = branch_i.start_node, branch_i.end_node
    di_to_start = pi
    di_to_end = branch_i.total_length - pi

    # To reach point on branch_j, can enter via start (cost pj) or end (cost L_j - pj)
    sj_s, sj_e = branch_j.start_node, branch_j.end_node
    dj_from_start = pj
    dj_from_end = branch_j.total_length - pj

    # 4 possible paths through endpoint pairs
    best = np.inf
    for exit_node, d_exit in [(si_s, di_to_start), (si_e, di_to_end)]:
        for enter_node, d_enter in [(sj_s, dj_from_start), (sj_e, dj_from_end)]:
            ei = node_to_idx[exit_node]
            ej = node_to_idx[enter_node]
            total = d_exit + endpoint_dists[ei, ej] + d_enter
            if total < best:
                best = total

    return best


def fast_geodesic_distance_matrix(skeleton, snap_result, node_to_idx=None,
                                   endpoint_dists=None):
    """Compute N×N pairwise geodesic distance matrix using precomputed endpoints.

    Parameters
    ----------
    skeleton : NeuronSkeleton
    snap_result : SnapResult
    node_to_idx : dict, optional
        If None, calls precompute_branch_endpoint_distances.
    endpoint_dists : ndarray, optional

    Returns
    -------
    D : ndarray, shape (N, N)
    """
    if node_to_idx is None or endpoint_dists is None:
        _, node_to_idx, endpoint_dists = precompute_branch_endpoint_distances(skeleton)

    n = len(snap_result.branch_ids)
    D = np.zeros((n, n))

    # Precompute per-synapse exit costs to avoid repeated computation
    # For each synapse: (branch_id, pos, start_node_idx, end_node_idx, d_to_start, d_to_end)
    syn_data = []
    for i in range(n):
        bi = snap_result.branch_ids[i]
        pi = snap_result.branch_positions[i]
        branch = skeleton.branches[bi]
        s_idx = node_to_idx[branch.start_node]
        e_idx = node_to_idx[branch.end_node]
        d_start = pi
        d_end = branch.total_length - pi
        syn_data.append((bi, pi, s_idx, e_idx, d_start, d_end))

    for i in range(n):
        bi_i, pi_i, si_s, si_e, di_s, di_e = syn_data[i]
        for j in range(i + 1, n):
            bi_j, pi_j, sj_s, sj_e, dj_s, dj_e = syn_data[j]

            if bi_i == bi_j:
                d = abs(pi_i - pi_j)
            else:
                # 4 paths through endpoint pairs
                d = min(
                    di_s + endpoint_dists[si_s, sj_s] + dj_s,
                    di_s + endpoint_dists[si_s, sj_e] + dj_e,
                    di_e + endpoint_dists[si_e, sj_s] + dj_s,
                    di_e + endpoint_dists[si_e, sj_e] + dj_e,
                )
            D[i, j] = d
            D[j, i] = d

    return D


def compute_soma_distances(skeleton, snap_result):
    """Compute geodesic distance from soma to each synapse.

    Parameters
    ----------
    skeleton : NeuronSkeleton
    snap_result : SnapResult

    Returns
    -------
    soma_dists : ndarray, shape (N,)
        Geodesic distance from soma (root) to each synapse, in nm.
    """
    _, node_to_idx, endpoint_dists = precompute_branch_endpoint_distances(skeleton)

    # Find soma node index — root_id should be an endpoint
    if skeleton.root_id in node_to_idx:
        soma_idx = node_to_idx[skeleton.root_id]
    else:
        # Root may not be a branch endpoint if it has degree 2
        # Find the closest endpoint to root
        root = skeleton.nodes[skeleton.root_id]
        root_pos = np.array([root.x, root.y, root.z])
        best_dist = np.inf
        soma_idx = 0
        for nid, idx in node_to_idx.items():
            node = skeleton.nodes[nid]
            d = np.sqrt((node.x - root_pos[0])**2 + (node.y - root_pos[1])**2 +
                        (node.z - root_pos[2])**2)
            if d < best_dist:
                best_dist = d
                soma_idx = idx

    n = len(snap_result.branch_ids)
    soma_dists = np.empty(n)

    for i in range(n):
        bi = snap_result.branch_ids[i]
        pi = snap_result.branch_positions[i]
        branch = skeleton.branches[bi]

        s_idx = node_to_idx[branch.start_node]
        e_idx = node_to_idx[branch.end_node]

        # Distance via start endpoint or end endpoint
        d_via_start = pi + endpoint_dists[soma_idx, s_idx]
        d_via_end = (branch.total_length - pi) + endpoint_dists[soma_idx, e_idx]
        soma_dists[i] = min(d_via_start, d_via_end)

    return soma_dists


def estimate_distance_density_profile(soma_distances, skeleton, n_bins=30,
                                       bandwidth=None):
    """Estimate synapse density as a function of soma distance.

    Density = synapse_count(d) / path_length(d), smoothed with Gaussian KDE.

    Parameters
    ----------
    soma_distances : ndarray
        Soma distance for each synapse.
    skeleton : NeuronSkeleton
    n_bins : int
        Number of distance bins.
    bandwidth : float, optional
        KDE bandwidth in nm. Default: 10% of distance range.

    Returns
    -------
    bin_centers : ndarray, shape (n_bins,)
    density : ndarray, shape (n_bins,)
        Smoothed density (synapses per nm of path length).
    """
    from scipy.ndimage import gaussian_filter1d

    # Filter out infinite/NaN soma distances (disconnected branches)
    finite_mask = np.isfinite(soma_distances)
    finite_dists = soma_distances[finite_mask]
    if len(finite_dists) == 0:
        return np.zeros(n_bins), np.zeros(n_bins)

    d_max = finite_dists.max() * 1.05
    d_min = 0.0
    bin_edges = np.linspace(d_min, d_max, n_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    # Count synapses per bin
    syn_counts, _ = np.histogram(soma_distances, bins=bin_edges)

    # Compute path length per distance bin
    path_lengths = compute_path_length_by_distance(skeleton, bin_edges)

    # Density = count / path_length (avoid division by zero)
    with np.errstate(divide='ignore', invalid='ignore'):
        raw_density = np.where(path_lengths > 0, syn_counts / path_lengths, 0.0)

    # Smooth with Gaussian kernel
    if bandwidth is None:
        bandwidth = (d_max - d_min) * 0.10
    bin_width = bin_edges[1] - bin_edges[0]
    sigma = bandwidth / bin_width
    density = gaussian_filter1d(raw_density.astype(float), sigma=max(sigma, 0.5))

    # Ensure non-negative
    density = np.maximum(density, 0.0)

    return bin_centers, density


def compute_path_length_by_distance(skeleton, bin_edges):
    """Compute total dendritic path length available at each soma distance band.

    For each skeleton edge, compute the soma distance at its midpoint and
    accumulate the edge length into the corresponding distance bin.

    Parameters
    ----------
    skeleton : NeuronSkeleton
    bin_edges : ndarray, shape (n_bins + 1,)

    Returns
    -------
    path_lengths : ndarray, shape (n_bins,)
        Total path length (nm) in each distance band.
    """
    _, node_to_idx, endpoint_dists = precompute_branch_endpoint_distances(skeleton)

    # Find soma endpoint index
    if skeleton.root_id in node_to_idx:
        soma_idx = node_to_idx[skeleton.root_id]
    else:
        root = skeleton.nodes[skeleton.root_id]
        root_pos = np.array([root.x, root.y, root.z])
        best_dist = np.inf
        soma_idx = 0
        for nid, idx in node_to_idx.items():
            node = skeleton.nodes[nid]
            d = np.sqrt((node.x - root_pos[0])**2 + (node.y - root_pos[1])**2 +
                        (node.z - root_pos[2])**2)
            if d < best_dist:
                best_dist = d
                soma_idx = idx

    n_bins = len(bin_edges) - 1
    path_lengths = np.zeros(n_bins)

    for branch in skeleton.branches:
        s_idx = node_to_idx[branch.start_node]
        e_idx = node_to_idx[branch.end_node]
        d_soma_start = endpoint_dists[soma_idx, s_idx]
        d_soma_end = endpoint_dists[soma_idx, e_idx]

        # Walk along edges, compute soma distance at midpoint
        cum = 0.0
        for ei in range(len(branch.edge_lengths)):
            elen = branch.edge_lengths[ei]
            mid_pos = cum + elen / 2.0

            # Interpolate soma distance at midpoint
            t = mid_pos / branch.total_length if branch.total_length > 0 else 0.5
            # Soma distance at a point along the branch: min path via start or end
            d_via_start = mid_pos + d_soma_start
            d_via_end = (branch.total_length - mid_pos) + d_soma_end
            soma_dist_mid = min(d_via_start, d_via_end)

            # Assign edge length to the appropriate bin (skip disconnected)
            if np.isfinite(soma_dist_mid):
                bin_idx = np.searchsorted(bin_edges, soma_dist_mid, side='right') - 1
                if 0 <= bin_idx < n_bins:
                    path_lengths[bin_idx] += elen

            cum += elen

    return path_lengths


def compute_segment_weights(skeleton, snap_result, bin_centers, density,
                            bin_edges, segment_spacing=100.0):
    """Compute sampling weights for skeleton segments based on distance-density.

    Divides each branch into ~100nm segments and assigns each a weight
    proportional to segment_length × density(soma_distance_at_midpoint).

    Parameters
    ----------
    skeleton : NeuronSkeleton
    snap_result : SnapResult (unused, kept for API compatibility)
    bin_centers : ndarray from estimate_distance_density_profile
    density : ndarray from estimate_distance_density_profile
    bin_edges : ndarray
    segment_spacing : float
        Target segment length in nm.

    Returns
    -------
    segments : list of (branch_idx, start_pos, end_pos, weight)
    total_weight : float
    """
    _, node_to_idx, endpoint_dists = precompute_branch_endpoint_distances(skeleton)

    if skeleton.root_id in node_to_idx:
        soma_idx = node_to_idx[skeleton.root_id]
    else:
        root = skeleton.nodes[skeleton.root_id]
        root_pos = np.array([root.x, root.y, root.z])
        best_dist = np.inf
        soma_idx = 0
        for nid, idx in node_to_idx.items():
            node = skeleton.nodes[nid]
            d = np.sqrt((node.x - root_pos[0])**2 + (node.y - root_pos[1])**2 +
                        (node.z - root_pos[2])**2)
            if d < best_dist:
                best_dist = d
                soma_idx = idx

    segments = []
    for bi, branch in enumerate(skeleton.branches):
        if branch.total_length < 1e-6:
            continue

        s_idx = node_to_idx[branch.start_node]
        e_idx = node_to_idx[branch.end_node]
        d_soma_start = endpoint_dists[soma_idx, s_idx]
        d_soma_end = endpoint_dists[soma_idx, e_idx]

        n_segs = max(1, int(np.ceil(branch.total_length / segment_spacing)))
        seg_len = branch.total_length / n_segs

        for si in range(n_segs):
            start_pos = si * seg_len
            end_pos = (si + 1) * seg_len
            mid_pos = (start_pos + end_pos) / 2.0

            # Soma distance at midpoint
            d_via_start = mid_pos + d_soma_start
            d_via_end = (branch.total_length - mid_pos) + d_soma_end
            soma_dist = min(d_via_start, d_via_end)

            # Skip segments with infinite soma distance (disconnected)
            if not np.isfinite(soma_dist):
                segments.append((bi, start_pos, end_pos, 0.0))
                continue

            # Look up density
            bin_idx = np.searchsorted(bin_edges, soma_dist, side='right') - 1
            bin_idx = np.clip(bin_idx, 0, len(density) - 1)
            w = seg_len * density[bin_idx]

            segments.append((bi, start_pos, end_pos, w))

    total_weight = sum(s[3] for s in segments)
    return segments, total_weight
