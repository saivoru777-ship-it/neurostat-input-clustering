"""Extract presynaptic partner data from the MICrONS HDF5 connectome.

HDF5 structure (microns_mm3_connectome_v1181.h5):
  connectivity/full/edge_indices/block0_values: (13.5M, 2)
    col 0 = pre_vertex_idx, col 1 = post_vertex_idx
  connectivity/full/edges/block0_values: (13.5M, 8)
    cols: size, id, x_nm, y_nm, z_nm, delta_x_nm, delta_y_nm, delta_z_nm
  connectivity/full/vertex_properties/table (structured):
    values_block_0[:,0] = root_id (int64)
    values_block_2[:,1] = cell_type string (e.g. 'excitatory_neuron')

Memory strategy: Load only post column, boolean-mask for target neuron,
then load matching rows for coordinates and pre-column.
"""

import numpy as np
import pandas as pd


def load_vertex_properties(hdf5_path):
    """Load vertex properties (root_id, cell_type) from HDF5.

    Parameters
    ----------
    hdf5_path : str or Path
        Path to microns_mm3_connectome_v1181.h5

    Returns
    -------
    DataFrame with columns: vertex_idx, root_id, cell_type
    """
    import h5py

    with h5py.File(str(hdf5_path), 'r') as f:
        vp = f['connectivity']['full']['vertex_properties']['table']

        # root_id is in values_block_0, column 0
        root_ids = vp['values_block_0'][:, 0].astype(np.int64)

        # cell_type is in values_block_2, column 1
        # (col 0 is a flag like 't'/'f', col 1 is the type string)
        # Stored as bytes — decode to string
        try:
            cell_types_raw = vp['values_block_2'][:, 1]
            if isinstance(cell_types_raw[0], bytes):
                cell_types = np.array([ct.decode('utf-8') for ct in cell_types_raw])
            else:
                cell_types = cell_types_raw.astype(str)
        except (KeyError, IndexError):
            cell_types = np.full(len(root_ids), 'unknown', dtype=object)

    df = pd.DataFrame({
        'vertex_idx': np.arange(len(root_ids)),
        'root_id': root_ids,
        'cell_type': cell_types,
    })
    return df


def extract_presynaptic_partners(hdf5_path, post_vertex_idx, vertex_df=None):
    """Extract all presynaptic partners for a given post-synaptic neuron.

    Parameters
    ----------
    hdf5_path : str or Path
        Path to microns_mm3_connectome_v1181.h5
    post_vertex_idx : int
        Vertex index of the post-synaptic neuron in the HDF5 vertex table.
    vertex_df : DataFrame, optional
        Output of load_vertex_properties(). Loaded if not provided.

    Returns
    -------
    DataFrame with columns: x_nm, y_nm, z_nm, pre_root_id, pre_cell_type, pre_vertex_idx
    """
    import h5py

    if vertex_df is None:
        vertex_df = load_vertex_properties(hdf5_path)

    with h5py.File(str(hdf5_path), 'r') as f:
        # edge_indices are in block0_values: (N, 2) — col 0=pre, col 1=post
        edge_idx = f['connectivity']['full']['edge_indices']['block0_values']
        post_col = edge_idx[:, 1]

        # Boolean mask for target neuron
        mask = post_col == post_vertex_idx
        n_synapses = mask.sum()

        if n_synapses == 0:
            return pd.DataFrame(columns=['x_nm', 'y_nm', 'z_nm',
                                         'pre_root_id', 'pre_cell_type',
                                         'pre_vertex_idx'])

        # Load matching pre vertex indices
        pre_vertices = edge_idx[mask, 0]

        # Load synapse coordinates from edges/block0_values
        # Columns: size, id, x_nm, y_nm, z_nm, delta_x, delta_y, delta_z
        edges_block = f['connectivity']['full']['edges']['block0_values']
        coords = edges_block[mask, 2:5]

    # Map pre vertex indices to root_id and cell_type
    vid_to_root = dict(zip(vertex_df['vertex_idx'], vertex_df['root_id']))
    vid_to_type = dict(zip(vertex_df['vertex_idx'], vertex_df['cell_type']))

    pre_root_ids = np.array([vid_to_root.get(v, -1) for v in pre_vertices])
    pre_cell_types = np.array([vid_to_type.get(v, 'unknown') for v in pre_vertices])

    result = pd.DataFrame({
        'x_nm': coords[:, 0],
        'y_nm': coords[:, 1],
        'z_nm': coords[:, 2],
        'pre_root_id': pre_root_ids,
        'pre_cell_type': pre_cell_types,
        'pre_vertex_idx': pre_vertices,
    })
    return result


def find_vertex_idx(vertex_df, root_id):
    """Find the vertex index for a given root_id.

    Parameters
    ----------
    vertex_df : DataFrame from load_vertex_properties
    root_id : int

    Returns
    -------
    int or None
    """
    matches = vertex_df[vertex_df['root_id'] == root_id]
    if len(matches) == 0:
        return None
    return int(matches.iloc[0]['vertex_idx'])


def extract_presynaptic_partners_by_root_id(hdf5_path, post_root_id, vertex_df=None):
    """Convenience wrapper: extract partners using post-synaptic root_id.

    Parameters
    ----------
    hdf5_path : str or Path
    post_root_id : int
        Root ID of the post-synaptic neuron.
    vertex_df : DataFrame, optional

    Returns
    -------
    DataFrame with columns: x_nm, y_nm, z_nm, pre_root_id, pre_cell_type, pre_vertex_idx
    """
    if vertex_df is None:
        vertex_df = load_vertex_properties(hdf5_path)

    post_vidx = find_vertex_idx(vertex_df, post_root_id)
    if post_vidx is None:
        print(f"  Warning: root_id {post_root_id} not found in vertex table")
        return pd.DataFrame(columns=['x_nm', 'y_nm', 'z_nm',
                                     'pre_root_id', 'pre_cell_type',
                                     'pre_vertex_idx'])

    return extract_presynaptic_partners(hdf5_path, post_vidx, vertex_df)


def classify_cell_type_broad(cell_type_str):
    """Map fine cell type string to broad category.

    Returns one of: 'excitatory', 'inhibitory', 'other', 'unknown'
    """
    ct = str(cell_type_str).lower().strip()

    if ct in ('', 'unknown', 'nan', 'none'):
        return 'unknown'

    # Check inhibitory first — avoids false matches from short excitatory
    # patterns (e.g. 'it' matching inside 'inhibitory')
    inh_patterns = ['inhibitory', 'interneuron', 'gaba', 'pvalb', 'sst',
                    'vip', 'lamp5', 'basket', 'chandelier', 'martinotti',
                    'bipolar']
    for p in inh_patterns:
        if p in ct:
            return 'inhibitory'

    # Excitatory types
    exc_patterns = ['excitatory', 'pyramidal', 'spiny', 'stellate']
    for p in exc_patterns:
        if p in ct:
            return 'excitatory'

    return 'other'
