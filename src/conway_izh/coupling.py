"""Coupling mechanisms between Conway Game of Life and Izhikevich neurons."""

from typing import List, Optional, Sequence, Tuple

import numpy as np

from conway_izh.config import CouplingParams


def gol_to_current(
    gol_state: np.ndarray,
    neighbor_count: np.ndarray,
    coupling: CouplingParams,
) -> np.ndarray:
    """
    Convert Conway Game of Life state to input current for neurons.

    I = k_neighbors * neighbors + k_alive * alive + bias

    Args:
        gol_state: Binary Conway grid (H, W) dtype uint8
        neighbor_count: Neighbor count grid (H, W) dtype uint8
        coupling: Coupling parameters

    Returns:
        Input current array (H, W) dtype float64
    """
    I = (
        coupling.k_neighbors * neighbor_count.astype(np.float64)
        + coupling.k_alive * gol_state.astype(np.float64)
        + coupling.bias
    )
    return I


def adjacency_from_edges(
    edges: Sequence[Tuple[int, int]], n_nodes: int
) -> List[np.ndarray]:
    """CSR-style neighbor lists: adj[i] = array of vertices j connected by an edge."""
    buckets: List[List[int]] = [[] for _ in range(n_nodes)]
    for a, b in edges:
        ai = int(a)
        bi = int(b)
        if ai == bi or not (0 <= ai < n_nodes and 0 <= bi < n_nodes):
            continue
        buckets[ai].append(bi)
        buckets[bi].append(ai)
    return [np.array(sorted(set(xs)), dtype=np.int32) for xs in buckets]


def graph_trace_to_drive(
    trace_flat: np.ndarray, edges_arr: np.ndarray, k_syn: float
) -> np.ndarray:
    """
    Sum trace over undirected edges: drive[i] = k_syn * sum_{j~(i,j)} trace[j].

    Args:
        trace_flat: (N,) spike trace per node (flattened row-major grid)
        edges_arr: (E, 2) int32 endpoints
        k_syn: Scaling (same units as gol_to_current bias term)

    Returns:
        drive (N,)
    Time: O(E), vectorized.
    """
    if k_syn == 0.0 or edges_arr.size == 0:
        return np.zeros_like(trace_flat, dtype=np.float64)
    n = trace_flat.shape[0]
    acc = np.zeros(n, dtype=np.float64)
    a = edges_arr[:, 0]
    b = edges_arr[:, 1]
    ta = trace_flat[b]
    tb = trace_flat[a]
    np.add.at(acc, a, k_syn * ta)
    np.add.at(acc, b, k_syn * tb)
    return acc


def neuron_to_gol_feedback(
    gol_state: np.ndarray,
    spikes: np.ndarray,
    graph_adj: Optional[List[np.ndarray]] = None,
    excite_neighbors: bool = False,
) -> np.ndarray:
    """
    Apply neuron spike feedback to Conway grid.

    Base: spike locations become alive.
    If excite_neighbors and graph_adj: same for each graph neighbor of spiking nodes.
    """
    new_state = gol_state.copy()
    new_state[spikes] = 1
    if not excite_neighbors or graph_adj is None:
        return new_state

    flat = spikes.ravel()
    for idx in np.flatnonzero(flat.astype(np.uint8)):
        for j in graph_adj[int(idx)]:
            r = int(j) // gol_state.shape[1]
            c = int(j) % gol_state.shape[1]
            new_state[r, c] = 1
    return new_state
