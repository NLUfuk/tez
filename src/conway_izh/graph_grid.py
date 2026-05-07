"""Graph-based Conway + Izhikevich orchestrator for arbitrary topologies.

This module mirrors :class:`conway_izh.grid.NeuralGrid` but operates on a
flat ``(N,)`` representation backed by a SciPy CSR adjacency matrix. It is
the engine the front-end's "Topoloji Seçimi" feature drives when one or
more SWC morphologies (or a small-world ring) are merged into a unified
graph.

Memory rationale
----------------
For the unified graph we cannot reuse the dense ``(H, W)`` arrays because
SWC morphologies are not rectangular. Storing state as ``(N,)`` 1-D
arrays plus a sparse adjacency is O(N + E) which fits comfortably in the
8 GB RAM budget even for tens of thousands of nodes.

Conway-on-graph rules
---------------------
We keep the standard B/S notation (default B3/S23) but interpret
"neighbors" as graph neighbors (degree depends on the topology). This is
the canonical generalisation of Conway to arbitrary graphs and makes the
existing UI rule controls (Birth/Survive lists) apply unchanged.

Per-step time complexity:
    O(E) for neighbor counts and synaptic drive (sparse mat-vec)
    + O(N) for Izhikevich update, much cheaper than a dense O(N^2).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import sparse as sp

from conway_izh.config import SimulationConfig
from conway_izh.izhikevich import initialize_izhikevich, step_izhikevich
from conway_izh.topology_manager import UnifiedTopology


# ---------------------------------------------------------------------------
# Helpers (graph-flavoured analogues of the dense functions in conway_izh.*)
# ---------------------------------------------------------------------------


def graph_neighbor_count(adjacency: sp.csr_matrix, state_flat: np.ndarray) -> np.ndarray:
    """Count alive neighbors for each node via sparse mat-vec.

    ``adj @ state`` is the cheapest correct implementation: O(E).
    Returned dtype is ``np.uint16`` to safely accommodate hub nodes whose
    degree may exceed 255 in dense small-world settings.
    """
    if adjacency.shape[0] == 0:
        return np.zeros(0, dtype=np.uint16)
    counts = adjacency.dot(state_flat.astype(np.uint16, copy=False))
    return np.asarray(counts, dtype=np.uint16)


def graph_update_conway(
    state_flat: np.ndarray,
    neighbor_count: np.ndarray,
    degree: np.ndarray,
    *,
    birth: Sequence[int] = (3,),
    survive: Sequence[int] = (2, 3),
    low_degree_loose: bool = True,
) -> np.ndarray:
    """One generation of B/S Conway over an arbitrary graph.

    Conway is degree-8 by construction. To generalize cleanly we *quantize*
    each cell's local fill ratio to the canonical 0..8 axis and then test
    membership in the user's B/S sets:

        equivalent_count = round(alive_neighbors * 8 / max(degree, 1))
        birth  if dead  and equivalent_count ∈ B
        survive if alive and equivalent_count ∈ S

    On a 2-D Moore grid (degree=8) this collapses exactly to the classic
    rule. On heterogeneous graphs it preserves the *spirit* of B3/S23 –
    cells thrive when ~30 % of their neighbours are alive – which is what
    the user's UI controls expect.

    Low-degree fallback (default ON):
        Cells with ``degree ≤ 2`` (dendrite leaves and chains in SWC
        morphologies) cannot reach the canonical [25 %, 38 %] band: they
        either have 0 alive neighbours (0 %) or 1 (50 % / 100 %). Without
        a fallback the entire dendritic backbone dies in one step,
        flat-lining the simulation. Letting them survive whenever at
        least one neighbour is alive preserves visual continuity along
        branches without affecting the small-world / hub dynamics.

    Time:  O(N), fully vectorised.
    """
    birth_arr = np.asarray(tuple(birth) or (3,), dtype=np.uint16)
    survive_arr = np.asarray(tuple(survive) or (2, 3), dtype=np.uint16)

    deg_safe = np.maximum(degree.astype(np.float32, copy=False), np.float32(1.0))
    equiv = np.rint(
        neighbor_count.astype(np.float32, copy=False) * np.float32(8.0) / deg_safe
    ).astype(np.uint16, copy=False)

    alive = state_flat.astype(bool, copy=False)
    birth_mask = np.isin(equiv, birth_arr)
    survive_mask = np.isin(equiv, survive_arr)

    new_state = np.zeros_like(state_flat, dtype=np.uint8)
    new_state[(~alive) & birth_mask] = 1
    new_state[alive & survive_mask] = 1

    if low_degree_loose:
        low_deg = degree.astype(np.uint16, copy=False) <= np.uint16(2)
        loose_survive = alive & low_deg & (neighbor_count >= np.uint16(1))
        new_state[loose_survive] = 1

    return new_state


def graph_trace_drive(
    adjacency: sp.csr_matrix, trace_flat: np.ndarray, k_syn: float
) -> np.ndarray:
    """Spike-trace driven synaptic input: ``k_syn * (A @ trace)``. O(E)."""
    if k_syn == 0.0 or adjacency.nnz == 0:
        return np.zeros(adjacency.shape[0], dtype=np.float64)
    drive = adjacency.dot(trace_flat.astype(np.float64, copy=False))
    return (k_syn * drive).astype(np.float64, copy=False)


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------


class GraphNeuralGrid:
    """Conway-Izhikevich loop over a :class:`UnifiedTopology`.

    The class deliberately mimics ``NeuralGrid`` so the streaming layer can
    treat both interchangeably (duck typing on ``gol_state``, ``v``, ``u``,
    ``step()`` and ``config``).

    Note that ``gol_state``, ``v``, ``u``, etc. are 1-D ``(N,)`` arrays
    here, not 2-D ``(H, W)`` grids. Code that needs to map an index back to
    XYZ should use ``self.topology.coords``.
    """

    def __init__(self, config: SimulationConfig, topology: UnifiedTopology):
        if topology is None or topology.n_nodes == 0:
            raise ValueError("GraphNeuralGrid requires a non-empty UnifiedTopology")

        self.config = config
        self.topology = topology
        self.n_nodes = topology.n_nodes

        # Conway state
        rng = np.random.default_rng(config.seed if config.seed is not None else 42)
        self.gol_state = (
            rng.random(self.n_nodes) < 0.30
        ).astype(np.uint8)

        # Izhikevich state
        self.v, self.u = initialize_izhikevich(
            (self.n_nodes,), config.seed
        )

        # Auxiliary buffers required by the existing Game-Theory + memory
        # bookkeeping. They live as 1-D arrays here (the GT code path is
        # disabled for graph mode by default to keep this tractable).
        self.previous_spikes = np.zeros(self.n_nodes, dtype=bool)
        self.memory_state = np.zeros(self.n_nodes, dtype=np.float64)
        self.strategy_map = np.ones(self.n_nodes, dtype=np.int8)
        self.spike_trace = np.zeros(self.n_nodes, dtype=np.float64)

        # Pre-cache adjacency in CSR (already CSR but defensive).
        self.adjacency: sp.csr_matrix = topology.adjacency.tocsr()
        self.adjacency.data = self.adjacency.data.astype(np.uint8, copy=False)
        self.edges_arr = topology.edges_arr  # (E, 2) int32

        # Per-node degree, cached once. uint16 covers tens-of-thousands hubs.
        deg = np.asarray(self.adjacency.sum(axis=1)).reshape(-1)
        self.degree = deg.astype(np.uint16, copy=False)

        self.metrics_history: List[Dict[str, float]] = []
        self.spike_history: List[np.ndarray] = []

    # -- API parity helpers --------------------------------------------------

    @property
    def graph_edges(self) -> List[Tuple[int, int]]:
        """Match :class:`NeuralGrid.graph_edges` (used by StreamPublisher)."""
        if self.edges_arr.size == 0:
            return []
        return [(int(a), int(b)) for a, b in self.edges_arr]

    def reset_state(self, seed: Optional[int] = None) -> None:
        """Reset Conway and Izhikevich states (does not change topology)."""
        rng = np.random.default_rng(seed if seed is not None else 42)
        self.gol_state = (rng.random(self.n_nodes) < 0.30).astype(np.uint8)
        self.v, self.u = initialize_izhikevich((self.n_nodes,), seed)
        self.previous_spikes[:] = False
        self.memory_state[:] = 0.0
        self.strategy_map[:] = 1
        self.spike_trace[:] = 0.0

    # -- core step -----------------------------------------------------------

    def step(self) -> Tuple[Dict[str, float], np.ndarray]:
        """One Conway+Izhikevich generation. Returns (metrics, spikes_flat)."""
        cp = self.config.coupling

        neighbor_count = graph_neighbor_count(self.adjacency, self.gol_state)

        I = (
            cp.k_neighbors * neighbor_count.astype(np.float64)
            + cp.k_alive * self.gol_state.astype(np.float64)
            + cp.bias
        )

        if cp.k_syn != 0.0 and self.adjacency.nnz > 0:
            I = I + graph_trace_drive(self.adjacency, self.spike_trace, cp.k_syn)

        v_new, u_new, spikes = step_izhikevich(
            self.v, self.u, I, self.config.izh_params, self.config.dt
        )
        self.v, self.u = v_new, u_new

        gamma = float(np.clip(cp.spike_trace_decay, 0.0, 0.9999))
        self.spike_trace = (
            gamma * self.spike_trace + spikes.astype(np.float64)
        )

        new_state = graph_update_conway(
            self.gol_state,
            neighbor_count,
            self.degree,
            birth=self.config.birth_neighbors,
            survive=self.config.survive_neighbors,
        )

        if cp.feedback_enabled:
            # Direct feedback: spike forces "alive". For graph-neighbor revive
            # (option C) we light up *immediate* graph neighbors of any spike.
            new_state[spikes] = 1
            if cp.feedback_graph_neighbors and self.adjacency.nnz > 0:
                neighbor_excitation = self.adjacency.dot(spikes.astype(np.uint8))
                new_state[neighbor_excitation > 0] = 1

        self.gol_state = new_state
        self.previous_spikes = spikes.copy()

        if self.config.memory.enabled:
            spike_term = spikes.astype(np.float64) * self.config.memory.spike_gain
            # Normalise neighbor count by an estimate of the typical degree
            # so the memory term stays in [0, 1] regardless of topology.
            denom = max(1.0, float(neighbor_count.max()) if neighbor_count.size else 1.0)
            neighbor_term = (
                neighbor_count.astype(np.float64) / denom
            ) * self.config.memory.neighbor_gain
            self.memory_state = (
                self.config.memory.decay * self.memory_state
                + spike_term
                + neighbor_term
            )
            np.clip(
                self.memory_state,
                self.config.memory.clip_min,
                self.config.memory.clip_max,
                out=self.memory_state,
            )

        metrics = self._compute_metrics(spikes)
        return metrics, spikes

    # -- metrics -------------------------------------------------------------

    def _compute_metrics(self, spikes: np.ndarray) -> Dict[str, float]:
        from conway_izh.efficiency import (
            compute_stability_score,
            compute_information_score,
            compute_memory_score,
            compute_cost_score,
            compute_efficiency_score,
        )

        n = max(1, self.n_nodes)
        alive_count = int(self.gol_state.sum())
        spike_count = int(spikes.sum())
        alive_ratio = alive_count / n
        firing_rate = spike_count / n
        mean_v = float(self.v.mean()) if self.v.size else 0.0
        mean_memory = float(self.memory_state.mean()) if self.memory_state.size else 0.0
        cooperative_ratio = float((self.strategy_map == 1).mean()) if self.strategy_map.size else 0.0
        if self.metrics_history:
            prev_ratio = float(self.metrics_history[-1].get("cooperative_ratio", cooperative_ratio))
            strategy_shift = abs(cooperative_ratio - prev_ratio)
        else:
            strategy_shift = 0.0

        stability = compute_stability_score(alive_ratio, firing_rate)
        information = compute_information_score(firing_rate, cooperative_ratio, strategy_shift)
        memory_score = compute_memory_score(mean_memory, strategy_shift)
        cost = compute_cost_score(firing_rate, spike_count, n)
        efficiency = compute_efficiency_score(stability, information, memory_score, cost)

        metrics = {
            "step": len(self.metrics_history),
            "alive_count": alive_count,
            "alive_ratio": alive_ratio,
            "spike_count": spike_count,
            "mean_v": mean_v,
            "firing_rate": firing_rate,
            "mean_memory": mean_memory,
            "cooperative_ratio": cooperative_ratio,
            "strategy_shift": strategy_shift,
            "stability_score": stability,
            "information_score": information,
            "memory_score": memory_score,
            "cost_score": cost,
            "efficiency_score": efficiency,
        }
        self.metrics_history.append(metrics)
        return metrics
