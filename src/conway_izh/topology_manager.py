"""Unified topology builder: merge several networks into a single sparse graph.

This module consumes user selections coming from the front-end (e.g.
``["small_world", "granule_test", "pyramidal_test"]``) and produces a
single :class:`UnifiedTopology` containing:

  * ``coords``   – (N, 3) float32 XYZ coordinates with per-component
                   spatial offsets so the networks do not visually overlap.
  * ``adjacency`` – (N, N) symmetric ``uint8`` SciPy CSR matrix built as a
                    block-diagonal of each component's adjacency.
  * ``components`` – metadata describing the offset/size of each component.

Why SciPy CSR everywhere
------------------------
The host machine has 8 GB of RAM. A dense (N, N) adjacency for the
combined networks (granule + medium spiny + pyramidal + a small world of
3600 nodes ≈ 8000 nodes) would already be ~64 MB just for ``bool`` and
~512 MB for ``float64``. With CSR we keep memory proportional to the
number of edges, which is O(N) for tree-shaped morphologies.

Time complexity for ``build_unified_topology``:
    O(sum_i (N_i + E_i))  – linear in the size of every selected component.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import sparse as sp

from conway_izh.swc_loader import (
    SWCMorphology,
    list_swc_files,
    normalize_morphology,
    parse_swc,
)


SMALL_WORLD_KEY = "small_world"
SMALL_WORLD_LABEL = "Mevcut Topoloji (Rastgele Small-World)"

# 3D space placement: alternating along the X axis. The spec asks for
# (+200, -200, +400, ...). Offsets are kept symmetric so the camera ends up
# centered on the origin.
DEFAULT_AXIS_OFFSET = 200.0


# --------------------------------------------------------------------------
# Component representation
# --------------------------------------------------------------------------


@dataclass
class TopologyComponent:
    """One sub-network inside the unified graph."""

    key: str
    label: str
    kind: str  # "small_world" or "swc"
    coords: np.ndarray  # (n, 3) float32 LOCAL coords (already normalized)
    adjacency: sp.csr_matrix  # (n, n) uint8
    offset: np.ndarray  # (3,) float32 placement applied at merge time
    color_hint: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    n_nodes: int = field(init=False)
    n_edges: int = field(init=False)

    def __post_init__(self) -> None:
        self.n_nodes = int(self.coords.shape[0])
        self.n_edges = int(self.adjacency.nnz // 2)


@dataclass
class UnifiedTopology:
    """Unified graph fed into :class:`GraphNeuralGrid`."""

    coords: np.ndarray  # (N, 3) float32 final coords (with offsets applied)
    adjacency: sp.csr_matrix  # (N, N) uint8 symmetric CSR
    components: List[Dict[str, object]]
    selection: List[str]
    edges_arr: np.ndarray  # (E, 2) int32 unique undirected edges
    bbox_min: np.ndarray  # (3,) float32
    bbox_max: np.ndarray  # (3,) float32

    @property
    def n_nodes(self) -> int:
        return int(self.coords.shape[0])

    @property
    def n_edges(self) -> int:
        return int(self.edges_arr.shape[0])


# --------------------------------------------------------------------------
# Component factories
# --------------------------------------------------------------------------


def _component_offset(slot: int, axis_step: float = DEFAULT_AXIS_OFFSET) -> np.ndarray:
    """Return ``(3,)`` float32 offset for the ``slot``-th selected component.

    Layout follows the spec: 1st centered, 2nd +X, 3rd -X, 4th +Z, 5th -Z, ...
    Use Z (not Y) so morphologies stay relatively flat when viewed from
    above, mimicking a coronal/sagittal plane.
    """
    if slot == 0:
        return np.zeros(3, dtype=np.float32)
    pair_index = (slot - 1) // 2  # 0,0,1,1,2,2,...
    sign = 1.0 if slot % 2 == 1 else -1.0
    axis = 0 if pair_index % 2 == 0 else 2  # X, X, Z, Z, X, X, ...
    out = np.zeros(3, dtype=np.float32)
    out[axis] = np.float32(sign * axis_step * (pair_index // 2 + 1))
    return out


def _build_small_world_component(
    n_nodes: int,
    *,
    k_neighbors: int,
    rewire_prob: float,
    seed: Optional[int],
    target_extent: float = 80.0,
) -> TopologyComponent:
    """Build a Watts-Strogatz component placed on a noisy 3D ring.

    The ring is positioned in the XY plane (Z slightly perturbed) so that
    visually it reads as a connected disk, contrasting with the dendritic
    SWC trees.

    Time:  O(n * k) edges, O(n * k) memory.
    """
    if n_nodes <= 1:
        coords = np.zeros((max(1, n_nodes), 3), dtype=np.float32)
        adj = sp.csr_matrix((max(1, n_nodes), max(1, n_nodes)), dtype=np.uint8)
        return TopologyComponent(
            key=SMALL_WORLD_KEY,
            label=SMALL_WORLD_LABEL,
            kind="small_world",
            coords=coords,
            adjacency=adj,
            offset=np.zeros(3, dtype=np.float32),
            color_hint=(0.49, 0.31, 0.78),
        )

    rng = np.random.default_rng(seed)
    k = max(2, min(k_neighbors, n_nodes - 1))
    if k % 2 == 1:
        k -= 1
    half = k // 2

    # Lay nodes on a ring in XY, gentle Z noise to avoid co-planarity.
    theta = np.linspace(0.0, 2.0 * np.pi, n_nodes, endpoint=False, dtype=np.float32)
    radius = np.float32(target_extent * 0.5)
    x = np.cos(theta) * radius
    y = np.sin(theta) * radius
    z = (rng.standard_normal(n_nodes).astype(np.float32) * np.float32(target_extent * 0.04))
    coords = np.stack([x, y, z], axis=1).astype(np.float32, copy=False)

    # Edge construction with rewiring; keep undirected uniqueness via sorted
    # tuples in a Python set — this is O(n*k) and acceptable for n ~ a few
    # thousand nodes.
    edge_set: set[Tuple[int, int]] = set()
    for i in range(n_nodes):
        for d in range(1, half + 1):
            j = (i + d) % n_nodes
            a, b = (i, j) if i < j else (j, i)
            edge_set.add((a, b))

    original = list(edge_set)
    for a, b in original:
        if float(rng.random()) >= rewire_prob:
            continue
        edge_set.discard((a, b))
        new_b = int(rng.integers(0, n_nodes))
        attempts = 0
        while attempts < 16 and (
            new_b == a
            or ((a, new_b) if a < new_b else (new_b, a)) in edge_set
        ):
            new_b = int(rng.integers(0, n_nodes))
            attempts += 1
        x_, y_ = (a, new_b) if a < new_b else (new_b, a)
        edge_set.add((x_, y_))

    if edge_set:
        u = np.fromiter((e[0] for e in edge_set), dtype=np.int64, count=len(edge_set))
        v = np.fromiter((e[1] for e in edge_set), dtype=np.int64, count=len(edge_set))
        rows = np.concatenate([u, v])
        cols = np.concatenate([v, u])
        data = np.ones(rows.shape[0], dtype=np.uint8)
        adj = sp.csr_matrix((data, (rows, cols)), shape=(n_nodes, n_nodes), dtype=np.uint8)
        adj.data[:] = 1
        adj.setdiag(0)
        adj.eliminate_zeros()
    else:
        adj = sp.csr_matrix((n_nodes, n_nodes), dtype=np.uint8)

    return TopologyComponent(
        key=SMALL_WORLD_KEY,
        label=SMALL_WORLD_LABEL,
        kind="small_world",
        coords=coords,
        adjacency=adj,
        offset=np.zeros(3, dtype=np.float32),
        color_hint=(0.49, 0.31, 0.78),
    )


def _build_swc_component(
    morpho: SWCMorphology,
    *,
    target_extent: float = 80.0,
    color_hint: Tuple[float, float, float] = (0.16, 0.94, 0.72),
) -> TopologyComponent:
    norm = normalize_morphology(morpho, target_extent=target_extent)
    return TopologyComponent(
        key=morpho.name,
        label=morpho.name,
        kind="swc",
        coords=norm.coords,
        adjacency=norm.adjacency,
        offset=np.zeros(3, dtype=np.float32),
        color_hint=color_hint,
    )


# --------------------------------------------------------------------------
# Discovery for the front-end
# --------------------------------------------------------------------------


@dataclass
class TopologyOption:
    key: str
    label: str
    kind: str
    n_nodes_hint: Optional[int] = None


def discover_topologies(swc_dir: Path | str) -> List[TopologyOption]:
    """Return the list of selectable topologies, including the small-world.

    The small-world is always first. SWC files are sorted alphabetically.
    """
    options: List[TopologyOption] = [
        TopologyOption(SMALL_WORLD_KEY, SMALL_WORLD_LABEL, "small_world")
    ]
    for path in list_swc_files(swc_dir):
        options.append(
            TopologyOption(key=path.stem, label=path.stem, kind="swc")
        )
    return options


# --------------------------------------------------------------------------
# Builder
# --------------------------------------------------------------------------


# Distinct, perceptually separated colors for each component slot.
_DEFAULT_PALETTE: List[Tuple[float, float, float]] = [
    (0.49, 0.31, 0.78),  # purple   – small world (default first slot)
    (0.16, 0.94, 0.72),  # teal     – granule
    (0.98, 0.55, 0.18),  # orange   – medium spiny
    (1.00, 0.84, 0.32),  # gold     – pyramidal
    (0.32, 0.68, 1.00),  # blue
    (0.93, 0.32, 0.62),  # pink
]


def build_unified_topology(
    selection: Sequence[str],
    *,
    swc_dir: Path | str,
    small_world_n: int = 3600,
    small_world_k: int = 6,
    small_world_rewire: float = 0.08,
    seed: Optional[int] = 42,
    axis_step: float = DEFAULT_AXIS_OFFSET,
    target_extent: float = 80.0,
) -> UnifiedTopology:
    """Build the unified topology for ``selection``.

    Args:
        selection: ordered keys to merge. Unknown keys are silently dropped
            so the front-end can ship a naive list. If the resulting list is
            empty we fall back to the small-world component to keep the UI
            functional.
        swc_dir: directory containing the candidate ``.swc`` files.

    Returns:
        :class:`UnifiedTopology` whose ``adjacency`` is block-diagonal
        (no inter-component edges by design – the spec wants the networks to
        live independently in space).
    """
    swc_dir = Path(swc_dir)
    swc_lookup: Dict[str, Path] = {p.stem: p for p in list_swc_files(swc_dir)}

    cleaned: List[str] = []
    for key in selection:
        key = str(key).strip()
        if not key:
            continue
        if key == SMALL_WORLD_KEY or key in swc_lookup:
            if key not in cleaned:
                cleaned.append(key)
    if not cleaned:
        cleaned = [SMALL_WORLD_KEY]

    components: List[TopologyComponent] = []
    for slot, key in enumerate(cleaned):
        color = _DEFAULT_PALETTE[slot % len(_DEFAULT_PALETTE)]
        if key == SMALL_WORLD_KEY:
            comp = _build_small_world_component(
                small_world_n,
                k_neighbors=small_world_k,
                rewire_prob=small_world_rewire,
                seed=seed,
                target_extent=target_extent,
            )
            comp.color_hint = color
        else:
            morpho = parse_swc(swc_lookup[key])
            comp = _build_swc_component(
                morpho, target_extent=target_extent, color_hint=color
            )
        comp.offset = _component_offset(slot, axis_step=axis_step)
        components.append(comp)

    # Concatenate coords with per-component offset applied.
    coord_chunks: List[np.ndarray] = []
    component_meta: List[Dict[str, object]] = []
    cursor = 0
    for comp in components:
        shifted = comp.coords + comp.offset  # broadcast (n,3) + (3,)
        coord_chunks.append(shifted.astype(np.float32, copy=False))
        component_meta.append(
            {
                "key": comp.key,
                "label": comp.label,
                "kind": comp.kind,
                "start": int(cursor),
                "n_nodes": int(comp.n_nodes),
                "n_edges": int(comp.n_edges),
                "offset": [float(comp.offset[0]), float(comp.offset[1]), float(comp.offset[2])],
                "color": [float(comp.color_hint[0]), float(comp.color_hint[1]), float(comp.color_hint[2])],
            }
        )
        cursor += comp.n_nodes

    if coord_chunks:
        coords = np.concatenate(coord_chunks, axis=0).astype(np.float32, copy=False)
    else:
        coords = np.zeros((0, 3), dtype=np.float32)

    # Block-diagonal merge of adjacency matrices preserves index space and
    # keeps the merged graph sparse. Re-cast to CSR + uint8 to be safe.
    adj_blocks: List[sp.csr_matrix] = [c.adjacency for c in components]
    if adj_blocks:
        merged = sp.block_diag(adj_blocks, format="csr").astype(np.uint8, copy=False)
    else:
        merged = sp.csr_matrix((0, 0), dtype=np.uint8)

    # Pre-compute the unique edge list (E, 2) as int32 for downstream code
    # (graph_trace_to_drive, three.js segments). Use the upper triangle to
    # avoid duplicate (i,j) / (j,i) pairs.
    upper = sp.triu(merged, k=1).tocoo()
    if upper.nnz:
        edges_arr = np.stack(
            [upper.row.astype(np.int32), upper.col.astype(np.int32)], axis=1
        )
    else:
        edges_arr = np.zeros((0, 2), dtype=np.int32)

    if coords.size:
        bbox_min = coords.min(axis=0).astype(np.float32, copy=False)
        bbox_max = coords.max(axis=0).astype(np.float32, copy=False)
    else:
        bbox_min = np.zeros(3, dtype=np.float32)
        bbox_max = np.zeros(3, dtype=np.float32)

    return UnifiedTopology(
        coords=coords,
        adjacency=merged,
        components=component_meta,
        selection=list(cleaned),
        edges_arr=edges_arr,
        bbox_min=bbox_min,
        bbox_max=bbox_max,
    )
