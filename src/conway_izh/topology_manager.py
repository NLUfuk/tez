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
LEGACY_CLUSTER_KEY = "legacy_cluster"
LEGACY_CLUSTER_LABEL = "Klasik Topoloji (Legacy Cluster)"

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


def _dedup_undirected_edges(
    a: np.ndarray, b: np.ndarray, n: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Deduplicate undirected edges via a single integer key ``a*n + b``.

    Accepts arrays whose entries are NOT guaranteed to satisfy ``a < b``;
    the function reorders them first. ``int64`` is sufficient as long as
    ``n < 2**31`` (we have ~10^4 nodes at most). Returns sorted, unique
    ``(a, b)`` pairs as ``int64``. Time: O(E log E).
    """
    if a.size == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    a = a.astype(np.int64, copy=False)
    b = b.astype(np.int64, copy=False)
    lo = np.minimum(a, b)
    hi = np.maximum(a, b)
    keep = lo != hi
    lo = lo[keep]
    hi = hi[keep]
    if lo.size == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    keys = lo * np.int64(n) + hi
    keys = np.unique(keys)
    out_a = (keys // np.int64(n)).astype(np.int64, copy=False)
    out_b = (keys % np.int64(n)).astype(np.int64, copy=False)
    return out_a, out_b


def _csr_from_undirected(
    a: np.ndarray, b: np.ndarray, n: int
) -> sp.csr_matrix:
    """Build a symmetric ``uint8`` CSR adjacency from undirected edges."""
    if a.size == 0:
        return sp.csr_matrix((n, n), dtype=np.uint8)
    rows = np.concatenate([a, b])
    cols = np.concatenate([b, a])
    data = np.ones(rows.shape[0], dtype=np.uint8)
    adj = sp.csr_matrix((data, (rows, cols)), shape=(n, n), dtype=np.uint8)
    adj.data[:] = 1
    adj.setdiag(0)
    adj.eliminate_zeros()
    return adj


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

    The construction is fully vectorised: the ring edge set, the rewire
    mask and the destination resampling are all NumPy operations, then
    duplicates are collapsed via a single ``unique`` call on the integer
    key ``a*n + b``. This replaces the previous pure-Python double loop
    that dominated start-up time for n in the thousands.

    Time:  O(n * k) NumPy ops + O(E log E) for the unique-edge collapse.
    Memory: O(n * k).
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

    if half == 0:
        return TopologyComponent(
            key=SMALL_WORLD_KEY,
            label=SMALL_WORLD_LABEL,
            kind="small_world",
            coords=coords,
            adjacency=sp.csr_matrix((n_nodes, n_nodes), dtype=np.uint8),
            offset=np.zeros(3, dtype=np.float32),
            color_hint=(0.49, 0.31, 0.78),
        )

    # Initial ring: each i connects to (i+1)..(i+half) mod n. Shape: (n*half,)
    i_idx = np.repeat(np.arange(n_nodes, dtype=np.int64), half)
    d_off = np.tile(np.arange(1, half + 1, dtype=np.int64), n_nodes)
    j_idx = (i_idx + d_off) % np.int64(n_nodes)

    if rewire_prob > 0.0:
        # One Bernoulli sample per edge; vectorised rewire to a random target.
        mask = rng.random(i_idx.shape[0]) < float(rewire_prob)
        m_count = int(mask.sum())
        if m_count > 0:
            new_b = rng.integers(0, n_nodes, size=m_count).astype(np.int64)
            anchor = i_idx[mask]
            # Avoid self-loop: bump colliding targets by 1 (still uniform mod n).
            collide = new_b == anchor
            if collide.any():
                new_b[collide] = (new_b[collide] + 1) % np.int64(n_nodes)
            j_idx[mask] = new_b

    a_arr, b_arr = _dedup_undirected_edges(i_idx, j_idx, n_nodes)
    adj = _csr_from_undirected(a_arr, b_arr, n_nodes)

    return TopologyComponent(
        key=SMALL_WORLD_KEY,
        label=SMALL_WORLD_LABEL,
        kind="small_world",
        coords=coords,
        adjacency=adj,
        offset=np.zeros(3, dtype=np.float32),
        color_hint=(0.49, 0.31, 0.78),
    )


def _build_legacy_cluster_component(
    n_nodes: int,
    *,
    seed: Optional[int],
    target_extent: float = 80.0,
) -> TopologyComponent:
    """Build a legacy-like hollow shell (dense border + sparse inner cavity).

    Reference look is a glowing "cube/ring" shell with many near-neighbor
    surface links and a controlled set of long-range chords crossing the
    empty interior. The point cloud and every edge family (k-NN, distance-
    weighted Bernoulli, long-range chords) are now constructed with
    NumPy vector ops; the previous O(n^2) Python double loop dominated
    start-up time (2-5 s at n~1000) and is replaced by a single batched
    Bernoulli sample over the upper triangle plus a one-shot ``unique``.

    Memory: the transient ``(n, n)`` distance matrix is the floor.
    For ``n <= 1800`` (max we allow at the call site) this is ~12 MB
    float32 — well below the 8 GB RAM target.

    Time:  O(n^2) NumPy ops, no Python-level iteration.
    """
    n = max(80, int(n_nodes))
    rng = np.random.default_rng(seed)

    shell_half = np.float32(target_extent * 0.65)
    face_eps = np.float32(target_extent * 0.11)
    jitter = np.float32(target_extent * 0.035)

    coords = np.zeros((n, 3), dtype=np.float32)
    shell_mask = np.zeros(n, dtype=bool)
    corner_mask = np.zeros(n, dtype=bool)

    corner_count = max(8, int(round(n * 0.11)))
    shell_count = max(corner_count + 20, int(round(n * 0.78)))
    shell_count = min(shell_count, n)

    corners = np.array(
        [
            [-1.0, -1.0, -1.0],
            [-1.0, -1.0, 1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, 1.0, 1.0],
            [1.0, -1.0, -1.0],
            [1.0, -1.0, 1.0],
            [1.0, 1.0, -1.0],
            [1.0, 1.0, 1.0],
        ],
        dtype=np.float32,
    )

    # ---- corner blobs (vectorised) ---------------------------------------
    if corner_count > 0:
        idx = np.arange(corner_count) % 8
        base = corners[idx] * shell_half
        noise = rng.normal(
            0.0, float(face_eps * 0.52), size=(corner_count, 3)
        ).astype(np.float32, copy=False)
        coords[:corner_count] = base + noise
        shell_mask[:corner_count] = True
        corner_mask[:corner_count] = True

    # ---- shell population (vectorised) -----------------------------------
    shell_extra = shell_count - corner_count
    if shell_extra > 0:
        p = rng.uniform(
            -float(shell_half), float(shell_half), size=(shell_extra, 3)
        ).astype(np.float32, copy=False)
        axes = rng.integers(0, 3, size=shell_extra)
        signs = np.where(
            rng.random(shell_extra) < 0.5, np.float32(1.0), np.float32(-1.0)
        )
        # Snap the chosen axis component to ±shell_half ± face_eps.
        face_noise = rng.normal(0.0, float(face_eps), size=shell_extra).astype(
            np.float32, copy=False
        )
        rows_idx = np.arange(shell_extra)
        p[rows_idx, axes] = signs * shell_half + face_noise
        p += rng.normal(0.0, float(jitter), size=(shell_extra, 3)).astype(
            np.float32, copy=False
        )
        np.clip(p, -float(shell_half) * 1.15, float(shell_half) * 1.15, out=p)
        coords[corner_count:shell_count] = p
        shell_mask[corner_count:shell_count] = True

    # ---- background haze (vectorised) ------------------------------------
    haze_extra = n - shell_count
    if haze_extra > 0:
        coords[shell_count:] = rng.normal(
            0.0, float(shell_half * 0.35), size=(haze_extra, 3)
        ).astype(np.float32, copy=False)

    # ---- distance matrix once --------------------------------------------
    # We deliberately use plain NumPy broadcasting instead of
    # ``scipy.spatial.distance.cdist`` here. ``cdist`` would be marginally
    # faster *per call*, but importing ``scipy.spatial`` triggers loading
    # qhull / cKDTree C extensions, adding ~2 s to the FIRST topology load
    # in a fresh Python process. That single delay (visible right after
    # the user clicks "Topolojileri Yukle ve Calistir") is exactly what
    # the visible "geç açılıyor" lag was. Broadcasting allocates an
    # (n, n, 3) float32 intermediate; at n <= 1800 that is <= 38 MB,
    # comfortably within our 8 GB budget.
    delta = coords[:, None, :] - coords[None, :, :]
    dist = np.linalg.norm(delta, axis=2).astype(np.float32, copy=False)
    del delta  # release the (n, n, 3) intermediate immediately
    np.fill_diagonal(dist, np.float32(np.inf))

    # ---- k-NN edges (vectorised) -----------------------------------------
    k_nn = min(5, n - 1)
    if k_nn > 0:
        knn = np.argpartition(dist, k_nn, axis=1)[:, :k_nn]  # (n, k_nn)
        ii = np.repeat(np.arange(n, dtype=np.int64), k_nn)
        jj = knn.ravel().astype(np.int64, copy=False)
        a_knn = np.minimum(ii, jj)
        b_knn = np.maximum(ii, jj)
        keep = a_knn != b_knn
        a_knn = a_knn[keep]
        b_knn = b_knn[keep]
    else:
        a_knn = np.zeros(0, dtype=np.int64)
        b_knn = np.zeros(0, dtype=np.int64)

    # ---- distance-weighted Bernoulli edges over the upper triangle -------
    iu, ju = np.triu_indices(n, k=1)
    d_pair = dist[iu, ju]
    dist_scale = float(target_extent * 0.26)
    base_p = 0.18 * np.exp(-d_pair.astype(np.float64, copy=False) / dist_scale)

    si = shell_mask[iu]
    sj = shell_mask[ju]
    ci = corner_mask[iu]
    cj = corner_mask[ju]
    both_shell = si & sj
    neither_shell = (~si) & (~sj)
    # XOR captures the "exactly one is shell" case; equivalent to the
    # original ``else`` branch in the Python loop.
    one_shell = si ^ sj

    factor = np.ones_like(base_p)
    factor = np.where(both_shell, 1.6, factor)
    factor = np.where(neither_shell, 0.55, factor)
    factor = np.where(one_shell, 0.80, factor)
    factor = np.where(ci | cj, factor * 1.35, factor)
    p_pair = base_p * factor

    sample = rng.random(p_pair.shape[0])
    sel = sample < p_pair
    a_prob = iu[sel].astype(np.int64, copy=False)
    b_prob = ju[sel].astype(np.int64, copy=False)

    # ---- long-range chords through the cavity (vectorised) --------------
    extra_long = max(8, int(round(n * 0.07)))
    if shell_count >= 2 and extra_long > 0:
        # Oversample then filter by distance; this avoids a Python retry
        # loop while still respecting the >0.85 * extent gate.
        over = max(extra_long * 8, 32)
        ar = rng.integers(0, shell_count, size=over).astype(np.int64)
        br = rng.integers(0, shell_count, size=over).astype(np.int64)
        diff = ar != br
        ar = ar[diff]
        br = br[diff]
        if ar.size:
            long_thresh = np.float32(target_extent * 0.85)
            d_lr = dist[ar, br]
            keep_lr = d_lr >= long_thresh
            ar = ar[keep_lr]
            br = br[keep_lr]
        if ar.size > extra_long:
            ar = ar[:extra_long]
            br = br[:extra_long]
        a_lr = np.minimum(ar, br)
        b_lr = np.maximum(ar, br)
    else:
        a_lr = np.zeros(0, dtype=np.int64)
        b_lr = np.zeros(0, dtype=np.int64)

    # ---- merge + dedup ---------------------------------------------------
    all_a = np.concatenate([a_knn, a_prob, a_lr])
    all_b = np.concatenate([b_knn, b_prob, b_lr])
    a_arr, b_arr = _dedup_undirected_edges(all_a, all_b, n)
    adj = _csr_from_undirected(a_arr, b_arr, n)

    return TopologyComponent(
        key=LEGACY_CLUSTER_KEY,
        label=LEGACY_CLUSTER_LABEL,
        kind="legacy",
        coords=coords,
        adjacency=adj,
        offset=np.zeros(3, dtype=np.float32),
        color_hint=(0.36, 0.78, 1.0),
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
        TopologyOption(SMALL_WORLD_KEY, SMALL_WORLD_LABEL, "small_world"),
        TopologyOption(LEGACY_CLUSTER_KEY, LEGACY_CLUSTER_LABEL, "legacy"),
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
        if key in (SMALL_WORLD_KEY, LEGACY_CLUSTER_KEY) or key in swc_lookup:
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
        elif key == LEGACY_CLUSTER_KEY:
            legacy_n = max(280, min(1800, int(round(small_world_n * 0.26))))
            comp = _build_legacy_cluster_component(
                legacy_n,
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
