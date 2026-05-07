"""Parsing utilities for SWC neuronal morphology files.

The SWC format encodes a single neuronal tree as one row per sampled point:

    n  type  x  y  z  radius  parent

where ``parent`` is the row id of the parent point (or ``-1`` for the root).
We do not need biophysical fidelity; the file is consumed strictly as a
spatial graph. Coordinates are kept as ``float32`` to halve RAM versus
``float64`` (we have 8 GB of RAM so every byte counts), and the
adjacency is built directly as a SciPy CSR matrix to avoid materialising
any dense (N, N) buffer.

Memory note (Big-O):
    * Parsing                : O(N) lines, O(N) coords + O(N) parent edges.
    * Adjacency construction : O(N) non-zeros, O(N) memory in CSR form.
    * Returned ``adjacency`` : symmetric, no self loops, dtype uint8 to keep
      one byte per non-zero.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
from scipy import sparse as sp


SWC_DTYPE = np.dtype(
    [
        ("id", np.int64),
        ("type", np.int8),
        ("x", np.float32),
        ("y", np.float32),
        ("z", np.float32),
        ("radius", np.float32),
        ("parent", np.int64),
    ]
)


@dataclass
class SWCMorphology:
    """In-memory view of a parsed SWC file.

    Attributes:
        name: Logical short name (without .swc) used for display.
        coords: ``(N, 3)`` float32 XYZ coordinates.
        radii:  ``(N,)`` float32 sample radius.
        types:  ``(N,)`` int8 SWC structural type (1 soma, 2 axon, 3 basal,
                4 apical, ...). Carried mainly for downstream coloring.
        adjacency: ``(N, N)`` symmetric ``uint8`` CSR adjacency matrix
                   describing parent->child edges only (no self loops).
        source: Absolute path the file was read from (debug/audit).
    """

    name: str
    coords: np.ndarray
    radii: np.ndarray
    types: np.ndarray
    adjacency: sp.csr_matrix
    source: Path

    @property
    def n_nodes(self) -> int:
        return int(self.coords.shape[0])

    @property
    def n_edges(self) -> int:
        # Adjacency is symmetric so divide by two.
        return int(self.adjacency.nnz // 2)


def _iter_swc_records(path: Path) -> Iterable[Tuple[int, int, float, float, float, float, int]]:
    """Yield validated tuples for every non-comment, non-empty SWC line.

    The function is tolerant to trailing whitespace and varying indentation,
    which both occur in NeuroMorpho-derived files.
    """
    with path.open("r", encoding="utf-8", errors="ignore") as fh:
        for line_no, raw in enumerate(fh, start=1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            tokens = line.split()
            if len(tokens) < 7:
                # Skip malformed rows quietly; we trust the rest of the file.
                continue
            try:
                yield (
                    int(tokens[0]),
                    int(tokens[1]),
                    float(tokens[2]),
                    float(tokens[3]),
                    float(tokens[4]),
                    float(tokens[5]),
                    int(tokens[6]),
                )
            except ValueError:
                # Same reasoning: ignore a single corrupt line, keep going.
                continue


def parse_swc(path: Path | str, *, name: Optional[str] = None) -> SWCMorphology:
    """Parse a single ``.swc`` file into a :class:`SWCMorphology`.

    Edges are built from the parent column only, so the resulting graph is a
    rooted tree (one weakly connected component). The returned adjacency
    matrix is symmetric and has 0/1 ``uint8`` values.

    Time:  O(N)
    Space: O(N) (plus three tiny COO buffers, also O(N)).
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"SWC file not found: {path}")

    rows = list(_iter_swc_records(path))
    if not rows:
        raise ValueError(f"SWC file is empty or unparseable: {path}")

    n = len(rows)
    coords = np.empty((n, 3), dtype=np.float32)
    radii = np.empty(n, dtype=np.float32)
    types = np.empty(n, dtype=np.int8)
    parent_of = np.empty(n, dtype=np.int64)
    id_to_idx: dict[int, int] = {}

    for i, (sid, stype, x, y, z, r, parent) in enumerate(rows):
        coords[i, 0] = x
        coords[i, 1] = y
        coords[i, 2] = z
        radii[i] = max(float(r), 0.0)
        types[i] = max(min(int(stype), 127), -128)
        parent_of[i] = parent
        id_to_idx[sid] = i

    # Build COO buffers for parent->child edges only; drop root rows (-1) and
    # any references that point to a node we never saw.
    edge_u: List[int] = []
    edge_v: List[int] = []
    for i in range(n):
        p = int(parent_of[i])
        if p < 0:
            continue
        j = id_to_idx.get(p)
        if j is None or j == i:
            continue
        edge_u.append(i)
        edge_v.append(j)

    if edge_u:
        u = np.asarray(edge_u, dtype=np.int64)
        v = np.asarray(edge_v, dtype=np.int64)
        # Symmetrize once; CSR construction will sum duplicates (none expected).
        rows_idx = np.concatenate([u, v])
        cols_idx = np.concatenate([v, u])
        data = np.ones(rows_idx.shape[0], dtype=np.uint8)
        adjacency = sp.csr_matrix(
            (data, (rows_idx, cols_idx)), shape=(n, n), dtype=np.uint8
        )
        # Defensive: clamp >1 (none should appear) and drop accidental self loops.
        adjacency.data[:] = 1
        adjacency.setdiag(0)
        adjacency.eliminate_zeros()
    else:
        adjacency = sp.csr_matrix((n, n), dtype=np.uint8)

    return SWCMorphology(
        name=name or path.stem,
        coords=coords,
        radii=radii,
        types=types,
        adjacency=adjacency,
        source=path.resolve(),
    )


def list_swc_files(directory: Path | str) -> List[Path]:
    """Return all ``*.swc`` files under ``directory`` (non-recursive, sorted)."""
    directory = Path(directory)
    if not directory.is_dir():
        return []
    return sorted(p for p in directory.iterdir() if p.is_file() and p.suffix.lower() == ".swc")


def normalize_morphology(
    morpho: SWCMorphology, *, target_extent: float = 80.0
) -> SWCMorphology:
    """Return a copy of ``morpho`` whose XYZ coordinates are centered at the
    origin and rescaled so the largest extent matches ``target_extent``.

    SWC files arrive with µm units that vary wildly across cells (a small
    granule cell is < 50 µm, a pyramidal apical dendrite easily passes 500
    µm). Without normalisation the camera would have to fight extreme scale
    differences when several morphologies live in the same scene.

    Time:  O(N)
    Space: O(N) (new ``coords`` buffer, everything else is shared).
    """
    coords = morpho.coords
    if coords.size == 0:
        return morpho

    centered = coords - coords.mean(axis=0, dtype=np.float32)
    extent = float(np.max(np.ptp(centered, axis=0)))
    if extent <= 1e-6:
        scale = 1.0
    else:
        scale = float(target_extent) / extent
    rescaled = (centered * np.float32(scale)).astype(np.float32, copy=False)

    return SWCMorphology(
        name=morpho.name,
        coords=rescaled,
        radii=morpho.radii,
        types=morpho.types,
        adjacency=morpho.adjacency,
        source=morpho.source,
    )
