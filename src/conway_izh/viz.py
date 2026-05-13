"""Visualization and stream publishing for Conway-Izhikevich simulation.

Hot-path note: stream mode never touches matplotlib or imageio. Keeping
those imports module-level used to add ~0.7-1.5 s to cold start because
matplotlib eagerly builds its font cache and imageio probes its plugin
registry. The heavy imports are now lazy: they are paid only when a save
function is actually called (i.e. ``run_grid.py`` or notebook flows).
"""

import json
import threading
from collections import deque
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Callable, Deque, Dict, List, Optional, Sequence, Tuple

import numpy as np

from conway_izh.conway import initialize_conway
from conway_izh.efficiency import (
    compute_cost_score,
    compute_efficiency_score,
    compute_information_score,
    compute_memory_score,
    compute_stability_score,
)
from conway_izh.izhikevich import initialize_izhikevich
from conway_izh.topology_manager import (
    LEGACY_CLUSTER_KEY,
    SMALL_WORLD_KEY,
    UnifiedTopology,
    build_unified_topology,
    discover_topologies,
)


def save_gol_frame(state: np.ndarray, filepath: Path, cmap: str = 'gray'):
    """
    Save Conway Game of Life state as PNG.

    Args:
        state: Binary grid (H, W)
        filepath: Output file path
        cmap: Colormap name
    """
    import matplotlib.pyplot as plt  # lazy: skip in stream mode

    plt.figure(figsize=(10, 10))
    plt.imshow(state, cmap=cmap, interpolation='nearest')
    plt.title('Conway Game of Life State')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(filepath, dpi=100, bbox_inches='tight')
    plt.close()


def save_v_heatmap(v: np.ndarray, filepath: Path, cmap: str = 'viridis'):
    """
    Save membrane potential as heatmap.

    Args:
        v: Membrane potential array (H, W)
        filepath: Output file path
        cmap: Colormap name
    """
    import matplotlib.pyplot as plt  # lazy: skip in stream mode

    plt.figure(figsize=(10, 10))
    plt.imshow(v, cmap=cmap, interpolation='nearest')
    plt.colorbar(label='Membrane Potential (mV)')
    plt.title('Membrane Potential Heatmap')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(filepath, dpi=100, bbox_inches='tight')
    plt.close()


def save_spike_raster(spike_history: List[np.ndarray], filepath: Path):
    """
    Save spike raster plot (time x cell index).

    Args:
        spike_history: List of spike masks (H, W) bool arrays
        filepath: Output file path
    """
    if not spike_history:
        return

    import matplotlib.pyplot as plt  # lazy: skip in stream mode

    H, W = spike_history[0].shape
    N = H * W
    T = len(spike_history)

    # Flatten to (T, N) raster
    raster = np.zeros((T, N), dtype=bool)
    for t, spikes in enumerate(spike_history):
        raster[t, :] = spikes.flatten()

    plt.figure(figsize=(12, 8))
    # Plot only where spikes occur
    spike_times, spike_indices = np.where(raster)
    plt.scatter(spike_times, spike_indices, s=0.5, c='black', alpha=0.6)
    plt.xlabel('Time Step')
    plt.ylabel('Cell Index')
    plt.title('Spike Raster Plot')
    plt.tight_layout()
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()


def save_metrics_csv(metrics_list: List[dict], filepath: Path):
    """
    Save metrics to CSV file.

    Args:
        metrics_list: List of metric dictionaries
        filepath: Output CSV file path
    """
    import csv

    if not metrics_list:
        return

    with open(filepath, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=metrics_list[0].keys())
        writer.writeheader()
        writer.writerows(metrics_list)


def create_gif(frame_paths: List[Path], output_path: Path, duration: float = 0.1):
    """
    Create GIF from frame images.

    Args:
        frame_paths: List of image file paths
        output_path: Output GIF path
        duration: Frame duration in seconds
    """
    import imageio  # lazy: only needed when actually writing a GIF

    images = []
    for path in frame_paths:
        if path.exists():
            images.append(imageio.imread(path))

    if images:
        imageio.mimsave(output_path, images, duration=duration)


def _clamp_rules(values: Sequence[int]) -> Tuple[int, ...]:
    cleaned = sorted({int(v) for v in values if 0 <= int(v) <= 8})
    return tuple(cleaned) if cleaned else (3,)


class StreamBridge:
    """Thread-safe bridge between simulation loop and HTTP control plane."""

    def __init__(self):
        self._lock = threading.Lock()
        self._latest_state: Optional[Dict] = None
        self._controls: Deque[Dict] = deque()

    def set_state(self, payload: Dict):
        with self._lock:
            self._latest_state = payload

    def get_state(self) -> Optional[Dict]:
        with self._lock:
            if self._latest_state is None:
                return None
            return dict(self._latest_state)

    def push_control(self, payload: Dict):
        with self._lock:
            self._controls.append(payload)

    def pop_controls(self) -> List[Dict]:
        with self._lock:
            out = list(self._controls)
            self._controls.clear()
            return out


def create_small_world_graph(
    node_count: int,
    k_neighbors: int = 6,
    rewire_prob: float = 0.08,
    seed: Optional[int] = None,
) -> List[Tuple[int, int]]:
    """Create undirected Watts-Strogatz style edge list."""
    if node_count <= 1:
        return []
    k = max(2, min(k_neighbors, node_count - 1))
    if k % 2 == 1:
        k -= 1
    rng = np.random.default_rng(seed)
    half = k // 2
    edge_set = set()

    for i in range(node_count):
        for d in range(1, half + 1):
            j = (i + d) % node_count
            a, b = (i, j) if i < j else (j, i)
            edge_set.add((a, b))

    original = list(edge_set)
    for a, b in original:
        if float(rng.random()) >= rewire_prob:
            continue
        edge_set.discard((a, b))
        new_b = int(rng.integers(0, node_count))
        while new_b == a or ((a, new_b) if a < new_b else (new_b, a)) in edge_set:
            new_b = int(rng.integers(0, node_count))
        x, y = (a, new_b) if a < new_b else (new_b, a)
        edge_set.add((x, y))

    return list(edge_set)


def build_control_handler(bridge: StreamBridge, *, swc_dir: Optional[Path] = None):
    """Build an HTTP handler bound to a bridge instance.

    If ``swc_dir`` is supplied the handler exposes ``GET /topologies``
    listing the available networks (small-world + every ``.swc`` file).
    """
    swc_path = Path(swc_dir).resolve() if swc_dir is not None else None

    class StreamHandler(BaseHTTPRequestHandler):
        def _write_json(self, status: int, payload: Dict):
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.end_headers()
            self.wfile.write(body)

        def do_OPTIONS(self):
            self._write_json(HTTPStatus.OK, {"ok": True})

        def do_GET(self):
            if self.path == "/state":
                payload = bridge.get_state()
                if payload is None:
                    self._write_json(HTTPStatus.OK, {"ready": False})
                    return
                self._write_json(HTTPStatus.OK, payload)
                return
            if self.path == "/topologies":
                if swc_path is None:
                    self._write_json(HTTPStatus.OK, {
                        "options": [
                            {
                                "key": SMALL_WORLD_KEY,
                                "label": "Mevcut Topoloji (Rastgele Small-World)",
                                "kind": "small_world",
                            },
                            {
                                "key": LEGACY_CLUSTER_KEY,
                                "label": "Klasik Topoloji (Legacy Cluster)",
                                "kind": "legacy",
                            },
                        ],
                    })
                    return
                opts = discover_topologies(swc_path)
                self._write_json(HTTPStatus.OK, {
                    "options": [
                        {"key": o.key, "label": o.label, "kind": o.kind}
                        for o in opts
                    ],
                })
                return
            self._write_json(HTTPStatus.NOT_FOUND, {"error": "Not Found"})

        def do_POST(self):
            if self.path != "/control":
                self._write_json(HTTPStatus.NOT_FOUND, {"error": "Not Found"})
                return
            try:
                content_len = int(self.headers.get("Content-Length", "0"))
                body = self.rfile.read(content_len).decode("utf-8")
                data = json.loads(body) if body else {}
            except Exception:
                self._write_json(HTTPStatus.BAD_REQUEST, {"error": "Invalid payload"})
                return
            action = str(data.get("action", "")).strip()
            if not action:
                self._write_json(HTTPStatus.BAD_REQUEST, {"error": "Missing action"})
                return
            bridge.push_control(data)
            self._write_json(HTTPStatus.OK, {"ok": True})

        def log_message(self, _fmt, *_args):
            return

    return StreamHandler


class StreamServer:
    """Thin wrapper around HTTP server lifecycle."""

    def __init__(
        self,
        bridge: StreamBridge,
        host: str,
        port: int,
        *,
        swc_dir: Optional[Path] = None,
    ):
        self._bridge = bridge
        self._server = ThreadingHTTPServer(
            (host, port), build_control_handler(bridge, swc_dir=swc_dir)
        )
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    def start(self):
        self._thread.start()

    def stop(self):
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=1.0)


class StreamPublisher:
    """Publishes simulation state and applies live control updates.

    The publisher is grid-agnostic: it can wrap either a classic
    :class:`conway_izh.grid.NeuralGrid` (rectangular ``H × W`` mode) or a
    :class:`conway_izh.graph_grid.GraphNeuralGrid` (arbitrary topology
    mode). It exposes :py:meth:`set_grid` so the host loop can hot-swap
    the active engine when the front-end issues a ``set_topology``
    control. The wire payload always mirrors whichever grid is currently
    attached.
    """

    def __init__(
        self,
        bridge: StreamBridge,
        grid,
        *,
        swc_dir: Optional[Path] = None,
        topology: Optional[UnifiedTopology] = None,
    ):
        """``grid`` may be a ``NeuralGrid`` or ``GraphNeuralGrid``.

        ``swc_dir`` is required to support hot-swap to topology mode.
        """
        self.bridge = bridge
        self.swc_dir: Optional[Path] = (
            Path(swc_dir).resolve() if swc_dir is not None else None
        )
        self.generation_period_ms = 50
        self.paused = False
        self.graph_version = 1
        self._sent_graph_payload_version = -1
        self._sent_coords_version = -1
        self.topology: Optional[UnifiedTopology] = topology
        self.current_selection: List[str] = []
        self._pending_grid = None  # set by set_grid; main loop swaps in
        # Per-component scalar memory used to compute strategy_shift
        # locally (see ``_per_component_metrics``). The map is keyed by the
        # 1-based slot index so re-orderings of the selection invalidate the
        # cache automatically.
        self._per_component_prev: Dict[int, Dict[str, float]] = {}
        self._set_grid_internal(grid)

    # -- attachment ----------------------------------------------------------

    def _set_grid_internal(self, grid) -> None:
        """Bind to a new grid object and reset cached metadata."""
        self.grid = grid
        if hasattr(grid, "topology") and grid.topology is not None:
            self.topology = grid.topology
            self.node_count = int(grid.n_nodes)
            self.width = int(grid.n_nodes)
            self.height = 1
            self.current_selection = list(self.topology.selection)
            self.is_topology_mode = True
        else:
            self.topology = None
            self.width = int(grid.config.width)
            self.height = int(grid.config.height)
            self.node_count = self.width * self.height
            self.current_selection = [SMALL_WORLD_KEY]
            self.is_topology_mode = False
        self.birth_neighbors: Tuple[int, ...] = tuple(grid.config.birth_neighbors)
        self.survive_neighbors: Tuple[int, ...] = tuple(grid.config.survive_neighbors)
        self.graph_version += 1
        self._sent_graph_payload_version = -1
        self._sent_coords_version = -1
        # Drop cached per-component history; component layout may have shifted.
        self._per_component_prev = {}

    def consume_pending_grid(self):
        """Return any grid queued by :py:meth:`set_grid` and clear the slot.

        The simulation loop should call this once per step; if a new grid is
        returned the loop must replace its local reference *before* invoking
        ``step()`` to avoid mixing state across topologies.
        """
        if self._pending_grid is None:
            return None
        next_grid = self._pending_grid
        self._pending_grid = None
        self._set_grid_internal(next_grid)
        return next_grid

    def apply_controls(
        self,
        grid,
        set_manual_spike: Callable[[int], None],
        set_toggle_conway: Callable[[int], None],
    ):
        for control in self.bridge.pop_controls():
            action = str(control.get("action", "")).strip()
            if action == "manual_spike":
                idx = int(control.get("index", -1))
                if 0 <= idx < self.node_count:
                    set_manual_spike(idx)
            elif action == "toggle_conway":
                idx = int(control.get("index", -1))
                if 0 <= idx < self.node_count:
                    set_toggle_conway(idx)
            elif action == "set_rules":
                birth = control.get("birth", [3])
                survive = control.get("survive", [2, 3])
                self.birth_neighbors = _clamp_rules(birth)
                self.survive_neighbors = _clamp_rules(survive)
                grid.config.birth_neighbors = self.birth_neighbors
                grid.config.survive_neighbors = self.survive_neighbors
            elif action == "set_generation_ms":
                raw = int(control.get("generation_ms", self.generation_period_ms))
                self.generation_period_ms = max(1, min(1000, raw))
            elif action == "set_paused":
                self.paused = bool(control.get("paused", False))
                live_state = self.bridge.get_state()
                if live_state is not None:
                    live_state["paused"] = bool(self.paused)
                    self.bridge.set_state(live_state)
            elif action == "toggle_pause":
                self.paused = not self.paused
                live_state = self.bridge.get_state()
                if live_state is not None:
                    live_state["paused"] = bool(self.paused)
                    self.bridge.set_state(live_state)
            elif action == "set_coupling":
                c = grid.config.coupling
                if "k_alive" in control:
                    c.k_alive = float(np.clip(float(control["k_alive"]), 0.0, 20.0))
                if "k_neighbors" in control:
                    c.k_neighbors = float(np.clip(float(control["k_neighbors"]), 0.0, 5.0))
                if "bias" in control:
                    c.bias = float(np.clip(float(control["bias"]), -5.0, 5.0))
                if "feedback_enabled" in control:
                    c.feedback_enabled = bool(control["feedback_enabled"])
                if "k_syn" in control:
                    c.k_syn = float(np.clip(float(control["k_syn"]), 0.0, 20.0))
                if "spike_trace_decay" in control:
                    c.spike_trace_decay = float(
                        np.clip(float(control["spike_trace_decay"]), 0.0, 0.9999)
                    )
                if "feedback_graph_neighbors" in control:
                    c.feedback_graph_neighbors = bool(control["feedback_graph_neighbors"])
            elif action == "reset_grid":
                raw_seed = control.get("seed")
                s = int(raw_seed) if raw_seed is not None else grid.config.seed
                if self.is_topology_mode and hasattr(grid, "reset_state"):
                    grid.reset_state(s)
                else:
                    grid.gol_state = initialize_conway(
                        grid.config.height, grid.config.width, s
                    )
                    grid.v, grid.u = initialize_izhikevich(
                        (grid.config.height, grid.config.width), s
                    )
                    grid.previous_spikes[:] = False
                    grid.memory_state[:] = 0.0
                    grid.strategy_map[:] = 1
                    new_edges = create_small_world_graph(
                        self.node_count,
                        k_neighbors=grid.config.small_world_k,
                        rewire_prob=grid.config.small_world_rewire,
                        seed=s,
                    )
                    grid.attach_graph_edges(new_edges)
                self.graph_version += 1
                self._sent_graph_payload_version = -1
                self._sent_coords_version = -1
            elif action == "set_topology":
                self._handle_set_topology(control, grid)

    def _handle_set_topology(self, control: Dict, grid) -> None:
        """Build a new GraphNeuralGrid for the requested selection.

        The newly built grid is queued via ``_pending_grid`` so the
        simulation loop swaps it in atomically *between* steps – this
        avoids mid-step shape mismatches.

        Refusing to swap (no swc_dir, empty selection, ...) is silent on
        purpose; the front-end keeps the previous topology running.
        """
        if self.swc_dir is None:
            return
        from conway_izh.graph_grid import GraphNeuralGrid

        raw_selection = control.get("selection") or control.get("topologies") or []
        if not isinstance(raw_selection, (list, tuple)):
            return
        cleaned = [str(s).strip() for s in raw_selection if str(s).strip()]
        if not cleaned:
            cleaned = [SMALL_WORLD_KEY]

        seed = control.get("seed")
        try:
            seed_val = int(seed) if seed is not None else grid.config.seed
        except (TypeError, ValueError):
            seed_val = grid.config.seed

        small_world_n = int(control.get("small_world_n", 3600))
        small_world_n = max(50, min(small_world_n, 20000))

        try:
            topology = build_unified_topology(
                cleaned,
                swc_dir=self.swc_dir,
                small_world_n=small_world_n,
                small_world_k=int(grid.config.small_world_k),
                small_world_rewire=float(grid.config.small_world_rewire),
                seed=seed_val,
                axis_step=float(control.get("axis_step", 200.0)),
            )
        except Exception:
            # Build failures should not crash the streaming loop; just keep
            # serving the previous grid.
            return

        next_grid = GraphNeuralGrid(grid.config, topology)
        self._pending_grid = next_grid

    def _per_component_metrics(
        self,
        grid,
        gol_flat: np.ndarray,
        spike_flat: np.ndarray,
    ) -> List[Dict]:
        """Compute lightweight per-topology metric snapshots.

        Each entry mirrors the global ``metrics`` block but is restricted to
        a single contiguous index slice declared in
        ``self.topology.components``. The slice is provided by the
        ``UnifiedTopology`` builder which guarantees indices for component
        ``k`` live in ``[start, start + n_nodes)``.

        Why backend-side and not browser-side:
            *   We already own the spike + GoL arrays; slicing them in NumPy
                is O(n) and uses contiguous memory.
            *   Keeping the efficiency formula in one place (``efficiency.py``)
                avoids a JS reimplementation and possible drift between the
                two views.

        The cooperative-strategy term is always 1.0 in graph mode (the
        Game-Theory module is disabled) so we approximate ``strategy_shift``
        from the per-component ``alive_ratio`` delta. This still produces a
        meaningful information/memory contrast between SWC trees and the
        small-world ring without inventing arbitrary state.

        Returns an empty list when topology mode is inactive or no
        component metadata is present.
        """
        if not self.is_topology_mode or self.topology is None:
            return []
        components = list(self.topology.components or [])
        if not components:
            return []

        results: List[Dict] = []
        v_arr = getattr(grid, "v", None)
        mem_arr = getattr(grid, "memory_state", None)
        coords = getattr(self.topology, "coords", None)
        for slot, comp in enumerate(components):
            start = int(comp.get("start", 0))
            n_nodes = int(comp.get("n_nodes", 0))
            if n_nodes <= 0:
                continue
            stop = start + n_nodes
            gol_slice = gol_flat[start:stop]
            spike_slice = spike_flat[start:stop]
            denom = max(1, n_nodes)
            alive_count = int(gol_slice.sum())
            spike_count = int(spike_slice.sum())
            alive_ratio = alive_count / denom
            firing_rate = spike_count / denom
            mean_v = (
                float(np.asarray(v_arr[start:stop]).mean())
                if v_arr is not None and v_arr.size >= stop
                else 0.0
            )
            mean_memory = (
                float(np.asarray(mem_arr[start:stop]).mean())
                if mem_arr is not None and mem_arr.size >= stop
                else 0.0
            )
            cooperative_ratio = 1.0
            prev = self._per_component_prev.get(slot)
            prev_alive_ratio = prev["alive_ratio"] if prev is not None else alive_ratio
            strategy_shift = abs(alive_ratio - prev_alive_ratio)
            self._per_component_prev[slot] = {"alive_ratio": alive_ratio}

            stability = compute_stability_score(alive_ratio, firing_rate)
            information = compute_information_score(
                firing_rate, cooperative_ratio, strategy_shift
            )
            memory = compute_memory_score(mean_memory, strategy_shift)
            cost = compute_cost_score(firing_rate, spike_count, n_nodes)
            efficiency = compute_efficiency_score(stability, information, memory, cost)

            center = None
            if coords is not None and coords.shape[0] >= stop:
                segment = coords[start:stop]
                if segment.size:
                    c = segment.mean(axis=0)
                    center = [float(c[0]), float(c[1]), float(c[2])]

            results.append(
                {
                    "key": str(comp.get("key", f"comp-{slot}")),
                    "label": str(comp.get("label", comp.get("key", f"comp-{slot}"))),
                    "kind": str(comp.get("kind", "swc")),
                    "slot": slot,
                    "start": start,
                    "n_nodes": n_nodes,
                    "n_edges": int(comp.get("n_edges", 0)),
                    "color": list(comp.get("color", [1.0, 1.0, 1.0])),
                    "offset": list(comp.get("offset", [0.0, 0.0, 0.0])),
                    "center": center,
                    "metrics": {
                        "alive_count": alive_count,
                        "alive_ratio": float(alive_ratio),
                        "spike_count": spike_count,
                        "firing_rate": float(firing_rate),
                        "mean_v": mean_v,
                        "mean_memory": mean_memory,
                        "stability_score": float(stability),
                        "information_score": float(information),
                        "memory_score": float(memory),
                        "cost_score": float(cost),
                        "efficiency_score": float(efficiency),
                    },
                }
            )

        # Garbage-collect stale slot history if the selection shrank.
        valid_slots = {entry["slot"] for entry in results}
        for stale_slot in list(self._per_component_prev.keys()):
            if stale_slot not in valid_slots:
                self._per_component_prev.pop(stale_slot, None)

        return results

    def payload(self, grid, metrics: Dict, spikes: np.ndarray) -> Dict:
        step_idx = int(metrics["step"])
        topology_kind = "topology" if self.is_topology_mode else "small_world"
        graph_payload: Dict = {
            "version": self.graph_version,
            "kind": topology_kind,
        }

        # We re-send heavy fields (edges + coords) only when the topology
        # version actually changes, plus every ~300 steps as a heartbeat
        # safety net. This keeps each frame on the wire small and lets the
        # browser dispose() old GL buffers exactly when needed.
        send_edges = (
            self.graph_version != self._sent_graph_payload_version
            or step_idx % 300 == 0
        )
        if send_edges:
            # Prefer the cached ``edges_arr`` (int32 ndarray) when the grid
            # exposes it: ``ndarray.tolist()`` is a single C-level call that
            # produces nested Python ints directly, whereas the legacy
            # ``graph_edges`` property allocates a fresh comprehension on
            # every access AND we then re-comprehended it here. For ~20k
            # edges that double-traversal cost ~30-50 ms per heartbeat.
            edges_arr = getattr(grid, "edges_arr", None)
            if edges_arr is not None and getattr(edges_arr, "size", 0):
                graph_payload["edges"] = edges_arr.tolist()
            else:
                graph_payload["edges"] = [
                    [int(a), int(b)] for a, b in grid.graph_edges
                ]
            self._sent_graph_payload_version = self.graph_version

        gol_flat = np.asarray(grid.gol_state).astype(np.uint8, copy=False).ravel()
        spike_flat = np.asarray(spikes).astype(np.uint8, copy=False).ravel()

        topology_payload: Optional[Dict] = None
        if self.is_topology_mode and self.topology is not None:
            send_coords = (
                self.graph_version != self._sent_coords_version
                or step_idx % 300 == 0
            )
            topology_payload = {
                "active": True,
                "selection": list(self.current_selection),
                "components": list(self.topology.components),
                "n_nodes": int(self.topology.n_nodes),
                "bbox_min": self.topology.bbox_min.tolist(),
                "bbox_max": self.topology.bbox_max.tolist(),
                "version": self.graph_version,
                # Per-topology metric snapshot: empty when only one component
                # is selected so the front-end can collapse the panel cheaply.
                "per_component": self._per_component_metrics(
                    grid, gol_flat, spike_flat
                ),
            }
            if send_coords:
                # Flat Float32-friendly array; the browser will read it
                # straight into a Float32Array (no per-vertex object alloc).
                topology_payload["node_coords"] = (
                    self.topology.coords.astype(np.float32, copy=False)
                    .ravel()
                    .tolist()
                )
                self._sent_coords_version = self.graph_version
        else:
            topology_payload = {"active": False}

        return {
            "ready": True,
            "width": self.width,
            "height": self.height,
            "n_nodes": int(self.node_count),
            "gol_state": gol_flat.tolist(),
            "spikes": spike_flat.tolist(),
            "metrics": {
                "step": int(metrics["step"]),
                "alive_count": int(metrics["alive_count"]),
                "spike_count": int(metrics["spike_count"]),
                "stability_score": float(metrics["stability_score"]),
                "information_score": float(metrics["information_score"]),
                "memory_score": float(metrics["memory_score"]),
                "cost_score": float(metrics["cost_score"]),
                "efficiency_score": float(metrics["efficiency_score"]),
            },
            "rules": {
                "birth": list(self.birth_neighbors),
                "survive": list(self.survive_neighbors),
            },
            "graph": graph_payload,
            "topology": topology_payload,
            "generation_period_ms": self.generation_period_ms,
            "paused": bool(self.paused),
            "coupling": {
                "k_alive": float(grid.config.coupling.k_alive),
                "k_neighbors": float(grid.config.coupling.k_neighbors),
                "bias": float(grid.config.coupling.bias),
                "feedback_enabled": bool(grid.config.coupling.feedback_enabled),
                "k_syn": float(grid.config.coupling.k_syn),
                "spike_trace_decay": float(grid.config.coupling.spike_trace_decay),
                "feedback_graph_neighbors": bool(
                    grid.config.coupling.feedback_graph_neighbors
                ),
            },
        }

