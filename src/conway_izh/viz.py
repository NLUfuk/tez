"""Visualization and stream publishing for Conway-Izhikevich simulation."""

import json
import threading
from collections import deque
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Callable, Deque, Dict, List, Optional, Sequence, Tuple

import imageio
from conway_izh.conway import initialize_conway
from conway_izh.izhikevich import initialize_izhikevich
import matplotlib.pyplot as plt
import numpy as np


def save_gol_frame(state: np.ndarray, filepath: Path, cmap: str = 'gray'):
    """
    Save Conway Game of Life state as PNG.
    
    Args:
        state: Binary grid (H, W)
        filepath: Output file path
        cmap: Colormap name
    """
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


def build_control_handler(bridge: StreamBridge):
    """Build an HTTP handler bound to a bridge instance."""

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
            if self.path != "/state":
                self._write_json(HTTPStatus.NOT_FOUND, {"error": "Not Found"})
                return
            payload = bridge.get_state()
            if payload is None:
                self._write_json(HTTPStatus.OK, {"ready": False})
                return
            self._write_json(HTTPStatus.OK, payload)

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

    def __init__(self, bridge: StreamBridge, host: str, port: int):
        self._bridge = bridge
        self._server = ThreadingHTTPServer((host, port), build_control_handler(bridge))
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    def start(self):
        self._thread.start()

    def stop(self):
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=1.0)


class StreamPublisher:
    """Publishes simulation state and applies live control updates."""

    def __init__(self, bridge: StreamBridge, grid):
        """Uses ``grid.graph_edges`` as canonical topology (physics + viz)."""
        self.bridge = bridge
        self.grid = grid
        self.width = int(grid.config.width)
        self.height = int(grid.config.height)
        self.node_count = self.width * self.height
        self.generation_period_ms = 50
        self.birth_neighbors: Tuple[int, ...] = tuple(grid.config.birth_neighbors)
        self.survive_neighbors: Tuple[int, ...] = tuple(grid.config.survive_neighbors)
        self.graph_version = 1
        self._sent_graph_payload_version = -1

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

    def payload(self, grid, metrics: Dict, spikes: np.ndarray) -> Dict:
        step_idx = int(metrics["step"])
        graph_payload: Dict = {
            "version": self.graph_version,
            "kind": "small_world",
        }
        send_edges = (
            self.graph_version != self._sent_graph_payload_version
            or step_idx % 300 == 0
        )
        if send_edges:
            graph_payload["edges"] = [
                [int(a), int(b)] for a, b in grid.graph_edges
            ]
            self._sent_graph_payload_version = self.graph_version

        return {
            "ready": True,
            "width": self.width,
            "height": self.height,
            "gol_state": grid.gol_state.astype(np.uint8).ravel().tolist(),
            "spikes": spikes.astype(np.uint8).ravel().tolist(),
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
            "generation_period_ms": self.generation_period_ms,
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

