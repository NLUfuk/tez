#!/usr/bin/env python3
"""Run Conway-Izhikevich simulation with live animation or JSON streaming."""

import argparse
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

# Add src to path for imports
_project_root = Path(__file__).parent.parent
_src_path = _project_root / "src"
if str(_src_path) not in sys.path:
    sys.path.insert(0, str(_src_path))

from conway_izh.config import CouplingParams, SimulationConfig
from conway_izh.grid import NeuralGrid
from conway_izh.viz import StreamBridge, StreamPublisher, StreamServer


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run Conway-Izhikevich simulation with live animation"
    )

    parser.add_argument("--height", type=int, default=60, help="Grid height (default: 60)")
    parser.add_argument("--width", type=int, default=60, help="Grid width (default: 60)")
    parser.add_argument("--steps", type=int, default=300, help="Number of simulation steps (default: 300)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--feedback", action="store_true", help="Enable neuron->GoL feedback")
    parser.add_argument("--interval", type=int, default=50, help="Animation interval in milliseconds (default: 50)")
    parser.add_argument("--k-alive", type=float, default=4.0, help="Coupling weight for alive cells (default: 4.0)")
    parser.add_argument(
        "--k-syn",
        type=float,
        default=2.8,
        help="Stream: graph spike-trace coupling (0 disables). Default 2.8",
    )
    parser.add_argument(
        "--trace-decay",
        type=float,
        default=0.88,
        dest="trace_decay",
        help="Stream: exponential decay of spike trace on edges [0..1)",
    )
    parser.add_argument(
        "--no-feedback",
        action="store_true",
        help="Stream: disable spike→Conway feedback (matplotlib uses --feedback)",
    )
    parser.add_argument(
        "--no-graph-feedback-neighbors",
        action="store_true",
        help="Stream: only spike cell revives GoL, not graph neighbors",
    )
    parser.add_argument("--mode", choices=["matplotlib", "stream"], default="matplotlib",
                        help="Live mode: matplotlib UI or JSON stream server")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host for stream mode")
    parser.add_argument("--port", type=int, default=8765, help="Port for stream mode")

    return parser.parse_args()


class LiveSimulation:
    """Live animation wrapper for NeuralGrid."""

    def __init__(self, grid: NeuralGrid):
        self.grid = grid
        self.step_count = 0
        self.max_steps = grid.config.steps

        self.fig = plt.figure(figsize=(20, 12))
        self.fig.suptitle(
            "Conway Game of Life + Izhikevich Neuron Simulation (Live)",
            fontsize=16,
            fontweight="bold",
        )

        gs = self.fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

        ax_gol = self.fig.add_subplot(gs[0, 0])
        self.im_gol = ax_gol.imshow(
            self.grid.gol_state, cmap="gray", interpolation="nearest", vmin=0, vmax=1
        )
        ax_gol.set_title("Conway Game of Life", fontsize=12, fontweight="bold")
        ax_gol.axis("off")

        ax_v = self.fig.add_subplot(gs[0, 1])
        self.im_v = ax_v.imshow(self.grid.v, cmap="viridis", interpolation="nearest")
        ax_v.set_title("Membrane Potential (mV)", fontsize=12, fontweight="bold")
        ax_v.axis("off")
        plt.colorbar(self.im_v, ax=ax_v, fraction=0.046)

        ax_stats = self.fig.add_subplot(gs[0, 2])
        ax_stats.axis("off")
        self.stats_text = ax_stats.text(
            0.1, 0.5, "", fontsize=11, verticalalignment="center", family="monospace", transform=ax_stats.transAxes
        )

        self.ax_alive = self.fig.add_subplot(gs[1, 0])
        self.ax_alive.set_title("Alive Cells Over Time", fontsize=11, fontweight="bold")
        self.ax_alive.set_xlabel("Step")
        self.ax_alive.set_ylabel("Count")
        self.ax_alive.grid(True, alpha=0.3)
        self.line_alive, = self.ax_alive.plot([], [], "b-", linewidth=2)
        self.alive_history = []
        self.step_history = []

        self.ax_spikes = self.fig.add_subplot(gs[1, 1])
        self.ax_spikes.set_title("Spike Count Over Time", fontsize=11, fontweight="bold")
        self.ax_spikes.set_xlabel("Step")
        self.ax_spikes.set_ylabel("Count")
        self.ax_spikes.grid(True, alpha=0.3)
        self.line_spikes, = self.ax_spikes.plot([], [], "r-", linewidth=2)
        self.spike_history = []

        self.ax_meanv = self.fig.add_subplot(gs[1, 2])
        self.ax_meanv.set_title("Mean Membrane Potential", fontsize=11, fontweight="bold")
        self.ax_meanv.set_xlabel("Step")
        self.ax_meanv.set_ylabel("V (mV)")
        self.ax_meanv.grid(True, alpha=0.3)
        self.ax_meanv.axhline(y=30, color="orange", linestyle="--", alpha=0.5, label="Threshold")
        self.line_meanv, = self.ax_meanv.plot([], [], "g-", linewidth=2)
        self.meanv_history = []
        self.ax_meanv.legend()

    def update(self, _frame):
        """Update animation frame."""
        if self.step_count >= self.max_steps:
            return [self.im_gol, self.im_v, self.line_alive, self.line_spikes, self.line_meanv]

        metrics, spikes = self.grid.step()
        self.step_count += 1

        self.step_history.append(self.step_count)
        self.alive_history.append(metrics["alive_count"])
        self.spike_history.append(metrics["spike_count"])
        self.meanv_history.append(metrics["mean_v"])

        self.im_gol.set_array(self.grid.gol_state)
        self.im_v.set_array(self.grid.v)
        self.im_v.set_clim(vmin=self.grid.v.min(), vmax=self.grid.v.max())

        self.line_alive.set_data(self.step_history, self.alive_history)
        self.ax_alive.relim()
        self.ax_alive.autoscale_view()

        self.line_spikes.set_data(self.step_history, self.spike_history)
        self.ax_spikes.relim()
        self.ax_spikes.autoscale_view()

        self.line_meanv.set_data(self.step_history, self.meanv_history)
        self.ax_meanv.relim()
        self.ax_meanv.autoscale_view()

        stats_str = f"""Simulation Stats:

Step: {self.step_count}/{self.max_steps}

Conway:
  Alive cells: {metrics['alive_count']}
  Density: {metrics['alive_count']/(self.grid.config.height*self.grid.config.width)*100:.2f}%

Neurons:
  Spike count: {metrics['spike_count']}
  Mean V: {metrics['mean_v']:.2f} mV
  Firing rate: {metrics['firing_rate']*100:.4f}%

Coupling:
  Feedback: {'ON' if self.grid.config.coupling.feedback_enabled else 'OFF'}
"""
        self.stats_text.set_text(stats_str)

        if self.step_count % 50 == 0:
            print(
                f"Step {self.step_count}/{self.max_steps}: "
                f"alive={metrics['alive_count']}, spikes={metrics['spike_count']}"
            )

        return [self.im_gol, self.im_v, self.line_alive, self.line_spikes, self.line_meanv]

    def run(self, interval=50):
        """Run live animation."""
        print("Starting live simulation...")
        print(f"Grid size: {self.grid.config.height}x{self.grid.config.width}")
        print(f"Steps: {self.max_steps}")
        print("Close the window or press Ctrl+C to stop")
        print("-" * 50)

        self.anim = FuncAnimation(self.fig, self.update, interval=interval, blit=False, repeat=False)
        plt.show()


class StreamSimulation:
    """Headless loop that publishes per-step JSON frames for Three.js."""

    def __init__(self, grid: NeuralGrid, publisher: StreamPublisher):
        self.grid = grid
        self.publisher = publisher
        self.step_count = 0
        self.max_steps = grid.config.steps
        self.cell_count = grid.config.height * grid.config.width

    def run(self, interval_ms: int):
        print("Starting stream simulation mode...")
        print(f"Grid size: {self.grid.config.height}x{self.grid.config.width}")
        print(f"Steps: {self.max_steps}")
        print("HTTP stream endpoint: http://127.0.0.1:8765/state")
        print("-" * 50)
        while self.step_count < self.max_steps:
            def manual_spike(index: int):
                row = index // self.grid.config.width
                col = index % self.grid.config.width
                # Tıklama ile her hücrede zorlayıcı ateşleme (görsel + Izh tepkisi)
                self.grid.v[row, col] = 35.0

            def toggle_conway(index: int):
                row = index // self.grid.config.width
                col = index % self.grid.config.width
                self.grid.gol_state[row, col] = 0 if self.grid.gol_state[row, col] else 1

            self.publisher.apply_controls(self.grid, manual_spike, toggle_conway)
            metrics, spikes = self.grid.step()
            self.step_count += 1
            self.publisher.bridge.set_state(self.publisher.payload(self.grid, metrics, spikes))
            if self.step_count % 50 == 0:
                print(
                    f"Step {self.step_count}/{self.max_steps}: "
                    f"alive={metrics['alive_count']}, spikes={metrics['spike_count']}, "
                    f"E={metrics['efficiency_score']:.3f}"
                )
            time.sleep(max(0.001, self.publisher.generation_period_ms / 1000.0))


def main():
    """Main entry point."""
    args = parse_args()

    if args.mode == "stream":
        coupling = CouplingParams(
            k_alive=args.k_alive,
            feedback_enabled=(not args.no_feedback),
            k_syn=float(args.k_syn),
            spike_trace_decay=float(np.clip(args.trace_decay, 0.0, 0.9999)),
            feedback_graph_neighbors=(not args.no_graph_feedback_neighbors),
        )
    else:
        coupling = CouplingParams(feedback_enabled=args.feedback, k_alive=args.k_alive)

    config_seed = args.seed
    if args.mode == "stream" and config_seed is None:
        config_seed = 42

    config = SimulationConfig(
        height=args.height,
        width=args.width,
        steps=args.steps,
        seed=config_seed,
        coupling=coupling,
        output_dir="outputs",
        save_gif=False,
    )

    grid = NeuralGrid(config)

    if args.mode == "matplotlib":
        live_sim = LiveSimulation(grid)
        live_sim.run(interval=args.interval)
        return

    bridge = StreamBridge()
    publisher = StreamPublisher(bridge, grid)
    publisher.generation_period_ms = max(1, args.interval)
    server = StreamServer(bridge, args.host, args.port)
    server.start()
    print(f"Control server running at http://{args.host}:{args.port}")

    try:
        stream = StreamSimulation(grid, publisher)
        stream.run(interval_ms=args.interval)
    finally:
        server.stop()


if __name__ == "__main__":
    main()

