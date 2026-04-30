#!/usr/bin/env python3
"""Run Conway-Izhikevich simulation with live animation."""

import argparse
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from pathlib import Path

# Add src to path for imports
_project_root = Path(__file__).parent.parent
_src_path = _project_root / "src"
if str(_src_path) not in sys.path:
    sys.path.insert(0, str(_src_path))

from conway_izh.grid import NeuralGrid
from conway_izh.config import SimulationConfig, IzhikevichParams, CouplingParams


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run Conway-Izhikevich simulation with live animation"
    )
    
    parser.add_argument("--height", type=int, default=60,
                       help="Grid height (default: 60)")
    parser.add_argument("--width", type=int, default=60,
                       help="Grid width (default: 60)")
    parser.add_argument("--steps", type=int, default=300,
                       help="Number of simulation steps (default: 300)")
    parser.add_argument("--seed", type=int, default=None,
                       help="Random seed")
    parser.add_argument("--feedback", action="store_true",
                       help="Enable neuron->GoL feedback")
    parser.add_argument("--interval", type=int, default=50,
                       help="Animation interval in milliseconds (default: 50)")
    parser.add_argument("--k-alive", type=float, default=2.0,
                       help="Coupling weight for alive cells (default: 2.0)")
    
    return parser.parse_args()


class LiveSimulation:
    """Live animation wrapper for NeuralGrid."""
    
    def __init__(self, grid: NeuralGrid):
        self.grid = grid
        self.step_count = 0
        self.max_steps = grid.config.steps
        
        # Setup figure with subplots: 2x3 grid
        self.fig = plt.figure(figsize=(20, 12))
        self.fig.suptitle('Conway Game of Life + Izhikevich Neuron Simulation (Live)', 
                         fontsize=16, fontweight='bold')
        
        # Top row: Visualizations
        gs = self.fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
        
        # Conway plot
        ax_gol = self.fig.add_subplot(gs[0, 0])
        self.im_gol = ax_gol.imshow(
            self.grid.gol_state, cmap='gray', interpolation='nearest', vmin=0, vmax=1
        )
        ax_gol.set_title('Conway Game of Life', fontsize=12, fontweight='bold')
        ax_gol.axis('off')
        
        # Membrane potential plot
        ax_v = self.fig.add_subplot(gs[0, 1])
        self.im_v = ax_v.imshow(
            self.grid.v, cmap='viridis', interpolation='nearest'
        )
        ax_v.set_title('Membrane Potential (mV)', fontsize=12, fontweight='bold')
        ax_v.axis('off')
        plt.colorbar(self.im_v, ax=ax_v, fraction=0.046)
        
        # Stats text
        ax_stats = self.fig.add_subplot(gs[0, 2])
        ax_stats.axis('off')
        self.stats_text = ax_stats.text(
            0.1, 0.5, '', fontsize=11, verticalalignment='center',
            family='monospace', transform=ax_stats.transAxes
        )
        
        # Bottom row: Time series plots
        # Alive count
        self.ax_alive = self.fig.add_subplot(gs[1, 0])
        self.ax_alive.set_title('Alive Cells Over Time', fontsize=11, fontweight='bold')
        self.ax_alive.set_xlabel('Step')
        self.ax_alive.set_ylabel('Count')
        self.ax_alive.grid(True, alpha=0.3)
        self.line_alive, = self.ax_alive.plot([], [], 'b-', linewidth=2)
        self.alive_history = []
        self.step_history = []
        
        # Spike count
        self.ax_spikes = self.fig.add_subplot(gs[1, 1])
        self.ax_spikes.set_title('Spike Count Over Time', fontsize=11, fontweight='bold')
        self.ax_spikes.set_xlabel('Step')
        self.ax_spikes.set_ylabel('Count')
        self.ax_spikes.grid(True, alpha=0.3)
        self.line_spikes, = self.ax_spikes.plot([], [], 'r-', linewidth=2)
        self.spike_history = []
        
        # Mean V
        self.ax_meanv = self.fig.add_subplot(gs[1, 2])
        self.ax_meanv.set_title('Mean Membrane Potential', fontsize=11, fontweight='bold')
        self.ax_meanv.set_xlabel('Step')
        self.ax_meanv.set_ylabel('V (mV)')
        self.ax_meanv.grid(True, alpha=0.3)
        self.ax_meanv.axhline(y=30, color='orange', linestyle='--', alpha=0.5, label='Threshold')
        self.line_meanv, = self.ax_meanv.plot([], [], 'g-', linewidth=2)
        self.meanv_history = []
        self.ax_meanv.legend()
        
        self.axes = [ax_gol, ax_v, ax_stats, self.ax_alive, self.ax_spikes, self.ax_meanv]
    
    def update(self, frame):
        """Update animation frame."""
        if self.step_count >= self.max_steps:
            return [self.im_gol, self.im_v, self.line_alive, self.line_spikes, self.line_meanv]
        
        # Run one step
        metrics, spikes = self.grid.step()
        self.step_count += 1
        
        # Store history
        self.step_history.append(self.step_count)
        self.alive_history.append(metrics['alive_count'])
        self.spike_history.append(metrics['spike_count'])
        self.meanv_history.append(metrics['mean_v'])
        
        # Update Conway plot
        self.im_gol.set_array(self.grid.gol_state)
        
        # Update membrane potential plot
        self.im_v.set_array(self.grid.v)
        self.im_v.set_clim(vmin=self.grid.v.min(), vmax=self.grid.v.max())
        
        # Update time series plots
        self.line_alive.set_data(self.step_history, self.alive_history)
        self.ax_alive.relim()
        self.ax_alive.autoscale_view()
        
        self.line_spikes.set_data(self.step_history, self.spike_history)
        self.ax_spikes.relim()
        self.ax_spikes.autoscale_view()
        
        self.line_meanv.set_data(self.step_history, self.meanv_history)
        self.ax_meanv.relim()
        self.ax_meanv.autoscale_view()
        
        # Update stats
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
        
        # Print progress every 50 steps
        if self.step_count % 50 == 0:
            print(f"Step {self.step_count}/{self.max_steps}: "
                  f"alive={metrics['alive_count']}, "
                  f"spikes={metrics['spike_count']}")
        
        return [self.im_gol, self.im_v, self.line_alive, self.line_spikes, self.line_meanv]
    
    def run(self, interval=50):
        """Run live animation."""
        print(f"Starting live simulation...")
        print(f"Grid size: {self.grid.config.height}x{self.grid.config.width}")
        print(f"Steps: {self.max_steps}")
        print(f"Close the window or press Ctrl+C to stop")
        print("-" * 50)
        
        # Create animation
        self.anim = FuncAnimation(
            self.fig, self.update, interval=interval, blit=False, repeat=False
        )
        
        # Show plot
        plt.show()


def main():
    """Main entry point."""
    args = parse_args()
    
    # Create configuration
    config = SimulationConfig(
        height=args.height,
        width=args.width,
        steps=args.steps,
        seed=args.seed,
        coupling=CouplingParams(
            feedback_enabled=args.feedback,
            k_alive=args.k_alive
        ),
        output_dir="outputs",  # Not used in live mode
        save_gif=False  # No file output in live mode
    )
    
    # Create grid
    grid = NeuralGrid(config)
    
    # Create and run live simulation
    live_sim = LiveSimulation(grid)
    live_sim.run(interval=args.interval)


if __name__ == "__main__":
    main()

