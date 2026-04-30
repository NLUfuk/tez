#!/usr/bin/env python3
"""Run Conway-Izhikevich simulation with live animation - Game Theory Mode."""

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
        description="Run Conway-Izhikevich simulation with live animation (Game Theory)"
    )
    
    parser.add_argument("--height", type=int, default=60,
                       help="Grid height (default: 60)")
    parser.add_argument("--width", type=int, default=60,
                       help="Grid width (default: 60)")
    parser.add_argument("--steps", type=int, default=300,
                       help="Number of simulation steps (default: 300)")
    parser.add_argument("--seed", type=int, default=None,
                       help="Random seed")
    parser.add_argument("--interval", type=int, default=50,
                       help="Animation interval in milliseconds (default: 50)")
    parser.add_argument("--propagation-strength", type=float, default=0.5,
                       help="Spike propagation strength (0-1, default: 0.5)")
    parser.add_argument("--cooperation-factor", type=float, default=0.3,
                       help="Cooperation factor (default: 0.3)")
    parser.add_argument("--cooperation-strength", type=float, default=0.7,
                       help="Feedback cooperation strength (default: 0.7)")
    
    return parser.parse_args()


class LiveGameTheorySimulation:
    """Live animation wrapper for NeuralGrid with Game Theory."""
    
    def __init__(self, grid: NeuralGrid):
        self.grid = grid
        self.step_count = 0
        self.max_steps = grid.config.steps
        
        # Setup figure with subplots: 2x3 grid
        self.fig = plt.figure(figsize=(20, 12))
        self.fig.suptitle('Game Theory Coupling: Conway + Izhikevich (Live)', 
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
        
        # Spike visualization
        ax_spikes = self.fig.add_subplot(gs[0, 2])
        self.im_spikes = ax_spikes.imshow(
            np.zeros_like(self.grid.gol_state), cmap='hot', interpolation='nearest', vmin=0, vmax=1
        )
        ax_spikes.set_title('Spike Activity', fontsize=12, fontweight='bold')
        ax_spikes.axis('off')
        plt.colorbar(self.im_spikes, ax=ax_spikes, fraction=0.046)
        
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
        self.ax_spikes_ts = self.fig.add_subplot(gs[1, 1])
        self.ax_spikes_ts.set_title('Spike Count Over Time', fontsize=11, fontweight='bold')
        self.ax_spikes_ts.set_xlabel('Step')
        self.ax_spikes_ts.set_ylabel('Count')
        self.ax_spikes_ts.grid(True, alpha=0.3)
        self.line_spikes, = self.ax_spikes_ts.plot([], [], 'r-', linewidth=2)
        self.spike_history = []
        
        # Firing rate
        self.ax_firing = self.fig.add_subplot(gs[1, 2])
        self.ax_firing.set_title('Firing Rate Over Time', fontsize=11, fontweight='bold')
        self.ax_firing.set_xlabel('Step')
        self.ax_firing.set_ylabel('Rate')
        self.ax_firing.grid(True, alpha=0.3)
        self.line_firing, = self.ax_firing.plot([], [], 'm-', linewidth=2)
        self.firing_history = []
    
    def update(self, frame):
        """Update animation frame."""
        if self.step_count >= self.max_steps:
            return [self.im_gol, self.im_v, self.im_spikes, self.line_alive, self.line_spikes, self.line_firing]
        
        # Run one step
        metrics, spikes = self.grid.step()
        self.step_count += 1
        
        # Store history
        self.step_history.append(self.step_count)
        self.alive_history.append(metrics['alive_count'])
        self.spike_history.append(metrics['spike_count'])
        self.firing_history.append(metrics['firing_rate'])
        
        # Update Conway plot
        self.im_gol.set_array(self.grid.gol_state)
        
        # Update membrane potential plot
        self.im_v.set_array(self.grid.v)
        self.im_v.set_clim(vmin=self.grid.v.min(), vmax=self.grid.v.max())
        
        # Update spike visualization (show current spikes)
        spike_vis = spikes.astype(float)
        self.im_spikes.set_array(spike_vis)
        
        # Update time series plots
        self.line_alive.set_data(self.step_history, self.alive_history)
        self.ax_alive.relim()
        self.ax_alive.autoscale_view()
        
        self.line_spikes.set_data(self.step_history, self.spike_history)
        self.ax_spikes_ts.relim()
        self.ax_spikes_ts.autoscale_view()
        
        self.line_firing.set_data(self.step_history, self.firing_history)
        self.ax_firing.relim()
        self.ax_firing.autoscale_view()
        
        # Print progress every 50 steps
        if self.step_count % 50 == 0:
            print(f"Step {self.step_count}/{self.max_steps}: "
                  f"alive={metrics['alive_count']}, "
                  f"spikes={metrics['spike_count']}, "
                  f"firing_rate={metrics['firing_rate']*100:.2f}%")
        
        return [self.im_gol, self.im_v, self.im_spikes, self.line_alive, self.line_spikes, self.line_firing]
    
    def run(self, interval=50):
        """Run live animation."""
        print(f"Starting Game Theory simulation...")
        print(f"Grid size: {self.grid.config.height}x{self.grid.config.width}")
        print(f"Steps: {self.max_steps}")
        print(f"Game Theory: ENABLED")
        print(f"Propagation strength: {self.grid.config.coupling.propagation_strength}")
        print(f"Cooperation factor: {self.grid.config.coupling.cooperation_factor}")
        print(f"Close the window or press Ctrl+C to stop")
        print("-" * 60)
        
        # Create animation
        self.anim = FuncAnimation(
            self.fig, self.update, interval=interval, blit=False, repeat=False
        )
        
        # Show plot
        plt.show()


def main():
    """Main entry point."""
    args = parse_args()
    
    # Create configuration with Game Theory enabled
    config = SimulationConfig(
        height=args.height,
        width=args.width,
        steps=args.steps,
        seed=args.seed,
        coupling=CouplingParams(
            feedback_enabled=True,  # Enable feedback for game theory
            use_game_theory=True,   # Enable game theory mode
            propagation_strength=args.propagation_strength,
            cooperation_factor=args.cooperation_factor,
            cooperation_strength=args.cooperation_strength
        ),
        output_dir="outputs",
        save_gif=False  # No file output in live mode
    )
    
    # Create grid
    grid = NeuralGrid(config)
    
    # Create and run live simulation
    live_sim = LiveGameTheorySimulation(grid)
    live_sim.run(interval=args.interval)


if __name__ == "__main__":
    main()

