#!/usr/bin/env python3
"""Compare standalone Izhikevich neurons vs Conway+Izhikevich integrated model."""

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
from conway_izh.izhikevich import initialize_izhikevich, step_izhikevich


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Compare standalone Izhikevich vs Conway+Izhikevich"
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
    parser.add_argument("--k-alive", type=float, default=2.0,
                       help="Coupling weight for alive cells (default: 2.0)")
    parser.add_argument("--bias", type=float, default=0.0,
                       help="Constant bias current (default: 0.0)")
    parser.add_argument("--save", action="store_true",
                       help="Save plots and analysis (default: False)")
    parser.add_argument("--show", action="store_true",
                       help="Show interactive plot (default: False)")
    parser.add_argument("--live", action="store_true",
                       help="Run with live animation (default: False)")
    
    return parser.parse_args()


class ComparisonSimulation:
    """Compare standalone Izhikevich vs Conway+Izhikevich."""
    
    def __init__(self, config: SimulationConfig):
        self.config = config
        self.step_count = 0
        
        # Initialize standalone Izhikevich (constant input)
        self.v_standalone, self.u_standalone = initialize_izhikevich(
            (config.height, config.width), config.seed
        )
        # Constant input for standalone
        self.I_standalone = np.full((config.height, config.width), config.coupling.bias, dtype=np.float64)
        
        # Initialize integrated Conway+Izhikevich
        self.grid = NeuralGrid(config)
        
        # History for plotting
        self.step_history = []
        self.standalone_spikes = []
        self.integrated_spikes = []
        self.standalone_meanv = []
        self.integrated_meanv = []
        self.alive_count = []
        
        # Setup figure
        self.fig = plt.figure(figsize=(20, 14))
        self.fig.suptitle('Comparison: Standalone Izhikevich vs Conway+Izhikevich Integrated', 
                         fontsize=16, fontweight='bold')
        
        gs = self.fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)
        
        # Top row: Visualizations
        # Standalone membrane potential
        ax_v_standalone = self.fig.add_subplot(gs[0, 0])
        self.im_v_standalone = ax_v_standalone.imshow(
            self.v_standalone, cmap='viridis', interpolation='nearest'
        )
        ax_v_standalone.set_title('Standalone Izhikevich\nMembrane Potential', fontsize=11, fontweight='bold')
        ax_v_standalone.axis('off')
        plt.colorbar(self.im_v_standalone, ax=ax_v_standalone, fraction=0.046)
        
        # Integrated Conway state
        ax_gol = self.fig.add_subplot(gs[0, 1])
        self.im_gol = ax_gol.imshow(
            self.grid.gol_state, cmap='gray', interpolation='nearest', vmin=0, vmax=1
        )
        ax_gol.set_title('Conway Game of Life\n(Integrated Model)', fontsize=11, fontweight='bold')
        ax_gol.axis('off')
        
        # Integrated membrane potential
        ax_v_integrated = self.fig.add_subplot(gs[0, 2])
        self.im_v_integrated = ax_v_integrated.imshow(
            self.grid.v, cmap='viridis', interpolation='nearest'
        )
        ax_v_integrated.set_title('Integrated Model\nMembrane Potential', fontsize=11, fontweight='bold')
        ax_v_integrated.axis('off')
        plt.colorbar(self.im_v_integrated, ax=ax_v_integrated, fraction=0.046)
        
        # Middle row: Time series comparisons
        # Spike count comparison
        ax_spikes = self.fig.add_subplot(gs[1, 0])
        ax_spikes.set_title('Spike Count Comparison', fontsize=12, fontweight='bold')
        ax_spikes.set_xlabel('Step')
        ax_spikes.set_ylabel('Spike Count')
        ax_spikes.grid(True, alpha=0.3)
        self.line_spikes_standalone, = ax_spikes.plot([], [], 'b-', linewidth=2, label='Standalone')
        self.line_spikes_integrated, = ax_spikes.plot([], [], 'r-', linewidth=2, label='Integrated')
        ax_spikes.legend()
        
        # Mean V comparison
        ax_meanv = self.fig.add_subplot(gs[1, 1])
        ax_meanv.set_title('Mean Membrane Potential Comparison', fontsize=12, fontweight='bold')
        ax_meanv.set_xlabel('Step')
        ax_meanv.set_ylabel('Mean V (mV)')
        ax_meanv.grid(True, alpha=0.3)
        ax_meanv.axhline(y=30, color='orange', linestyle='--', alpha=0.5, label='Threshold')
        self.line_meanv_standalone, = ax_meanv.plot([], [], 'b-', linewidth=2, label='Standalone')
        self.line_meanv_integrated, = ax_meanv.plot([], [], 'r-', linewidth=2, label='Integrated')
        ax_meanv.legend()
        
        # Alive count (only for integrated)
        ax_alive = self.fig.add_subplot(gs[1, 2])
        ax_alive.set_title('Conway: Alive Cells', fontsize=12, fontweight='bold')
        ax_alive.set_xlabel('Step')
        ax_alive.set_ylabel('Count')
        ax_alive.grid(True, alpha=0.3)
        self.line_alive, = ax_alive.plot([], [], 'g-', linewidth=2)
        
        # Bottom row: Statistics and difference
        # Stats text
        ax_stats = self.fig.add_subplot(gs[2, 0])
        ax_stats.axis('off')
        self.stats_text = ax_stats.text(
            0.1, 0.5, '', fontsize=10, verticalalignment='center',
            family='monospace', transform=ax_stats.transAxes
        )
        
        # Difference in spike count
        ax_diff_spikes = self.fig.add_subplot(gs[2, 1])
        ax_diff_spikes.set_title('Spike Count Difference\n(Integrated - Standalone)', fontsize=11, fontweight='bold')
        ax_diff_spikes.set_xlabel('Step')
        ax_diff_spikes.set_ylabel('Difference')
        ax_diff_spikes.grid(True, alpha=0.3)
        ax_diff_spikes.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        self.line_diff_spikes, = ax_diff_spikes.plot([], [], 'purple', linewidth=2)
        
        # Difference in mean V
        ax_diff_v = self.fig.add_subplot(gs[2, 2])
        ax_diff_v.set_title('Mean V Difference\n(Integrated - Standalone)', fontsize=11, fontweight='bold')
        ax_diff_v.set_xlabel('Step')
        ax_diff_v.set_ylabel('Difference (mV)')
        ax_diff_v.grid(True, alpha=0.3)
        ax_diff_v.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        self.line_diff_v, = ax_diff_v.plot([], [], 'purple', linewidth=2)
    
    def step(self):
        """Perform one simulation step for both models."""
        # Standalone Izhikevich step (constant input)
        self.v_standalone, self.u_standalone, spikes_standalone = step_izhikevich(
            self.v_standalone, self.u_standalone, self.I_standalone,
            self.config.izh_params, self.config.dt
        )
        
        # Integrated model step
        metrics, spikes_integrated = self.grid.step()
        
        # Store metrics
        self.step_count += 1
        self.step_history.append(self.step_count)
        
        spike_count_standalone = int(np.sum(spikes_standalone))
        spike_count_integrated = metrics['spike_count']
        
        self.standalone_spikes.append(spike_count_standalone)
        self.integrated_spikes.append(spike_count_integrated)
        self.standalone_meanv.append(float(np.mean(self.v_standalone)))
        self.integrated_meanv.append(metrics['mean_v'])
        self.alive_count.append(metrics['alive_count'])
        
        return metrics, spikes_standalone, spikes_integrated
    
    def update(self, frame):
        """Update animation frame."""
        if self.step_count >= self.config.steps:
            return []
        
        # Run one step
        metrics, spikes_standalone, spikes_integrated = self.step()
        
        # Update visualizations
        self.im_v_standalone.set_array(self.v_standalone)
        self.im_v_standalone.set_clim(vmin=self.v_standalone.min(), vmax=self.v_standalone.max())
        
        self.im_gol.set_array(self.grid.gol_state)
        
        self.im_v_integrated.set_array(self.grid.v)
        self.im_v_integrated.set_clim(vmin=self.grid.v.min(), vmax=self.grid.v.max())
        
        # Update time series plots
        self.line_spikes_standalone.set_data(self.step_history, self.standalone_spikes)
        self.line_spikes_integrated.set_data(self.step_history, self.integrated_spikes)
        
        self.line_meanv_standalone.set_data(self.step_history, self.standalone_meanv)
        self.line_meanv_integrated.set_data(self.step_history, self.integrated_meanv)
        
        self.line_alive.set_data(self.step_history, self.alive_count)
        
        # Update difference plots
        diff_spikes = np.array(self.integrated_spikes) - np.array(self.standalone_spikes)
        diff_v = np.array(self.integrated_meanv) - np.array(self.standalone_meanv)
        
        self.line_diff_spikes.set_data(self.step_history, diff_spikes)
        self.line_diff_v.set_data(self.step_history, diff_v)
        
        # Auto-scale all plots
        for ax in self.fig.axes:
            if hasattr(ax, 'relim'):
                ax.relim()
                ax.autoscale_view()
        
        # Update stats
        total_spikes_standalone = sum(self.standalone_spikes)
        total_spikes_integrated = sum(self.integrated_spikes)
        avg_v_standalone = np.mean(self.standalone_meanv) if self.standalone_meanv else 0
        avg_v_integrated = np.mean(self.integrated_meanv) if self.integrated_meanv else 0
        
        stats_str = f"""Step: {self.step_count}/{self.config.steps}

STANDALONE IZHIKEVICH:
  Total spikes: {total_spikes_standalone}
  Avg V: {avg_v_standalone:.2f} mV
  Current step spikes: {self.standalone_spikes[-1] if self.standalone_spikes else 0}

INTEGRATED MODEL:
  Total spikes: {total_spikes_integrated}
  Avg V: {avg_v_integrated:.2f} mV
  Current step spikes: {self.integrated_spikes[-1] if self.integrated_spikes else 0}
  Alive cells: {self.alive_count[-1] if self.alive_count else 0}

DIFFERENCE:
  Spike diff: {total_spikes_integrated - total_spikes_standalone:+d}
  V diff: {avg_v_integrated - avg_v_standalone:+.2f} mV
"""
        self.stats_text.set_text(stats_str)
        
        # Print progress
        if self.step_count % 50 == 0:
            print(f"Step {self.step_count}/{self.config.steps}: "
                  f"Standalone spikes={self.standalone_spikes[-1]}, "
                  f"Integrated spikes={self.integrated_spikes[-1]}, "
                  f"Alive={self.alive_count[-1]}")
        
        return []
    
    def run(self, interval=50, save_plots=True, show_plot=False, live_animation=False):
        """Run comparison simulation."""
        print(f"Starting comparison simulation...")
        print(f"Grid size: {self.config.height}x{self.config.width}")
        print(f"Steps: {self.config.steps}")
        print(f"Standalone: Constant input I={self.config.coupling.bias}")
        print(f"Integrated: Conway coupling (k_alive={self.config.coupling.k_alive})")
        
        if live_animation:
            print(f"Live animation mode: ON (interval={interval}ms)")
            print(f"Close the window or press Ctrl+C to stop")
        else:
            print("Batch mode: Running all steps...")
        print("-" * 60)
        
        if live_animation:
            # Live animation mode with FuncAnimation
            self.anim = FuncAnimation(
                self.fig, self.update, interval=interval, blit=False, repeat=False,
                cache_frame_data=False
            )
            plt.show()
            
            # After animation completes, save if requested
            if save_plots:
                output_path = Path(self.config.output_dir) / "comparison_analysis"
                output_path.mkdir(parents=True, exist_ok=True)
                
                # Save the comparison figure
                plot_path = output_path / "comparison_plot.png"
                self.fig.savefig(plot_path, dpi=150, bbox_inches='tight')
                print(f"\nPlot saved to: {plot_path}")
                
                # Save detailed analysis
                self._save_analysis(output_path)
        else:
            # Batch mode: Run all steps
            for step in range(self.config.steps):
                self.update(step)
            
            print(f"\nSimulation complete!")
            print(f"Total standalone spikes: {sum(self.standalone_spikes)}")
            print(f"Total integrated spikes: {sum(self.integrated_spikes)}")
            print(f"Average standalone V: {np.mean(self.standalone_meanv):.2f} mV")
            print(f"Average integrated V: {np.mean(self.integrated_meanv):.2f} mV")
            
            # Save final plot
            if save_plots:
                output_path = Path(self.config.output_dir) / "comparison_analysis"
                output_path.mkdir(parents=True, exist_ok=True)
                
                # Save the comparison figure
                plot_path = output_path / "comparison_plot.png"
                self.fig.savefig(plot_path, dpi=150, bbox_inches='tight')
                print(f"Plot saved to: {plot_path}")
                
                # Save detailed analysis
                self._save_analysis(output_path)
            
            # Show plot if requested
            if show_plot:
                plt.show()
            else:
                plt.close(self.fig)
    
    def _save_analysis(self, output_dir: Path):
        """Save detailed analysis to files."""
        import json
        import csv
        
        # Save metrics as CSV
        csv_path = output_dir / "comparison_metrics.csv"
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['step', 'standalone_spikes', 'integrated_spikes', 
                           'standalone_meanv', 'integrated_meanv', 'alive_count',
                           'spike_diff', 'v_diff'])
            for i in range(len(self.step_history)):
                writer.writerow([
                    self.step_history[i],
                    self.standalone_spikes[i],
                    self.integrated_spikes[i],
                    self.standalone_meanv[i],
                    self.integrated_meanv[i],
                    self.alive_count[i],
                    self.integrated_spikes[i] - self.standalone_spikes[i],
                    self.integrated_meanv[i] - self.standalone_meanv[i]
                ])
        print(f"Metrics CSV saved to: {csv_path}")
        
        # Save summary statistics
        summary = {
            'config': {
                'height': self.config.height,
                'width': self.config.width,
                'steps': self.config.steps,
                'k_alive': self.config.coupling.k_alive,
                'bias': self.config.coupling.bias
            },
            'standalone': {
                'total_spikes': int(sum(self.standalone_spikes)),
                'avg_spikes_per_step': float(np.mean(self.standalone_spikes)),
                'std_spikes_per_step': float(np.std(self.standalone_spikes)),
                'avg_v': float(np.mean(self.standalone_meanv)),
                'std_v': float(np.std(self.standalone_meanv)),
                'min_v': float(np.min(self.standalone_meanv)),
                'max_v': float(np.max(self.standalone_meanv))
            },
            'integrated': {
                'total_spikes': int(sum(self.integrated_spikes)),
                'avg_spikes_per_step': float(np.mean(self.integrated_spikes)),
                'std_spikes_per_step': float(np.std(self.integrated_spikes)),
                'avg_v': float(np.mean(self.integrated_meanv)),
                'std_v': float(np.std(self.integrated_meanv)),
                'min_v': float(np.min(self.integrated_meanv)),
                'max_v': float(np.max(self.integrated_meanv)),
                'avg_alive_cells': float(np.mean(self.alive_count)),
                'std_alive_cells': float(np.std(self.alive_count))
            },
            'difference': {
                'total_spike_diff': int(sum(self.integrated_spikes) - sum(self.standalone_spikes)),
                'avg_spike_diff': float(np.mean(np.array(self.integrated_spikes) - np.array(self.standalone_spikes))),
                'avg_v_diff': float(np.mean(np.array(self.integrated_meanv) - np.array(self.standalone_meanv))),
                'spike_ratio': float(sum(self.integrated_spikes) / sum(self.standalone_spikes)) if sum(self.standalone_spikes) > 0 else float('inf')
            }
        }
        
        json_path = output_dir / "comparison_summary.json"
        with open(json_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"Summary JSON saved to: {json_path}")
        
        # Print summary
        print("\n" + "="*60)
        print("ANALYSIS SUMMARY")
        print("="*60)
        print(f"Standalone Model:")
        print(f"  Total spikes: {summary['standalone']['total_spikes']}")
        print(f"  Avg spikes/step: {summary['standalone']['avg_spikes_per_step']:.2f} ± {summary['standalone']['std_spikes_per_step']:.2f}")
        print(f"  Avg V: {summary['standalone']['avg_v']:.2f} ± {summary['standalone']['std_v']:.2f} mV")
        print(f"\nIntegrated Model:")
        print(f"  Total spikes: {summary['integrated']['total_spikes']}")
        print(f"  Avg spikes/step: {summary['integrated']['avg_spikes_per_step']:.2f} ± {summary['integrated']['std_spikes_per_step']:.2f}")
        print(f"  Avg V: {summary['integrated']['avg_v']:.2f} ± {summary['integrated']['std_v']:.2f} mV")
        print(f"  Avg alive cells: {summary['integrated']['avg_alive_cells']:.1f} ± {summary['integrated']['std_alive_cells']:.1f}")
        print(f"\nDifference (Integrated - Standalone):")
        print(f"  Total spike difference: {summary['difference']['total_spike_diff']:+d}")
        print(f"  Spike ratio: {summary['difference']['spike_ratio']:.3f}x")
        print(f"  Avg V difference: {summary['difference']['avg_v_diff']:+.2f} mV")
        print("="*60)


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
            k_alive=args.k_alive,
            bias=args.bias,
            feedback_enabled=False  # No feedback for fair comparison
        ),
        output_dir="outputs",
        save_gif=False
    )
    
    # Create and run comparison
    comparison = ComparisonSimulation(config)
    comparison.run(interval=args.interval, save_plots=args.save, show_plot=args.show, live_animation=args.live)


if __name__ == "__main__":
    main()

