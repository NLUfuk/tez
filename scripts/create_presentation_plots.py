#!/usr/bin/env python3
"""Create presentation-quality plots for Conway-Izhikevich simulation."""

import argparse
import sys
import numpy as np
import matplotlib.pyplot as plt
import csv
from pathlib import Path

# Add src to path for imports
_project_root = Path(__file__).parent.parent
_src_path = _project_root / "src"
if str(_src_path) not in sys.path:
    sys.path.insert(0, str(_src_path))


def load_metrics(csv_path: Path) -> dict:
    """Load metrics CSV file and return as dictionary of arrays."""
    data = {'step': [], 'alive_count': [], 'spike_count': [], 'mean_v': [], 'firing_rate': []}
    
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data['step'].append(int(row['step']))
            data['alive_count'].append(int(row['alive_count']))
            data['spike_count'].append(int(row['spike_count']))
            data['mean_v'].append(float(row['mean_v']))
            data['firing_rate'].append(float(row['firing_rate']))
    
    # Convert to numpy arrays
    for key in data:
        data[key] = np.array(data[key])
    
    return data


def create_time_series_plot(metrics: dict, output_path: Path, title: str = ""):
    """Create time series plot for metrics."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(title if title else 'Simulation Metrics Over Time', fontsize=16, fontweight='bold')
    
    steps = metrics['step']
    
    # Alive count
    axes[0, 0].plot(steps, metrics['alive_count'], 'b-', linewidth=2, alpha=0.8)
    axes[0, 0].set_xlabel('Time Step', fontsize=11)
    axes[0, 0].set_ylabel('Alive Cells', fontsize=11)
    axes[0, 0].set_title('Conway Game of Life: Alive Cells', fontsize=12, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].fill_between(steps, metrics['alive_count'], alpha=0.3)
    
    # Spike count
    axes[0, 1].plot(steps, metrics['spike_count'], 'r-', linewidth=2, alpha=0.8)
    axes[0, 1].set_xlabel('Time Step', fontsize=11)
    axes[0, 1].set_ylabel('Spike Count', fontsize=11)
    axes[0, 1].set_title('Izhikevich Neurons: Spike Count', fontsize=12, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].fill_between(steps, metrics['spike_count'], alpha=0.3, color='red')
    
    # Mean membrane potential
    axes[1, 0].plot(steps, metrics['mean_v'], 'g-', linewidth=2, alpha=0.8)
    axes[1, 0].set_xlabel('Time Step', fontsize=11)
    axes[1, 0].set_ylabel('Mean V (mV)', fontsize=11)
    axes[1, 0].set_title('Mean Membrane Potential', fontsize=12, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].axhline(y=30, color='orange', linestyle='--', alpha=0.5, label='Spike threshold')
    axes[1, 0].legend()
    
    # Firing rate
    axes[1, 1].plot(steps, metrics['firing_rate'], 'm-', linewidth=2, alpha=0.8)
    axes[1, 1].set_xlabel('Time Step', fontsize=11)
    axes[1, 1].set_ylabel('Firing Rate', fontsize=11)
    axes[1, 1].set_title('Neuron Firing Rate', fontsize=12, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].fill_between(steps, metrics['firing_rate'], alpha=0.3, color='magenta')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Time series plot saved: {output_path}")


def create_comparison_plot(metrics1: dict, metrics2: dict,
                          label1: str, label2: str, output_path: Path):
    """Create comparison plot between two simulations."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Feedback Effect Comparison', fontsize=16, fontweight='bold')
    
    steps1 = metrics1['step']
    steps2 = metrics2['step']
    
    # Alive count comparison
    axes[0, 0].plot(steps1, metrics1['alive_count'], 'b-', linewidth=2, alpha=0.7, label=label1)
    axes[0, 0].plot(steps2, metrics2['alive_count'], 'r--', linewidth=2, alpha=0.7, label=label2)
    axes[0, 0].set_xlabel('Time Step', fontsize=11)
    axes[0, 0].set_ylabel('Alive Cells', fontsize=11)
    axes[0, 0].set_title('Conway: Alive Cells Comparison', fontsize=12, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Spike count comparison
    axes[0, 1].plot(steps1, metrics1['spike_count'], 'b-', linewidth=2, alpha=0.7, label=label1)
    axes[0, 1].plot(steps2, metrics2['spike_count'], 'r--', linewidth=2, alpha=0.7, label=label2)
    axes[0, 1].set_xlabel('Time Step', fontsize=11)
    axes[0, 1].set_ylabel('Spike Count', fontsize=11)
    axes[0, 1].set_title('Neurons: Spike Count Comparison', fontsize=12, fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Mean V comparison
    axes[1, 0].plot(steps1, metrics1['mean_v'], 'b-', linewidth=2, alpha=0.7, label=label1)
    axes[1, 0].plot(steps2, metrics2['mean_v'], 'r--', linewidth=2, alpha=0.7, label=label2)
    axes[1, 0].set_xlabel('Time Step', fontsize=11)
    axes[1, 0].set_ylabel('Mean V (mV)', fontsize=11)
    axes[1, 0].set_title('Mean Membrane Potential Comparison', fontsize=12, fontweight='bold')
    axes[1, 0].axhline(y=30, color='orange', linestyle=':', alpha=0.5)
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Firing rate comparison
    axes[1, 1].plot(steps1, metrics1['firing_rate'], 'b-', linewidth=2, alpha=0.7, label=label1)
    axes[1, 1].plot(steps2, metrics2['firing_rate'], 'r--', linewidth=2, alpha=0.7, label=label2)
    axes[1, 1].set_xlabel('Time Step', fontsize=11)
    axes[1, 1].set_ylabel('Firing Rate', fontsize=11)
    axes[1, 1].set_title('Firing Rate Comparison', fontsize=12, fontweight='bold')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Comparison plot saved: {output_path}")


def create_coupling_visualization(output_dir: Path):
    """Create visualization showing coupling mechanism."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Coupling Mechanism: GoL ↔ Izhikevich Neurons', fontsize=16, fontweight='bold')
    
    # Create example patterns
    gol_pattern = np.array([
        [0, 0, 1, 1, 0, 0],
        [0, 1, 1, 1, 1, 0],
        [1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1],
        [0, 1, 1, 1, 1, 0],
        [0, 0, 1, 1, 0, 0]
    ])
    
    # Neighbor count
    from conway_izh.conway import count_neighbors
    neighbors = count_neighbors(gol_pattern, wrap_around=False)
    
    # Simulated current (simplified)
    I = 0.5 * neighbors + 2.0 * gol_pattern
    
    # Plot 1: GoL state
    im1 = axes[0].imshow(gol_pattern, cmap='gray', interpolation='nearest')
    axes[0].set_title('Conway Game of Life\n(Alive = 1, Dead = 0)', fontsize=12, fontweight='bold')
    axes[0].axis('off')
    plt.colorbar(im1, ax=axes[0], fraction=0.046)
    
    # Plot 2: Neighbor count
    im2 = axes[1].imshow(neighbors, cmap='viridis', interpolation='nearest')
    axes[1].set_title('Neighbor Count\n(8-neighborhood)', fontsize=12, fontweight='bold')
    axes[1].axis('off')
    plt.colorbar(im2, ax=axes[1], fraction=0.046)
    
    # Plot 3: Input current
    im3 = axes[2].imshow(I, cmap='hot', interpolation='nearest')
    axes[2].set_title('Input Current to Neurons\nI = k_neighbors × neighbors + k_alive × alive', 
                     fontsize=12, fontweight='bold')
    axes[2].axis('off')
    plt.colorbar(im3, ax=axes[2], fraction=0.046)
    
    plt.tight_layout()
    output_path = output_dir / "coupling_mechanism.png"
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Coupling visualization saved: {output_path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Create presentation plots")
    parser.add_argument("--demo-dir", type=str, default="outputs/presentation_demo",
                       help="Demo output directory")
    parser.add_argument("--feedback-dir", type=str, default="outputs/presentation_feedback",
                       help="Feedback output directory")
    parser.add_argument("--out", type=str, default="outputs/presentation_plots",
                       help="Output directory for plots")
    
    args = parser.parse_args()
    
    demo_dir = Path(args.demo_dir)
    feedback_dir = Path(args.feedback_dir)
    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load metrics
    print("Loading metrics...")
    metrics_demo = load_metrics(demo_dir / "metrics.csv")
    metrics_feedback = load_metrics(feedback_dir / "metrics.csv")
    
    # Create time series plots
    print("Creating time series plots...")
    create_time_series_plot(metrics_demo, output_dir / "time_series_demo.png", 
                            "Simulation Metrics (No Feedback)")
    create_time_series_plot(metrics_feedback, output_dir / "time_series_feedback.png",
                            "Simulation Metrics (With Feedback)")
    
    # Create comparison plot
    print("Creating comparison plot...")
    create_comparison_plot(metrics_demo, metrics_feedback,
                          "No Feedback", "With Feedback",
                          output_dir / "comparison_feedback.png")
    
    # Create coupling visualization
    print("Creating coupling visualization...")
    create_coupling_visualization(output_dir)
    
    print(f"\nAll presentation plots saved to: {output_dir}")
    print("\nGenerated files:")
    print("  - time_series_demo.png")
    print("  - time_series_feedback.png")
    print("  - comparison_feedback.png")
    print("  - coupling_mechanism.png")


if __name__ == "__main__":
    main()

