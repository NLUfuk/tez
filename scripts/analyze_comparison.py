#!/usr/bin/env python3
"""Analyze comparison results and create detailed visualizations."""

import sys
import numpy as np
import matplotlib.pyplot as plt
import csv
from pathlib import Path
import json

# Add src to path for imports
_project_root = Path(__file__).parent.parent
_src_path = _project_root / "src"
if str(_src_path) not in sys.path:
    sys.path.insert(0, str(_src_path))


def load_comparison_data(analysis_dir: Path):
    """Load comparison analysis data."""
    csv_path = analysis_dir / "comparison_metrics.csv"
    json_path = analysis_dir / "comparison_summary.json"
    
    # Load CSV data
    data = {'step': [], 'standalone_spikes': [], 'integrated_spikes': [],
            'standalone_meanv': [], 'integrated_meanv': [], 'alive_count': [],
            'spike_diff': [], 'v_diff': []}
    
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data['step'].append(int(row['step']))
            data['standalone_spikes'].append(int(row['standalone_spikes']))
            data['integrated_spikes'].append(int(row['integrated_spikes']))
            data['standalone_meanv'].append(float(row['standalone_meanv']))
            data['integrated_meanv'].append(float(row['integrated_meanv']))
            data['alive_count'].append(int(row['alive_count']))
            data['spike_diff'].append(int(row['spike_diff']))
            data['v_diff'].append(float(row['v_diff']))
    
    # Convert to numpy arrays
    df = {k: np.array(v) for k, v in data.items()}
    
    with open(json_path, 'r') as f:
        summary = json.load(f)
    
    return df, summary


def create_detailed_analysis(analysis_dir: Path, output_dir: Path):
    """Create detailed analysis plots."""
    df, summary = load_comparison_data(analysis_dir)
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)
    
    # 1. Spike count over time
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(df['step'], df['standalone_spikes'], 'b-', linewidth=1.5, alpha=0.7, label='Standalone')
    ax1.plot(df['step'], df['integrated_spikes'], 'r-', linewidth=1.5, alpha=0.7, label='Integrated')
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Spike Count')
    ax1.set_title('Spike Count Over Time', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Spike count difference
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(df['step'], df['spike_diff'], 'purple', linewidth=1.5)
    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.3)
    ax2.set_xlabel('Step')
    ax2.set_ylabel('Spike Difference (Integrated - Standalone)')
    ax2.set_title('Spike Count Difference', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # 3. Membrane potential comparison
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(df['step'], df['standalone_meanv'], 'b-', linewidth=1.5, alpha=0.7, label='Standalone')
    ax3.plot(df['step'], df['integrated_meanv'], 'r-', linewidth=1.5, alpha=0.7, label='Integrated')
    ax3.axhline(y=30, color='orange', linestyle='--', alpha=0.5, label='Threshold')
    ax3.set_xlabel('Step')
    ax3.set_ylabel('Mean V (mV)')
    ax3.set_title('Mean Membrane Potential', fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. V difference
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.plot(df['step'], df['v_diff'], 'purple', linewidth=1.5)
    ax4.axhline(y=0, color='black', linestyle='--', alpha=0.3)
    ax4.set_xlabel('Step')
    ax4.set_ylabel('V Difference (mV)')
    ax4.set_title('Membrane Potential Difference', fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    # 5. Alive cells over time
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.plot(df['step'], df['alive_count'], 'g-', linewidth=1.5)
    ax5.set_xlabel('Step')
    ax5.set_ylabel('Alive Cells')
    ax5.set_title('Conway: Alive Cells Over Time', fontweight='bold')
    ax5.grid(True, alpha=0.3)
    
    # 6. Spike distribution histogram
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.hist(df['standalone_spikes'], bins=50, alpha=0.6, label='Standalone', color='blue', density=True)
    ax6.hist(df['integrated_spikes'], bins=50, alpha=0.6, label='Integrated', color='red', density=True)
    ax6.set_xlabel('Spikes per Step')
    ax6.set_ylabel('Density')
    ax6.set_title('Spike Distribution', fontweight='bold')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    # 7. Cumulative spikes
    ax7 = fig.add_subplot(gs[2, 0])
    ax7.plot(df['step'], np.cumsum(df['standalone_spikes']), 'b-', linewidth=2, label='Standalone')
    ax7.plot(df['step'], np.cumsum(df['integrated_spikes']), 'r-', linewidth=2, label='Integrated')
    ax7.set_xlabel('Step')
    ax7.set_ylabel('Cumulative Spikes')
    ax7.set_title('Cumulative Spike Count', fontweight='bold')
    ax7.legend()
    ax7.grid(True, alpha=0.3)
    
    # 8. V distribution histogram
    ax8 = fig.add_subplot(gs[2, 1])
    ax8.hist(df['standalone_meanv'], bins=50, alpha=0.6, label='Standalone', color='blue', density=True)
    ax8.hist(df['integrated_meanv'], bins=50, alpha=0.6, label='Integrated', color='red', density=True)
    ax8.set_xlabel('Mean V (mV)')
    ax8.set_ylabel('Density')
    ax8.set_title('Membrane Potential Distribution', fontweight='bold')
    ax8.legend()
    ax8.grid(True, alpha=0.3)
    
    # 9. Statistics text
    ax9 = fig.add_subplot(gs[2, 2])
    ax9.axis('off')
    
    stats_text = f"""
OPTIMIZATION ANALYSIS SUMMARY

STANDALONE MODEL:
  Total spikes: {summary['standalone']['total_spikes']}
  Avg spikes/step: {summary['standalone']['avg_spikes_per_step']:.2f} ± {summary['standalone']['std_spikes_per_step']:.2f}
  Avg V: {summary['standalone']['avg_v']:.2f} ± {summary['standalone']['std_v']:.2f} mV
  V range: [{summary['standalone']['min_v']:.2f}, {summary['standalone']['max_v']:.2f}] mV

INTEGRATED MODEL:
  Total spikes: {summary['integrated']['total_spikes']}
  Avg spikes/step: {summary['integrated']['avg_spikes_per_step']:.2f} ± {summary['integrated']['std_spikes_per_step']:.2f}
  Avg V: {summary['integrated']['avg_v']:.2f} ± {summary['integrated']['std_v']:.2f} mV
  V range: [{summary['integrated']['min_v']:.2f}, {summary['integrated']['max_v']:.2f}] mV
  Avg alive cells: {summary['integrated']['avg_alive_cells']:.1f} ± {summary['integrated']['std_alive_cells']:.1f}

KEY OPTIMIZATIONS:
  Spike variance reduction: {summary['standalone']['std_spikes_per_step'] / summary['integrated']['std_spikes_per_step']:.2f}×
  V variance reduction: {summary['standalone']['std_v'] / summary['integrated']['std_v']:.2f}×
  V range reduction: {(summary['standalone']['max_v'] - summary['standalone']['min_v']) / (summary['integrated']['max_v'] - summary['integrated']['min_v']):.2f}×
"""
    ax9.text(0.1, 0.5, stats_text, fontsize=10, verticalalignment='center',
             family='monospace', transform=ax9.transAxes)
    
    plt.suptitle('Detailed Comparison Analysis', fontsize=16, fontweight='bold', y=0.995)
    
    # Save figure
    output_path = output_dir / "detailed_analysis.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Detailed analysis plot saved to: {output_path}")
    plt.close()


def main():
    """Main entry point."""
    analysis_dir = Path("outputs/comparison_analysis")
    output_dir = analysis_dir
    
    if not analysis_dir.exists():
        print(f"Error: Analysis directory not found: {analysis_dir}")
        print("Please run compare_izhikevich.py with --save first.")
        return
    
    print("Creating detailed analysis...")
    create_detailed_analysis(analysis_dir, output_dir)
    print("Analysis complete!")


if __name__ == "__main__":
    main()

