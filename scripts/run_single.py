#!/usr/bin/env python3
"""Run single Izhikevich neuron simulation."""

import argparse
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import csv

# Add src to path for imports (works from any directory)
_project_root = Path(__file__).parent.parent
_src_path = _project_root / "src"
if str(_src_path) not in sys.path:
    sys.path.insert(0, str(_src_path))

from conway_izh.izhikevich import initialize_izhikevich, step_izhikevich, IzhikevichParams


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run single Izhikevich neuron simulation"
    )
    
    parser.add_argument("--seed", type=int, default=None,
                       help="Random seed")
    parser.add_argument("--steps", type=int, default=1000,
                       help="Number of simulation steps (default: 1000)")
    parser.add_argument("--dt", type=float, default=0.1,
                       help="Time step (default: 0.1)")
    parser.add_argument("--I", type=float, default=10.0,
                       help="Constant input current (default: 10.0)")
    parser.add_argument("--izh-a", type=float, default=0.02,
                       help="Izhikevich parameter a (default: 0.02)")
    parser.add_argument("--izh-b", type=float, default=0.2,
                       help="Izhikevich parameter b (default: 0.2)")
    parser.add_argument("--izh-c", type=float, default=-65.0,
                       help="Izhikevich parameter c (default: -65.0)")
    parser.add_argument("--izh-d", type=float, default=8.0,
                       help="Izhikevich parameter d (default: 8.0)")
    parser.add_argument("--out", type=str, default="outputs",
                       help="Output directory (default: outputs)")
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    # Setup output directory
    output_dir = Path(args.out) / "single_neuron"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize neuron
    params = IzhikevichParams(
        a=args.izh_a,
        b=args.izh_b,
        c=args.izh_c,
        d=args.izh_d
    )
    
    v, u = initialize_izhikevich((1,), args.seed)
    v = v[0]  # Scalar
    u = u[0]  # Scalar
    
    # Constant input current
    I = np.array([args.I])
    
    # Run simulation
    v_history = []
    spike_times = []
    
    print(f"Running single neuron simulation: {args.steps} steps")
    print(f"I = {args.I}, dt = {args.dt}")
    
    for step in range(args.steps):
        v_new, u_new, spikes = step_izhikevich(
            np.array([v]), np.array([u]), I, params, args.dt
        )
        v = v_new[0]
        u = u_new[0]
        
        v_history.append(v)
        
        if spikes[0]:
            spike_times.append(step * args.dt)
    
    # Plot membrane potential
    time = np.arange(args.steps) * args.dt
    
    plt.figure(figsize=(12, 6))
    plt.plot(time, v_history, 'b-', linewidth=1)
    plt.axhline(y=30, color='r', linestyle='--', alpha=0.5, label='Spike threshold')
    plt.xlabel('Time (ms)')
    plt.ylabel('Membrane Potential (mV)')
    plt.title(f'Single Izhikevich Neuron (I={args.I})')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "single_v.png", dpi=150)
    plt.close()
    
    # Save spike times
    with open(output_dir / "single_spikes.csv", 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['spike_time_ms'])
        for t in spike_times:
            writer.writerow([t])
    
    print(f"Spikes detected: {len(spike_times)}")
    print(f"Outputs saved to: {output_dir}")
    print("Done!")


if __name__ == "__main__":
    main()

