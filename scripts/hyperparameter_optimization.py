#!/usr/bin/env python3
"""Hyperparameter optimization for Game Theory coupling."""

import argparse
import sys
import numpy as np
import csv
from pathlib import Path
import time
from itertools import product
import json
from typing import List, Dict

# Add src to path for imports
_project_root = Path(__file__).parent.parent
_src_path = _project_root / "src"
if str(_src_path) not in sys.path:
    sys.path.insert(0, str(_src_path))

from conway_izh.grid import NeuralGrid
from conway_izh.config import SimulationConfig, IzhikevichParams, CouplingParams


def run_simulation(config: SimulationConfig, steps: int = 50) -> dict:
    """
    Run a single simulation and return metrics.
    
    Args:
        config: Simulation configuration
        steps: Number of steps to run (reduced for optimization)
        
    Returns:
        Dictionary with key metrics
    """
    # Temporarily override steps
    original_steps = config.steps
    config.steps = steps
    
    try:
        grid = NeuralGrid(config)
        
        # Run simulation
        alive_history = []
        spike_history = []
        firing_rate_history = []
        
        for _ in range(steps):
            metrics, spikes = grid.step()
            alive_history.append(metrics['alive_count'])
            spike_history.append(metrics['spike_count'])
            firing_rate_history.append(metrics['firing_rate'])
        
        # Calculate aggregate metrics
        total_cells = config.height * config.width
        final_alive = alive_history[-1]
        total_spikes = sum(spike_history)
        avg_firing_rate = np.mean(firing_rate_history)
        max_firing_rate = max(firing_rate_history)
        
        # Stability metrics (variance)
        alive_variance = np.var(alive_history)
        spike_variance = np.var(spike_history)
        
        # Activity metrics
        spike_activity = total_spikes / (steps * total_cells)
        alive_ratio = final_alive / total_cells
        
        return {
            'final_alive': final_alive,
            'total_spikes': total_spikes,
            'avg_firing_rate': avg_firing_rate,
            'max_firing_rate': max_firing_rate,
            'alive_variance': alive_variance,
            'spike_variance': spike_variance,
            'spike_activity': spike_activity,
            'alive_ratio': alive_ratio,
            'stability_score': 1.0 / (1.0 + alive_variance / 1000.0),  # Higher = more stable
            'activity_score': spike_activity * avg_firing_rate,  # Combined activity metric
        }
    finally:
        config.steps = original_steps


def grid_search_optimization(
    height: int = 30,
    width: int = 30,
    steps: int = 50,
    seed: int = 42
) -> List[Dict]:
    """
    Perform grid search over hyperparameter space.
    
    Optimizes:
    - propagation_strength: [0.3, 0.5, 0.7, 0.9]
    - cooperation_factor: [0.2, 0.3, 0.4, 0.5]
    - cooperation_strength: [0.5, 0.7, 0.9]
    - k_alive: [1.5, 2.0, 2.5, 3.0]
    
    Returns:
        DataFrame with all results
    """
    print("=" * 70)
    print("HYPERPARAMETER OPTIMIZATION - GRID SEARCH")
    print("=" * 70)
    print(f"Grid size: {height}x{width}")
    print(f"Steps per simulation: {steps}")
    print(f"Total combinations: 4 * 4 * 3 * 4 = 192")
    print("=" * 70)
    
    # Parameter ranges
    propagation_strengths = [0.3, 0.5, 0.7, 0.9]
    cooperation_factors = [0.2, 0.3, 0.4, 0.5]
    cooperation_strengths = [0.5, 0.7, 0.9]
    k_alives = [1.5, 2.0, 2.5, 3.0]
    
    results = []
    total_combinations = len(propagation_strengths) * len(cooperation_factors) * \
                        len(cooperation_strengths) * len(k_alives)
    current = 0
    
    start_time = time.time()
    
    for prop_str, coop_factor, coop_str, k_alive in product(
        propagation_strengths, cooperation_factors, cooperation_strengths, k_alives
    ):
        current += 1
        print(f"\n[{current}/{total_combinations}] Testing: "
              f"prop={prop_str:.1f}, coop_factor={coop_factor:.1f}, "
              f"coop_str={coop_str:.1f}, k_alive={k_alive:.1f}")
        
        config = SimulationConfig(
            height=height,
            width=width,
            steps=steps,
            seed=seed,
            coupling=CouplingParams(
                feedback_enabled=True,
                use_game_theory=True,
                propagation_strength=prop_str,
                cooperation_factor=coop_factor,
                cooperation_strength=coop_str,
                k_alive=k_alive,
                k_neighbors=0.5,
                bias=0.0
            )
        )
        
        try:
            metrics = run_simulation(config, steps)
            
            result = {
                'propagation_strength': prop_str,
                'cooperation_factor': coop_factor,
                'cooperation_strength': coop_str,
                'k_alive': k_alive,
                **metrics
            }
            results.append(result)
            
            print(f"  -> Alive: {metrics['final_alive']}, "
                  f"Spikes: {metrics['total_spikes']}, "
                  f"Firing: {metrics['avg_firing_rate']*100:.1f}%")
            
        except Exception as e:
            print(f"  X Error: {e}")
            continue
    
    elapsed_time = time.time() - start_time
    print(f"\n{'='*70}")
    print(f"Optimization completed in {elapsed_time:.2f} seconds")
    print(f"Average time per simulation: {elapsed_time/total_combinations:.3f} seconds")
    print(f"{'='*70}")
    
    return results


def analyze_results(results: List[Dict]) -> dict:
    """Analyze optimization results and find best parameters."""
    print("\n" + "=" * 70)
    print("RESULTS ANALYSIS")
    print("=" * 70)
    
    # Convert to list of dicts and calculate combined score
    for r in results:
        r['combined_score'] = r['activity_score'] * r['stability_score']
    
    # Find best parameters for different objectives
    best_activity = max(results, key=lambda x: x['activity_score'])
    best_stability = max(results, key=lambda x: x['stability_score'])
    best_combined = max(results, key=lambda x: x['combined_score'])
    best_spikes = max(results, key=lambda x: x['total_spikes'])
    best_alive = max(results, key=lambda x: x['final_alive'])
    
    print("\n1. BEST FOR ACTIVITY (spike activity * firing rate):")
    print(f"   Propagation: {best_activity['propagation_strength']:.2f}")
    print(f"   Cooperation Factor: {best_activity['cooperation_factor']:.2f}")
    print(f"   Cooperation Strength: {best_activity['cooperation_strength']:.2f}")
    print(f"   k_alive: {best_activity['k_alive']:.2f}")
    print(f"   Activity Score: {best_activity['activity_score']:.4f}")
    print(f"   Total Spikes: {best_activity['total_spikes']:.0f}")
    print(f"   Avg Firing Rate: {best_activity['avg_firing_rate']*100:.2f}%")
    
    print("\n2. BEST FOR STABILITY (low variance):")
    print(f"   Propagation: {best_stability['propagation_strength']:.2f}")
    print(f"   Cooperation Factor: {best_stability['cooperation_factor']:.2f}")
    print(f"   Cooperation Strength: {best_stability['cooperation_strength']:.2f}")
    print(f"   k_alive: {best_stability['k_alive']:.2f}")
    print(f"   Stability Score: {best_stability['stability_score']:.4f}")
    print(f"   Alive Variance: {best_stability['alive_variance']:.2f}")
    
    print("\n3. BEST COMBINED (activity * stability):")
    print(f"   Propagation: {best_combined['propagation_strength']:.2f}")
    print(f"   Cooperation Factor: {best_combined['cooperation_factor']:.2f}")
    print(f"   Cooperation Strength: {best_combined['cooperation_strength']:.2f}")
    print(f"   k_alive: {best_combined['k_alive']:.2f}")
    print(f"   Combined Score: {best_combined['combined_score']:.4f}")
    print(f"   Activity Score: {best_combined['activity_score']:.4f}")
    print(f"   Stability Score: {best_combined['stability_score']:.4f}")
    print(f"   Total Spikes: {best_combined['total_spikes']:.0f}")
    print(f"   Final Alive: {best_combined['final_alive']:.0f}")
    
    print("\n4. PARAMETER SENSITIVITY ANALYSIS:")
    # Calculate correlations manually
    combined_scores = [r['combined_score'] for r in results]
    for param in ['propagation_strength', 'cooperation_factor', 'cooperation_strength', 'k_alive']:
        param_values = [r[param] for r in results]
        correlation = np.corrcoef(param_values, combined_scores)[0, 1]
        print(f"   {param}: correlation = {correlation:.3f}")
    
    print("\n5. TOP 5 CONFIGURATIONS:")
    sorted_results = sorted(results, key=lambda x: x['combined_score'], reverse=True)
    top5 = sorted_results[:5]
    for idx, row in enumerate(top5, 1):
        print(f"   {idx}. Prop={row['propagation_strength']:.1f}, "
              f"CoopF={row['cooperation_factor']:.1f}, "
              f"CoopS={row['cooperation_strength']:.1f}, "
              f"k_alive={row['k_alive']:.1f} → Score={row['combined_score']:.4f}")
    
    return {
        'best_activity': best_activity,
        'best_stability': best_stability,
        'best_combined': best_combined,
        'top5': top5
    }


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Hyperparameter optimization")
    parser.add_argument("--height", type=int, default=30,
                       help="Grid height (default: 30, optimized for speed)")
    parser.add_argument("--width", type=int, default=30,
                       help="Grid width (default: 30)")
    parser.add_argument("--steps", type=int, default=50,
                       help="Steps per simulation (default: 50)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    parser.add_argument("--out", type=str, default="outputs/optimization",
                       help="Output directory")
    
    args = parser.parse_args()
    
    # Run optimization
    df = grid_search_optimization(
        height=args.height,
        width=args.width,
        steps=args.steps,
        seed=args.seed
    )
    
    # Analyze results
    analysis = analyze_results(df)
    
    # Save results
    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save results as CSV
    csv_path = output_dir / "optimization_results.csv"
    if results:
        fieldnames = list(results[0].keys())
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        print(f"\n[OK] Results saved to: {csv_path} ({len(results)} configurations)")
    
    # Save analysis
    json_path = output_dir / "best_parameters.json"
    with open(json_path, 'w') as f:
        json.dump(analysis, f, indent=2, default=str)
    print(f"[OK] Best parameters saved to: {json_path}")
    
    # Save summary
    summary_path = output_dir / "optimization_summary.txt"
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("HYPERPARAMETER OPTIMIZATION SUMMARY\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Grid size: {args.height}x{args.width}\n")
        f.write(f"Steps per simulation: {args.steps}\n")
        f.write(f"Total configurations tested: {len(results)}\n\n")
        
        best = analysis['best_combined']
        f.write("RECOMMENDED PARAMETERS (Best Combined Score):\n")
        f.write(f"  propagation_strength: {best['propagation_strength']:.2f}\n")
        f.write(f"  cooperation_factor: {best['cooperation_factor']:.2f}\n")
        f.write(f"  cooperation_strength: {best['cooperation_strength']:.2f}\n")
        f.write(f"  k_alive: {best['k_alive']:.2f}\n")
        f.write(f"  k_neighbors: 0.5 (default)\n")
        f.write(f"  bias: 0.0 (default)\n\n")
        
        f.write(f"Expected Performance:\n")
        f.write(f"  Total Spikes: {best['total_spikes']:.0f}\n")
        f.write(f"  Final Alive: {best['final_alive']:.0f}\n")
        f.write(f"  Avg Firing Rate: {best['avg_firing_rate']*100:.2f}%\n")
        f.write(f"  Combined Score: {best['combined_score']:.4f}\n")
    
    print(f"[OK] Summary saved to: {summary_path}")
    print("\n" + "=" * 70)
    print("OPTIMIZATION COMPLETE!")
    print("=" * 70)


if __name__ == "__main__":
    main()

