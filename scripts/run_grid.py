#!/usr/bin/env python3
"""Run Conway-Izhikevich grid simulation."""

import argparse
import sys
from pathlib import Path

# Add src to path for imports (works from any directory)
_project_root = Path(__file__).parent.parent
_src_path = _project_root / "src"
if str(_src_path) not in sys.path:
    sys.path.insert(0, str(_src_path))

from conway_izh.grid import NeuralGrid
from conway_izh.config import (
    SimulationConfig,
    IzhikevichParams,
    CouplingParams,
    MemoryParams,
    StrategyParams,
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run Conway Game of Life + Izhikevich neuron hybrid simulation"
    )
    
    # Grid dimensions
    parser.add_argument("--height", type=int, default=60,
                       help="Grid height (default: 60)")
    parser.add_argument("--width", type=int, default=60,
                       help="Grid width (default: 60)")
    
    # Simulation parameters
    parser.add_argument("--steps", type=int, default=300,
                       help="Number of simulation steps (default: 300)")
    parser.add_argument("--seed", type=int, default=None,
                       help="Random seed (default: None)")
    
    # Output settings
    parser.add_argument("--out", type=str, default="outputs",
                       help="Output directory (default: outputs)")
    parser.add_argument("--run-id", type=str, default=None,
                       help="Run ID for output subdirectory (default: auto-generated)")
    parser.add_argument("--gif", action="store_true",
                       help="Generate GIF animation")
    parser.add_argument("--frame-stride", type=int, default=5,
                       help="Save every N frames for GIF (default: 5)")
    
    # Conway parameters
    parser.add_argument("--wrap", action="store_true",
                       help="Enable wrap-around boundaries")
    
    # Coupling parameters
    parser.add_argument("--feedback", action="store_true",
                       help="Enable neuron->GoL feedback")
    parser.add_argument("--k-neighbors", type=float, default=0.5,
                       help="Coupling weight for neighbor count (default: 0.5)")
    parser.add_argument("--k-alive", type=float, default=2.0,
                       help="Coupling weight for alive cells (default: 2.0)")
    parser.add_argument("--bias", type=float, default=0.0,
                       help="Constant bias current (default: 0.0)")
    
    # Game theory parameters
    parser.add_argument("--game-theory", action="store_true",
                       help="Use game theory based spike generation (Conway rules)")
    parser.add_argument("--propagation-strength", type=float, default=0.5,
                       help="Spike propagation strength (0-1, default: 0.5)")
    parser.add_argument("--cooperation-factor", type=float, default=0.3,
                       help="Cooperation factor for spike decision (default: 0.3)")
    parser.add_argument("--cooperation-strength", type=float, default=0.7,
                       help="Feedback cooperation strength (default: 0.7)")

    # Memory and strategy parameters
    parser.add_argument("--memory-enabled", action="store_true",
                       help="Enable PDA-like per-cell memory dynamics")
    parser.add_argument("--memory-decay", type=float, default=0.92,
                       help="Memory decay factor (default: 0.92)")
    parser.add_argument("--memory-spike-gain", type=float, default=0.35,
                       help="Spike contribution to memory (default: 0.35)")
    parser.add_argument("--memory-neighbor-gain", type=float, default=0.08,
                       help="Neighbor contribution to memory (default: 0.08)")
    parser.add_argument("--cellwise-strategy", action="store_true",
                       help="Enable cell-wise greedy/cooperative strategy selection")
    parser.add_argument("--strategy-temperature", type=float, default=6.0,
                       help="Strategy soft decision temperature (default: 6.0)")
    parser.add_argument("--strategy-switch-cost", type=float, default=0.03,
                       help="Penalty for switching strategy (default: 0.03)")
    
    # Izhikevich parameters
    parser.add_argument("--izh-a", type=float, default=0.02,
                       help="Izhikevich parameter a (default: 0.02)")
    parser.add_argument("--izh-b", type=float, default=0.2,
                       help="Izhikevich parameter b (default: 0.2)")
    parser.add_argument("--izh-c", type=float, default=-65.0,
                       help="Izhikevich parameter c (default: -65.0)")
    parser.add_argument("--izh-d", type=float, default=8.0,
                       help="Izhikevich parameter d (default: 8.0)")
    parser.add_argument("--dt", type=float, default=0.1,
                       help="Time step for neuron dynamics (default: 0.1)")
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    # Create configuration
    config = SimulationConfig(
        height=args.height,
        width=args.width,
        steps=args.steps,
        dt=args.dt,
        seed=args.seed,
        wrap_around=args.wrap,
        izh_params=IzhikevichParams(
            a=args.izh_a,
            b=args.izh_b,
            c=args.izh_c,
            d=args.izh_d
        ),
        coupling=CouplingParams(
            k_neighbors=args.k_neighbors,
            k_alive=args.k_alive,
            bias=args.bias,
            feedback_enabled=args.feedback,
            use_game_theory=args.game_theory,
            propagation_strength=args.propagation_strength,
            cooperation_factor=args.cooperation_factor,
            cooperation_strength=args.cooperation_strength
        ),
        memory=MemoryParams(
            enabled=(args.memory_enabled or args.game_theory),
            decay=args.memory_decay,
            spike_gain=args.memory_spike_gain,
            neighbor_gain=args.memory_neighbor_gain
        ),
        strategy=StrategyParams(
            cellwise_enabled=(args.cellwise_strategy or args.game_theory),
            temperature=args.strategy_temperature,
            switch_cost=args.strategy_switch_cost
        ),
        output_dir=args.out,
        run_id=args.run_id,
        save_gif=args.gif,
        frame_stride=args.frame_stride
    )
    
    # Create and run simulation
    grid = NeuralGrid(config)
    grid.run()
    
    print("Done!")


if __name__ == "__main__":
    main()

